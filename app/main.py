"""FastAPI main application entry point.

This module exposes the RAG chatbot API:
- GET /health: Public service health check (no authentication).
- POST /api/v1/upload: Upload, validate, chunk, embed, and index a PDF document.
- POST /api/v1/chat: Ask grounded questions on uploaded documents (or fallback to general assistance).
- GET /api/v1/documents: List all registered documents belonging to a user.
- DELETE /api/v1/documents/{doc_id}: Purge a document's vector embeddings and registry metadata.

Security & Multi-Tenancy:
- All non-health endpoints require the 'X-API-Key' header.
- Multi-tenancy is enforced on all operations: users can only access and delete their own documents.
"""

import logging
from typing import List
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Path, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.chunker import chunk_pages
from app.config import get_settings
from app.document_registry import DocumentRegistry, get_document_registry
from app.embedder import E5Embedder, get_embedder
from app.llm import BaseLLMClient, get_llm_client
from app.pdf_loader import extract_text_from_pdf
from app.rag import answer_question
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteDocumentResponse,
    DocumentItem,
    HealthResponse,
    SourceSnippetItem,
    UploadResponse,
)
from app.security import validate_pdf_upload, verify_api_key
from app.vector_store import ChromaVectorStore, get_vector_store

logger = logging.getLogger(__name__)

settings = get_settings()

# Initialize FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection helpers (can be cleanly overridden in unit tests)
def get_registry_dep() -> DocumentRegistry:
    return get_document_registry()


def get_vector_store_dep() -> ChromaVectorStore:
    return get_vector_store()


def get_embedder_dep() -> E5Embedder:
    return get_embedder()


def get_llm_dep() -> BaseLLMClient:
    return get_llm_client()


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Check the operational health of the service.

    This public endpoint requires no authentication and is used by monitoring
    tools, cloud hosting platforms, and container health checks.
    """
    return HealthResponse(status="ok")


@app.post(
    "/api/v1/upload",
    response_model=UploadResponse,
    summary="Upload and Index PDF Document",
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document file to upload"),
    user_id: str = Form(..., description="Tenant user identifier"),
    _api_key: str = Depends(verify_api_key),
    registry: DocumentRegistry = Depends(get_registry_dep),
    vector_store: ChromaVectorStore = Depends(get_vector_store_dep),
    embedder: E5Embedder = Depends(get_embedder_dep),
) -> UploadResponse:
    """Validate, extract, chunk, embed, and store a PDF document under the user's account."""
    clean_user_id = user_id.strip()
    if not clean_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id cannot be empty.",
        )

    # 1. Validate the uploaded file for signature, size, and page limits
    pdf_bytes = await validate_pdf_upload(file)

    # 2. Extract page-by-page text
    try:
        pages = extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        logger.error("Failed to extract text from PDF: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extract text from PDF document.",
        ) from exc

    # 3. Generate unique document identifier
    doc_id = f"d_{uuid.uuid4().hex[:12]}"
    filename = file.filename or "document.pdf"

    # 4. Chunk text while strictly preserving 1-based page metadata
    chunks = chunk_pages(
        pages=pages,
        doc_id=doc_id,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    # 5. Compute vector embeddings and persist in ChromaDB & SQLite registry
    try:
        if chunks:
            chunk_texts = [c.text for c in chunks]
            embeddings = embedder.embed_passages(chunk_texts)
            vector_store.add_chunks(
                user_id=clean_user_id,
                chunks=chunks,
                embeddings=embeddings,
                filename=filename,
            )

        # Record document metadata in SQLite registry
        registry.add_document(
            doc_id=doc_id,
            user_id=clean_user_id,
            filename=filename,
            pages=len(pages),
            ocr_used=False,
        )
    except Exception as exc:
        logger.error("Error storing chunks or document metadata for %s: %s", doc_id, exc)
        # Roll back: clean up any partially added vectors or registry entries
        try:
            vector_store.delete_document(user_id=clean_user_id, doc_id=doc_id)
            registry.delete_document(doc_id=doc_id, user_id=clean_user_id)
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while indexing the document.",
        ) from exc

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        pages=len(pages),
        chunks=len(chunks),
        ocr_used=False,
    )


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    summary="Chat / Ask Document Questions",
    tags=["Chat"],
)
def chat(
    request: ChatRequest,
    _api_key: str = Depends(verify_api_key),
    registry: DocumentRegistry = Depends(get_registry_dep),
    vector_store: ChromaVectorStore = Depends(get_vector_store_dep),
    embedder: E5Embedder = Depends(get_embedder_dep),
    llm: BaseLLMClient = Depends(get_llm_dep),
) -> ChatResponse:
    """Ask a question about an uploaded document or receive general fallback assistance."""
    clean_user_id = request.user_id.strip()
    if not clean_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id cannot be empty.",
        )

    # If a specific doc_id is requested, verify it exists and is owned by this user
    if request.doc_id:
        doc_record = registry.get_document(
            doc_id=request.doc_id,
            user_id=clean_user_id,
        )
        if not doc_record:
            # We return 404 to avoid leaking whether doc_id exists under another user
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

    # Execute the core RAG pipeline flow
    response = answer_question(
        question=request.question,
        user_id=clean_user_id,
        doc_id=request.doc_id,
        vector_store=vector_store,
        embedder=embedder,
        llm=llm,
    )

    return ChatResponse(
        answer=response.answer,
        mode=response.mode,
        sources=[
            SourceSnippetItem(page=s.page, snippet=s.snippet)
            for s in response.sources
        ],
    )


@app.get(
    "/api/v1/documents",
    response_model=List[DocumentItem],
    summary="List User Documents",
    tags=["Documents"],
)
def list_documents(
    user_id: str = Query(..., min_length=1, description="Tenant user identifier"),
    _api_key: str = Depends(verify_api_key),
    registry: DocumentRegistry = Depends(get_registry_dep),
) -> List[DocumentItem]:
    """Retrieve metadata for all documents uploaded by the requesting user."""
    clean_user_id = user_id.strip()
    docs = registry.list_documents(user_id=clean_user_id)
    return [
        DocumentItem(
            doc_id=d.doc_id,
            filename=d.filename,
            pages=d.pages,
            uploaded_at=d.uploaded_at,
            ocr_used=d.ocr_used,
        )
        for d in docs
    ]


@app.delete(
    "/api/v1/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete Document",
    tags=["Documents"],
)
def delete_document(
    doc_id: str = Path(..., description="Identifier of the document to delete"),
    user_id: str = Query(..., min_length=1, description="Tenant user identifier"),
    _api_key: str = Depends(verify_api_key),
    registry: DocumentRegistry = Depends(get_registry_dep),
    vector_store: ChromaVectorStore = Depends(get_vector_store_dep),
) -> DeleteDocumentResponse:
    """Delete a document, purging its SQLite registry entry and ChromaDB vector embeddings."""
    clean_user_id = user_id.strip()

    # Enforce ownership check before deleting
    doc_record = registry.get_document(doc_id=doc_id, user_id=clean_user_id)
    if not doc_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    # 1. Purge vector chunks from ChromaDB
    vector_store.delete_document(user_id=clean_user_id, doc_id=doc_id)

    # 2. Delete metadata record from SQLite registry
    registry.delete_document(doc_id=doc_id, user_id=clean_user_id)

    return DeleteDocumentResponse(
        status="success",
        message="Document and associated vectors deleted successfully.",
        doc_id=doc_id,
    )
