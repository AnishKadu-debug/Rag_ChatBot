# WhatsApp Land Records RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) document chatbot designed for a WhatsApp bot service page handling Maharashtra land records (Digital 7/12, 8A Holding, Ferfar / Mutation, Property Card, e-Mojni).

Users can upload land record documents (PDF format, English or Marathi) and ask questions to receive accurate, grounded answers citing specific pages and text snippets.

> **Single Source of Truth**: All requirements, architecture specifications, API contracts, security rules, and implementation phases are defined in [`Project.md`](Project.md).

---

## Architecture Overview

```
WhatsApp Webview Page (Host Bot)
       │  (Chat icon links to chat UI with ?user_id=...)
       ▼
Chat UI (Standalone mobile-first web page, Phase 5)
       │  HTTPS + JSON (X-API-Key header)
       ▼
FastAPI Backend (Phase 2)
       ├── Security & Auth (X-API-Key, PDF magic-bytes & size validation)
       ├── Document Registry (SQLite metadata store)
       ├── PDF Loader      (PyMuPDF, OCR fallback with Tesseract)
       ├── Chunker         (Page-aware character chunking)
       ├── Embedder        (multilingual-e5 via sentence-transformers)
       ├── Vector Store    (ChromaDB persistent on disk, cosine distance)
       └── LLM Client      (Provider-agnostic wrapper, default Gemini)
```

---

## Project Structure

```
WhatsAppRag/
├── Project.md              # Master project specification (single source of truth)
├── README.md               # Project documentation and developer guide
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules (secrets, venv, pdfs, chroma, sqlite)
├── requirements.txt        # Pinned Python package dependencies
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI REST endpoints & dependency injection
│   ├── config.py           # Typed settings loaded from .env
│   ├── security.py         # API key auth & PDF file validation
│   ├── document_registry.py# SQLite persistent document metadata store
│   ├── schemas.py          # Pydantic request/response models
│   ├── pdf_loader.py       # Page-by-page PDF extraction
│   ├── chunker.py          # Page-preserving text chunking
│   ├── embedder.py         # multilingual-e5 embedder (passage: / query:)
│   ├── vector_store.py     # ChromaDB wrapper with multi-tenant isolation
│   ├── prompts.py          # Grounded document & fallback prompt templates
│   ├── llm.py              # Gemini & Mock LLM client wrappers
│   └── rag.py              # Retrieval & fallback coordinator
├── web/                    # Frontend UI files (Phase 5)
├── data/
│   ├── sample_pdfs/        # Test land record documents (gitignored)
│   ├── chroma/             # Persistent ChromaDB vector data (gitignored)
│   └── documents.sqlite3   # Document metadata SQLite database (gitignored)
├── eval/                   # Evaluation dataset and scripts (Phase 4)
├── notebooks/              # Experimentation scripts and demonstrations
└── tests/
    ├── __init__.py
    ├── test_health.py      # Health check tests
    ├── test_chunker.py     # Chunking & overlap tests
    ├── test_embedder.py    # E5 prefix & normalization tests
    ├── test_pdf_loader.py  # PDF text extraction tests
    ├── test_vector_store.py# ChromaDB multi-tenant isolation tests
    ├── test_rag.py         # RAG pipeline fallback & citations tests
    └── test_api.py         # Phase 2 FastAPI REST API tests
```

---

## Configuration & Environment Variables

Create your local `.env` file from the provided template:

```bash
cp .env.example .env
```

> **Security Rule**: Never commit `.env` to Git. The `.gitignore` file already excludes `.env`, `data/chroma/`, and `*.sqlite3`.

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `API_KEY` | Secret key required in the `X-API-Key` header for protected routes | `change-me` (set a strong random string) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins allowed to call the API | `http://localhost:3000,http://127.0.0.1:3000` |
| `LLM_PROVIDER` | LLM provider to use (`gemini`, `openai`, `anthropic`) | `gemini` |
| `LLM_MODEL` | Specific model identifier | `gemini-1.5-flash` |
| `GEMINI_API_KEY` | Google Gemini API key | *(your private Gemini API key)* |
| `EMBEDDING_MODEL` | Hugging Face model for embeddings | `intfloat/multilingual-e5-base` |
| `CHROMA_DIR` | Directory where ChromaDB vector data is saved | `./data/chroma` |
| `DOCS_DB_PATH` | Path to SQLite document registry database | `./data/documents.sqlite3` |
| `CHUNK_SIZE` | Target chunk size in characters | `500` |
| `CHUNK_OVERLAP` | Character overlap between consecutive chunks | `100` |
| `TOP_K` | Number of candidate chunks retrieved per query | `4` |
| `SIMILARITY_DISTANCE_THRESHOLD` | Cosine distance cutoff for document relevance | `0.45` |
| `MAX_UPLOAD_MB` | Maximum allowed PDF upload size in megabytes | `10` |
| `MAX_PAGES` | Maximum allowed page count per PDF upload | `30` |
| `OCR_LANGS` | Tesseract OCR languages (Phase 3) | `mar+eng` |

---

## Running the API Locally

### 1. In Windows Git Bash

```bash
# Navigate to the project directory
cd /c/Users/Anish/projects/WhatsAppRag

# Activate the virtual environment
source .venv/Scripts/activate

# Verify Python version (must be 3.14.7 or 3.11+)
python --version

# Run development server with auto-reload
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. In Windows PowerShell

```powershell
cd C:\Users\Anish\projects\WhatsAppRag
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## Interactive API Documentation (Swagger UI)

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser:

1. Click the **Authorize** button (green lock icon) at the top right.
2. In the `X-API-Key` field, enter the value set in your `.env` (e.g., `change-me` or your custom key).
3. Click **Authorize** then **Close**.
4. You can now test any endpoint directly from the browser UI.

Alternative ReDoc documentation is available at [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

---

## Example API Requests

### 1. Service Health Check (Public, No Auth)
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
**Response (200 OK):**
```json
{
  "status": "ok"
}
```

### 2. Upload and Index a PDF Document
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/upload" \
  -H "X-API-Key: change-me" \
  -F "file=@data/sample_pdfs/sample_land_record.pdf" \
  -F "user_id=user_alice"
```
**Response (200 OK):**
```json
{
  "doc_id": "d_8f3a9b1c2d3e",
  "filename": "sample_land_record.pdf",
  "pages": 2,
  "chunks": 4,
  "ocr_used": false
}
```

### 3. Ask a Grounded Question About the Document
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_alice",
    "doc_id": "d_8f3a9b1c2d3e",
    "question": "Who is the owner of Gat Number 45/2?"
  }'
```
**Response (200 OK):**
```json
{
  "answer": "The primary occupant and owner is Ramesh Vishnu Patil (Page 1).",
  "mode": "document",
  "sources": [
    {
      "page": 1,
      "snippet": "Digital 7/12 Extract... Gat Number: 45/2... Occupant: Ramesh Vishnu Patil..."
    }
  ]
}
```

### 4. General Assistant Chat (No Document Specified)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_alice",
    "question": "What is a 7/12 extract in Maharashtra?"
  }'
```
**Response (200 OK):**
```json
{
  "answer": "A 7/12 (Saat Bara) extract is an official land record document issued by the Revenue Department of Maharashtra containing details of agricultural land ownership, survey numbers, crops, and loans.",
  "mode": "general",
  "sources": []
}
```

### 5. List User Documents
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/documents?user_id=user_alice" \
  -H "X-API-Key: change-me"
```
**Response (200 OK):**
```json
[
  {
    "doc_id": "d_8f3a9b1c2d3e",
    "filename": "sample_land_record.pdf",
    "pages": 2,
    "uploaded_at": "2026-10-15T13:00:00+00:00",
    "ocr_used": false
  }
]
```

### 6. Delete a Document
```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/documents/d_8f3a9b1c2d3e?user_id=user_alice" \
  -H "X-API-Key: change-me"
```
**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Document and associated vectors deleted successfully.",
  "doc_id": "d_8f3a9b1c2d3e"
}
```

---

## Running the Automated Test Suite

All tests can be executed with pytest in the activated virtual environment:

```bash
# Run all unit and integration tests
pytest -v
```

**Test Breakdown:**
- `tests/test_api.py`: FastAPI endpoints, `X-API-Key` auth, PDF validation, user isolation, document listing and deletion (13 tests).
- `tests/test_health.py`: Health check and Swagger documentation accessibility (3 tests).
- `tests/test_chunker.py`: Character chunking and overlap without crossing page boundaries (5 tests).
- `tests/test_embedder.py`: Multilingual E5 prefixing and normalization (1 test).
- `tests/test_pdf_loader.py`: Text extraction and page index tracking (3 tests).
- `tests/test_vector_store.py`: ChromaDB storage, retrieval, and multi-tenant isolation (3 tests).
- `tests/test_rag.py`: RAG coordinator, distance threshold gatekeeper, and general fallback (4 tests).

Total: **32 passed**.

---

## Current Limitations

1. **Scanned Documents (OCR)**: Scanned image-based PDFs without digital text streams will yield 0 extracted characters until Tesseract OCR is integrated in Phase 3.
2. **Real Gemini LLM Usage**: For real Gemini answer generation, supply a valid `GEMINI_API_KEY` in `.env`. Automated tests use `MockLLMClient` and mock embeddings to run rapidly and deterministically without external API costs or network dependencies.
3. **Multi-Tenant Identity**: The API assumes the parent WhatsApp bot provides a verified `user_id` query/form parameter. Authentication protects the server endpoints from unauthorized callers, while `user_id` isolates tenant records.

---

## Implementation Roadmap

- [x] **Phase 0: Setup** — Project structure, virtual environment, dependencies, `.env`, FastAPI `/health` endpoint and test validation.
- [x] **Phase 1: Core RAG in Scripts** — Text extraction, chunking with page metadata, multilingual embeddings, ChromaDB indexing, grounded prompt generation, and fallback logic.
- [x] **Phase 2: FastAPI Service** — Full REST API endpoints (`/upload`, `/chat`, `/documents`, `/documents/{doc_id}`), `X-API-Key` security, PDF validation, SQLite metadata registry, and multi-tenant data isolation.
- [ ] **Phase 3: OCR and Marathi Support** — Tesseract OCR fallback for scanned land records and multilingual testing.
- [ ] **Phase 4: Evaluation and Tuning** — Automated evaluation with `eval/questions.json` and retrieval tuning.
- [ ] **Phase 5: Chat UI** — Mobile-first webview-compatible chat interface.
- [ ] **Phase 6: Deployment & Handover** — Production deployment and integration documentation.