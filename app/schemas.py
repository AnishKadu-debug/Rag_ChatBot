"""Pydantic schemas for API request and response data validation.

Pydantic models define the shape, types, constraints, and OpenAPI documentation
for all data passed across HTTP endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for service health check response."""

    status: str = Field(
        default="ok",
        description="Current operational status of the service (e.g. 'ok').",
        examples=["ok"],
    )


class UploadResponse(BaseModel):
    """Schema for successful PDF upload and indexing response."""

    doc_id: str = Field(
        ...,
        description="Unique identifier assigned to the uploaded document.",
        examples=["d_8f3a9b1c2d3e"],
    )
    filename: str = Field(
        ...,
        description="Original name of the uploaded PDF file.",
        examples=["7_12_extract.pdf"],
    )
    pages: int = Field(
        ...,
        description="Total physical pages found in the document.",
        examples=[2],
    )
    chunks: int = Field(
        ...,
        description="Total text chunks generated and stored in the vector database.",
        examples=[6],
    )
    ocr_used: bool = Field(
        default=False,
        description="Flag indicating whether OCR was applied to extract text.",
        examples=[False],
    )


class ChatMessage(BaseModel):
    """Represents a single message in the conversation history."""

    role: str = Field(
        ...,
        description="Sender role ('user' or 'assistant').",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="Content of the message.",
        examples=["Who is the owner?"],
    )


class ChatRequest(BaseModel):
    """Schema for incoming user chat/question requests."""

    user_id: str = Field(
        ...,
        min_length=1,
        description="Tenant user identifier for multi-tenant data isolation.",
        examples=["u123"],
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Optional target document ID. If omitted, assistant answers in general fallback mode.",
        examples=["d_8f3a9b1c2d3e"],
    )
    question: str = Field(
        ...,
        min_length=1,
        description="User question to be answered.",
        examples=["Who is the owner of survey number 45?"],
    )
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Recent conversation history turns for multi-turn dialogue context.",
    )


class SourceSnippetItem(BaseModel):
    """Schema for document citations supporting an answer."""

    page: int = Field(
        ...,
        description="1-based page number where the information was located.",
        examples=[1],
    )
    snippet: str = Field(
        ...,
        description="Short text excerpt from the document providing the evidence.",
        examples=["Ramesh Vishnu Patil is listed as the primary occupant..."],
    )


class ChatResponse(BaseModel):
    """Schema for chatbot response."""

    answer: str = Field(
        ...,
        description="Generated answer from the grounded document context or general assistant.",
        examples=["The owner of the land is Ramesh Vishnu Patil (Page 1)."],
    )
    mode: str = Field(
        ...,
        description="Mode used to answer: 'document' (grounded from PDF) or 'general' (fallback).",
        examples=["document"],
    )
    sources: List[SourceSnippetItem] = Field(
        default_factory=list,
        description="List of cited source excerpts. Empty when mode is 'general'.",
    )


class DocumentItem(BaseModel):
    """Schema for metadata of a single registered document."""

    doc_id: str = Field(
        ...,
        description="Unique document identifier.",
        examples=["d_8f3a9b1c2d3e"],
    )
    filename: str = Field(
        ...,
        description="Original filename.",
        examples=["7_12_extract.pdf"],
    )
    pages: int = Field(
        ...,
        description="Total pages in the document.",
        examples=[2],
    )
    uploaded_at: str = Field(
        ...,
        description="ISO-8601 formatted timestamp of upload.",
        examples=["2026-10-15T12:00:00Z"],
    )
    ocr_used: bool = Field(
        default=False,
        description="Whether OCR was used during extraction.",
        examples=[False],
    )


class DeleteDocumentResponse(BaseModel):
    """Schema for document deletion response."""

    status: str = Field(
        default="success",
        description="Status of the deletion operation.",
        examples=["success"],
    )
    message: str = Field(
        ...,
        description="Human-readable description of the result.",
        examples=["Document and associated vectors deleted successfully."],
    )
    doc_id: str = Field(
        ...,
        description="ID of the deleted document.",
        examples=["d_8f3a9b1c2d3e"],
    )
