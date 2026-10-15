"""Pydantic schemas for API request and response data validation.

Pydantic models define the shape, types, and constraints of data passing through
FastAPI endpoints. They automatically handle:
1. Data parsing & type casting.
2. Request validation and error generation (HTTP 422 if invalid).
3. Response serialization and OpenAPI schema documentation.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for service health check response."""

    status: str = Field(
        default="ok",
        description="Current operational status of the service (e.g. 'ok').",
        examples=["ok"],
    )
