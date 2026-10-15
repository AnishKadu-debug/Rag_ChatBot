"""FastAPI main application entry point.

This file sets up the FastAPI application instance, registers middleware (like CORS),
and defines the core endpoints.

In Phase 0, we provide:
- GET /health: Public health check endpoint returning {"status": "ok"}.
- Interactive OpenAPI documentation automatically available at /docs and /redoc.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import HealthResponse

# Load application settings from config module
settings = get_settings()

# Initialize the FastAPI application instance
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS)
# Allows the frontend chat UI (running on localhost or webview) to make requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Check the health status of the API service.

    This endpoint requires no authentication and is used by monitoring tools,
    hosting platforms, and clients to verify that the server is running.
    """
    return HealthResponse(status="ok")
