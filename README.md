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
Chat UI (Standalone mobile-first web page)
       │  HTTPS + JSON
       ▼
FastAPI Backend
       ├── PDF Loader      (PyMuPDF, OCR fallback with Tesseract)
       ├── Chunker         (Page-aware character chunking)
       ├── Embedder        (multilingual-e5 via sentence-transformers)
       ├── Vector Store    (ChromaDB persistent on disk)
       └── LLM Client      (Provider-agnostic wrapper, default Gemini)
```

---

## Project Structure

```
WhatsAppRag/
├── Project.md            # Master project specification
├── README.md             # Project documentation and guide
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules (secrets, venv, pdfs, chroma)
├── requirements.txt      # Pinned Python package dependencies
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application entry point & routes
│   ├── config.py         # Typed settings loaded from .env
│   └── schemas.py        # Pydantic models for validation
├── web/                  # Frontend UI files (Phase 5)
├── data/
│   ├── sample_pdfs/      # Test land record documents (PDFs gitignored)
│   └── chroma/           # Persistent ChromaDB vector data (gitignored)
├── eval/                 # Evaluation dataset and scripts (Phase 4)
├── notebooks/            # Jupyter/Colab experimentation notebooks
└── tests/
    ├── __init__.py
    └── test_health.py    # Health and documentation endpoint tests
```

---

## Getting Started (Phase 0 Setup)

### 1. Prerequisites
- **Python**: 3.14.7 (or Python 3.11+)
- **Git** / Git Bash on Windows

### 2. Environment Setup

#### In Windows Git Bash:
```bash
# Clone or navigate to the project directory
cd /c/Users/Anish/projects/WhatsAppRag

# Verify Python version
python --version

# Activate the virtual environment (.venv)
source .venv/Scripts/activate

# Create local environment file from template
cp .env.example .env
```

#### In Windows PowerShell:
```powershell
cd C:\Users\Anish\projects\WhatsAppRag
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Development Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 5. Verify the Endpoints
- **Health Check**: Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) in your browser. Expected response:
  ```json
  {"status": "ok"}
  ```
- **Interactive API Documentation (Swagger UI)**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Alternative Documentation**: Open [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 6. Run the Automated Tests
```bash
pytest -v
```

---

## Implementation Roadmap

- [x] **Phase 0: Setup** — Project structure, virtual environment, dependencies, `.env`, FastAPI `/health` endpoint and test validation.
- [ ] **Phase 1: Core RAG in Scripts** — Text extraction, chunking with page metadata, multilingual embeddings, ChromaDB indexing, grounded prompt generation, and fallback logic.
- [ ] **Phase 2: FastAPI Service** — Full REST API endpoints (`/upload`, `/chat`, `/documents`), authentication, file validation, and per-user data isolation.
- [ ] **Phase 3: OCR and Marathi Support** — Tesseract OCR fallback for scanned land records and multilingual testing.
- [ ] **Phase 4: Evaluation and Tuning** — Automated evaluation with `eval/questions.json` and retrieval tuning.
- [ ] **Phase 5: Chat UI** — Mobile-first webview-compatible chat interface.
- [ ] **Phase 6: Deployment & Handover** — Production deployment and integration documentation.