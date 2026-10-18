# WhatsApp Land Records RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) document chatbot designed for a WhatsApp bot service page handling Maharashtra land records (Digital 7/12, 8A Holding, Ferfar / Mutation, Property Card, e-Mojni).

Users can upload land record documents (PDF format, English or Marathi, digital or scanned) and ask questions to receive accurate, grounded answers citing specific pages and text snippets.

> **Single Source of Truth**: All requirements, architecture specifications, API contracts, security rules, and implementation phases are defined in [`Project.md`](Project.md).

---

## Architecture Overview

```
WhatsApp Webview Page / Browser (Mobile & Desktop)
       │  (Accesses /chat/ with bilingual English | मराठी UI)
       ▼
Mobile-First Chat UI (Served at /chat/)
       │  HTTPS + JSON (English/Marathi query, language parameter)
       ▼
FastAPI Backend
       ├── Security & Auth (X-API-Key, admin PDF validation)
       ├── Official Document Registry (SQLite tracking active document & indexing status)
       ├── Admin Ingestion (CLI & REST: page-by-page extraction, batched embeddings, safe replacement)
       ├── PDF Loader      (PyMuPDF with Tesseract OCR fallback for scanned pages)
       ├── Chunker         (Page-aware character chunking & OCR flag preservation)
       ├── Embedder        (multilingual-e5 via sentence-transformers, batch-processed)
       ├── Vector Store    (ChromaDB persistent on disk, cosine distance, tenant-isolated)
       ├── Retrieval & Fallback Coordinator (Three-tier: Grounded -> General LLM fallback -> 503)
       └── LLM Client      (Provider-agnostic wrapper, default Gemini 3.8-flash)
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
│   ├── main.py             # FastAPI REST endpoints, /chat/ static mount & admin routes
│   ├── config.py           # Typed settings loaded from .env (admin limits & official doc)
│   ├── security.py         # API key auth & PDF file validation (standard & admin ceilings)
│   ├── document_registry.py# SQLite persistent registry with official flag & status tracking
│   ├── ingest_official.py  # Administrator CLI & ingestion service with batched embedding
│   ├── schemas.py          # Pydantic request/response models (language parameter)
│   ├── pdf_loader.py       # Page-by-page PDF extraction with OCR fallback
│   ├── ocr.py              # Tesseract auto-discovery & Devanagari image OCR
│   ├── chunker.py          # Page-preserving text chunking
│   ├── embedder.py         # multilingual-e5 embedder (batch-size supported)
│   ├── vector_store.py     # ChromaDB wrapper with multi-tenant isolation
│   ├── prompts.py          # Grounded document & general fallback prompt templates (EN/MR)
│   ├── llm.py              # Gemini & Mock LLM client wrappers
│   └── rag.py              # Three-tier retrieval & fallback coordinator
├── web/                    # Mobile-first bilingual Chat Frontend (served at /chat/)
│   ├── index.html          # HTML5 chat interface with English/Marathi selector
│   ├── style.css           # Mobile-responsive CSS (>=320px, safe-area insets, >=44px touch)
│   ├── app.js              # Vanilla JS app with i18n, localStorage persistence & autoscroll
│   └── README.md           # Frontend architecture and integration documentation
├── data/
│   ├── sample_pdfs/        # Test land record documents (gitignored)
│   ├── chroma/             # Persistent ChromaDB vector data (gitignored)
│   └── documents.sqlite3   # Document metadata SQLite database (gitignored)
├── eval/                   # Evaluation dataset and scripts (Phase 4)
├── notebooks/              # Experimentation scripts and demonstrations
└── tests/
    ├── __init__.py
    ├── test_health.py      # Health check tests (3 tests)
    ├── test_chunker.py     # Chunking & overlap tests (5 tests)
    ├── test_embedder.py    # E5 prefix & normalization tests (1 test)
    ├── test_pdf_loader.py  # PDF text extraction tests (3 tests)
    ├── test_vector_store.py# ChromaDB multi-tenant isolation tests (3 tests)
    ├── test_rag.py         # RAG pipeline fallback & citations tests (4 tests)
    ├── test_api.py         # FastAPI REST API tests (13 tests)
    ├── test_ocr.py         # OCR fallback & multilingual tests (10 tests)
    ├── test_eval.py        # Evaluation dataset and retrieval metric tests (13 tests)
    ├── test_frontend.py    # Frontend bilingual UI & responsive tests (21 tests)
    └── test_official_doc.py# 500-page batching, 3 fallback tiers & admin ingestion tests (11 tests)
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
| `LLM_PROVIDER` | LLM provider to use (`openrouter`, `grok`, `groq`, `gemini`, `mock`) | `groq` |
| `LLM_MODEL` | Specific model identifier for the active provider | `llama-3.3-70b-versatile` |
| `GROQ_API_KEY` | GroqCloud API key (not xAI Grok) | *(your private GroqCloud API key)* |
| `GROQ_BASE_URL` | GroqCloud OpenAI-compatible API base URL | `https://api.groq.com/openai/v1` |
| `GEMINI_API_KEY` | Google Gemini API key | *(your private Gemini API key)* |
| `EMBEDDING_MODEL` | Hugging Face model for embeddings | `intfloat/multilingual-e5-base` |
| `CHROMA_DIR` | Directory where ChromaDB vector data is saved | `./data/chroma` |
| `DOCS_DB_PATH` | Path to SQLite document registry database | `./data/documents.sqlite3` |
| `CHUNK_SIZE` | Target chunk size in characters | `500` |
| `CHUNK_OVERLAP` | Character overlap between consecutive chunks | `100` |
| `TOP_K` | Number of candidate chunks retrieved per query | `4` |
| `SIMILARITY_DISTANCE_THRESHOLD` | Cosine distance cutoff for document relevance | `0.45` |
| `MAX_UPLOAD_MB` | Maximum allowed PDF upload size in megabytes (end-user) | `10` |
| `MAX_PAGES` | Maximum allowed page count per PDF upload (end-user) | `30` |
| `OFFICIAL_DOCUMENT_ID` | Override ID of the active official document (or leave blank for auto-discovery) | `""` |
| `ADMIN_MAX_UPLOAD_MB` | Administrator ceiling for official PDF file size | `100` |
| `ADMIN_MAX_PAGES` | Administrator ceiling for official PDF page count | `1000` |
| `OCR_ENABLED` | Toggle OCR fallback on scanned pages | `true` |
| `OCR_LANGS` | Tesseract language codes for OCR | `mar+eng` |
| `OCR_THRESHOLD_CHARS` | Minimum text characters on a page to avoid OCR | `30` |
| `OCR_DPI` | Rendering resolution for image OCR | `300` |
| `TESSERACT_CMD` | Custom path to `tesseract.exe` (optional) | `C:\Program Files\Tesseract-OCR\tesseract.exe` |

---

## Tesseract OCR Setup on Windows (Phase 3)

For documents that contain scanned pages or photographs with little or no extractable digital text, the system invokes Tesseract OCR.

### 1. Install Tesseract OCR on Windows
1. Download the Windows installer (64-bit) from the official UB-Mannheim repository:  
   👉 [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer:
   - When asked to choose components, expand **"Additional script data"** and check **"Devanagari script"**.
   - Expand **"Additional language data"** and check **"Marathi"** and **"Hindi"**.
   - Default install path is `C:\Program Files\Tesseract-OCR`.
3. Add Tesseract to your Windows System PATH:
   - Open Windows Search -> "Environment Variables" -> Edit `Path` -> Add `C:\Program Files\Tesseract-OCR`.
   - Alternatively, configure `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in your `.env`.

### 2. Verify Language Packs
In Windows PowerShell or Git Bash, run:
```bash
tesseract --list-langs
```
**Expected Output:**
```text
List of installed languages (4):
eng
hin
mar
osd
```
If `mar` is missing, download `mar.traineddata` from the [tessdata_fast repository](https://github.com/tesseract-ocr/tessdata_fast/raw/main/mar.traineddata) and place it in `C:\Program Files\Tesseract-OCR\tessdata\`.

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
2. In the `X-API-Key` field, enter the value set in your `.env` (e.g., `change-me`).
3. Click **Authorize** then **Close**.
4. Test endpoints directly from the browser UI.

---

### Administrator Ingestion: 500-Page Official Land Record PDF

The chatbot operates on one official Maharashtra land-record PDF containing approximately 500 pages. End users cannot upload or delete documents; the administrator ingests the document beforehand.

### 1. Ingestion CLI Workflow (Recommended)

Run the CLI tool from your terminal with the virtual environment activated:

```bash
# Ingest official PDF with memory-safe chunk embedding batches (default batch size: 64)
python -m app.ingest_official path/to/official_land_record.pdf

# Custom batch size for memory-constrained environments
python -m app.ingest_official path/to/official_land_record.pdf --batch-size 32
```

**Key Features of the Ingestion Pipeline:**
- **Memory-Safe Streaming**: Extracts text page-by-page from PyMuPDF, chunks per page retaining 1-based page numbers, and embeds in batches of 64 chunks to prevent RAM spikes on large documents.
- **Atomic Replacement**: Previous official document chunks are cleanly purged from persistent ChromaDB before the new document is marked active, preventing stale or duplicate chunk retrieval.
- **Transactional Rollback**: Documents start in `"indexing"` status and transition to `"ready"` only upon complete indexing. If indexing fails, status is marked `"failed"` and incomplete vectors are purged.

### 2. Administrator REST API Endpoint

Administrators can also ingest or replace the official PDF via HTTP:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/admin/upload-official" \
  -H "X-API-Key: change-me" \
  -F "file=@path/to/official_land_record.pdf"
```

**Response (200 OK):**
```json
{
  "doc_id": "d_e3b0c44298fc",
  "filename": "official_land_record.pdf",
  "pages": 500,
  "chunks": 980,
  "ocr_used": false,
  "is_official": true,
  "status": "ready"
}
```

---

## Three-Tier Retrieval & Fallback Behavior

End users do not need to supply a `doc_id`. The backend automatically binds all incoming queries to the active official document:

### Tier 1: Grounded Answer with Source Citations (Answer present in official PDF)
When relevant chunks are retrieved below the cosine distance threshold:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "गट क्रमांक ४५/२ चे खातेदार कोण आहेत?",
    "language": "mr"
  }'
```
**Response:**
```json
{
  "answer": "गट क्रमांक ४५/२ चे खातेदार रमेश विष्णू पाटील आहेत (Page 1).",
  "mode": "document",
  "sources": [
    {"page": 1, "snippet": "गाव नमुना सात... गट क्रमांक: ४५/२... खातेदार: रमेश विष्णू पाटील..."}
  ]
}
```

### Tier 2: General Answer without Citations (Information absent from official PDF)
When the question is not answerable from the document, the configured Gemini LLM provides a helpful general response in the selected language. No PDF page numbers or source snippets are shown:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the process to apply for agricultural land mutation in Maharashtra?",
    "language": "en"
  }'
```
**Response:**
```json
{
  "answer": "[Note: This information is not found in the official land record document. Providing general information.]\n\nTo apply for agricultural land mutation (Ferfar) in Maharashtra...",
  "mode": "general",
  "sources": []
}
```

### Tier 3: Service Unavailable (Official PDF missing or indexing incomplete)
If no official document has been ingested, or if indexing is in progress or failed, the API immediately returns HTTP 503:
```json
{
  "detail": "The official Maharashtra land records document is currently not indexed or unavailable. Please contact the administrator to ingest the official document."
}
```

---

## Mobile-First Bilingual Chat Interface

The frontend is served directly by FastAPI at **`http://127.0.0.1:8000/chat/`**:

1. **English Default**: First-time visitors are presented with an English interface.
2. **Language Toggle (`English | मराठी`)**: A prominent toggle in the header instantly switches all UI elements (welcome message, sample prompts, placeholders, loading states, error dialogs, empty states). Preference is persisted across sessions via `localStorage`.
3. **Backend Language Synchronization**: Sending a message automatically passes the active language (`en` or `mr`) to the backend, directing Gemini to respond in the appropriate language.
4. **Mobile Usability**:
   - Tested down to 320px viewport width without horizontal overflow.
   - Touch targets designed to $\ge 44\text{px}$.
   - CSS `env(safe-area-inset-bottom)` handles mobile notches and virtual keyboards.
   - Public PDF upload buttons, deletion controls, and drawers are completely removed from the user interface.

---

## RAG Evaluation and Tuning (Phase 4)

Phase 4 introduces an evaluation dataset and automated evaluation script to rigorously test retrieval quality, distance thresholds, LLM answers, and parameter tuning.

### 1. Evaluation Dataset (`eval/questions.json`)
The dataset contains **23 realistic questions** based on the sample land-record PDF (`data/sample_pdfs/sample_land_record.pdf`):
- **Explicit questions (12)**: Facts directly stated on Page 1 (7/12 extract) or Page 2 (Ferfar mutation register).
- **Multi-page questions (3)**: Questions requiring retrieval across both physical pages (e.g. Village/Taluka/District location across 7/12 and Ferfar; complete ownership succession timeline).
- **Paraphrased questions (4)**: Questions with alternative wording, synonyms, and cross-lingual Marathi queries (`या जमिनीचे एकूण क्षेत्रफळ आणि वहितीखालील जमीन किती आहे?`).
- **Absent / Out-of-scope questions (4)**: Domain-relevant missing facts (land tax, bank loans, ready reckoner rates) and general questions (misal pav recipe).

### 2. Running Evaluation
```bash
# Run retrieval evaluation & display metrics
python eval/run_eval.py

# Run with verbose per-question output
python eval/run_eval.py -v

# Run systematic parameter tuning grid
python eval/run_eval.py --tune

# Run with live Gemini LLM answers (requires GEMINI_API_KEY)
python eval/run_eval.py --api-key YOUR_KEY
```

> **Evaluation Safety**:
> - Evaluation operates in an isolated temporary Chroma directory and **never** touches or modifies production/user data in `data/chroma`.
> - If `GEMINI_API_KEY` is not configured, LLM answer evaluation is cleanly marked as skipped (`[LLM EVALUATION SKIPPED]`). Per project requirements, `MockLLMClient` is **never** used to fake real LLM quality.

### 3. Core Retrieval Benchmark Results

| Metric | Measured Value | Target | Evaluation Status |
| :--- | :--- | :--- | :--- |
| **Top-1 Hit Rate (P@1)** | **100.0%** (19/19) | > 90% | Excellent |
| **Top-k Hit Rate (R@k)** | **100.0%** (19/19) | > 95% | Excellent |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | > 0.90 | Perfect |
| **Multi-page Coverage** | **100.0%** (3/3) | > 90% | Excellent |

---

## Running the Automated Test Suite

Run the full automated test suite using pytest inside the virtual environment:

```powershell
.\.venv\Scripts\pytest -v
```

**Test Suite Breakdown (87 Tests Total, 0 Failures):**
- `tests/test_official_doc.py` (10 tests): 500-page batching simulation (105 pages), 3 fallback tiers (grounded, general, 503), safe replacement, isolation from legacy docs, language prompt switching, and admin ingestion endpoint.
- `tests/test_frontend.py` (4 tests): English default for first-time visitors, `English | मराठी` header toggle presence, Devanagari font stylesheet links, and mobile-responsive viewport meta tag.
- `tests/test_llm_grok.py` (17 tests): Provider integration, model resolution, mock client fallback, and error handling.
- `tests/test_eval.py` (14 tests): Evaluation dataset schema, retrieval metric calculations, answer classification, LLM skip fallback, and prompt injection defense.
- `tests/test_ocr.py` (10 tests): Page-level scanned detection, OCR fallback, mixed PDFs, missing Tesseract error handling, Devanagari character preservation.
- `tests/test_api.py` (13 tests): FastAPI REST endpoints, `X-API-Key` auth, PDF validation, user isolation, document listing and deletion.
- `tests/test_health.py` (3 tests): Health check and Swagger documentation accessibility.
- `tests/test_chunker.py` (5 tests): Character chunking, overlap, and page OCR flag propagation.
- `tests/test_embedder.py` (1 test): Multilingual E5 prefixing and normalization.
- `tests/test_pdf_loader.py` (3 tests): Text extraction and page index tracking.
- `tests/test_vector_store.py` (3 tests): ChromaDB storage, retrieval, and multi-tenant isolation.
- `tests/test_rag.py` (4 tests): RAG coordinator, distance threshold gatekeeper, and general fallback.

Total: **87 passed, 2 warnings in ~12s**.

---

## Current Status & Known Limitations

1. **System Tesseract Dependency**:
   - Automated unit tests use dependency injection and mocking to verify OCR fallback, quality guards, and Devanagari Unicode preservation without requiring system binaries.
   - For real OCR on physical scanned files, the host machine must have the Tesseract executable and `mar.traineddata` installed as detailed in the setup guide above.
2. **Real Gemini LLM Completions**:
   - Supplying a valid `GEMINI_API_KEY` in `.env` enables live Gemini generation (`gemini-3.8-flash` configured).
3. **500-Page Document Ingestion Verification**:
   - Automated tests verify large document batching, memory isolation, and page tracking with synthetic multi-page PDFs (up to 105 pages).
   - When deploying with your actual 500-page official PDF, run `python -m app.ingest_official <path-to-pdf>` and inspect console progress logs.

---

## Implementation Roadmap

- [x] **Phase 0: Setup** — Project structure, virtual environment, dependencies, `.env`, FastAPI `/health` endpoint and test validation.
- [x] **Phase 1: Core RAG in Scripts** — Text extraction, chunking with page metadata, multilingual embeddings, ChromaDB indexing, grounded prompt generation, and fallback logic.
- [x] **Phase 2: FastAPI Service** — Full REST API endpoints (`/upload`, `/chat`, `/documents`, `/documents/{doc_id}`), `X-API-Key` security, PDF validation, SQLite metadata registry, and multi-tenant data isolation.
- [x] **Phase 3: OCR and Marathi Support** — Page-by-page scanned detection, Tesseract Devanagari OCR (`mar+eng`), configurable thresholds, quality guards, and multilingual retrieval.
- [x] **Phase 4: Evaluation and Tuning** — Automated evaluation with `eval/questions.json`, retrieval metrics, distance gatekeeper analysis, real Gemini LLM evaluation with skip fallback, and parameter tuning grid.
- [x] **Phase 5: Mobile-First Bilingual Chat UI** — Mobile-first webview-compatible chat interface at `/chat/` with English default and `English | मराठी` selector.
- [x] **Phase 6: Official 500-Page Document Workflow & Three-Tier Fallback** — Ingestion CLI/API, atomic replacement, memory-safe batching, grounded citations, general fallback without citations, and HTTP 503 when unindexed.