# CURRENT_SYSTEM_ANALYSIS.md

Audit of the WhatsApp Land Records RAG Chatbot (Antigravity-based, local FastAPI service).
Repository: `c:\Users\Anish\Projects\WhatsAppRag` (branch `main`, commit `8be503b`).
Analysis method: static inspection of all source, config, test, and eval files; read-only SQLite inspection of `data/documents.sqlite3` and `data/chroma/chroma.sqlite3`. No source or config files were modified.

## 1. Project Overview

**Purpose.** A local Retrieval-Augmented Generation (RAG) service that answers questions about Maharashtra land-record documents (Digital 7/12, 8A Holding, Ferfar/Mutation, Property Card, e-Mojni) with page-level citations and a general-assistant fallback. Built as a standalone webview-compatible service to later be linked from a WhatsApp bot page. Documents are pre-ingested into a vector database; end users chat against a configured official document.

**Current technology stack.**
- Language: Python 3.11+ (requirements note says 3.14.7 tested; venv present)
- API framework: FastAPI `0.143.1`, uvicorn `0.54.0`, pydantic `2.14.1`
- PDF extraction: PyMuPDF `1.28.2` (native text) + Tesseract OCR via pytesseract `0.3.13` / pillow `12.3.0` (scanned fallback)
- Embeddings: sentence-transformers `6.1.0`, `intfloat/multilingual-e5-base`, torch `2.14.1`
- Vector database: ChromaDB `1.5.9` (persistent, on disk)
- LLM clients: google-genai `2.31.0` (Gemini), httpx `0.28.1` (Grok/OpenRouter), no LangChain
- Metadata store: SQLite (stdlib `sqlite3`)
- Frontend: vanilla HTML/CSS/JS served at `/chat/`
- Testing: pytest `9.1.1`

**Main entry points.**
- `app/main.py` — FastAPI application object `app`, HTTP routes
- `app/ingest_official.py` — `main()` CLI entry for administrator ingestion (`python -m app.ingest_official <pdf>`)
- `eval/run_eval.py` — evaluation harness entry

**Current execution method.**
- Start API: `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`
- Frontend: `http://127.0.0.1:8000/chat/?api_key=...&user_id=...` (mounted static)
- Admin ingestion: `python -m app.ingest_official <path-to-pdf>`

## 2. Architecture

Components and responsibilities:

| Component | File | Responsibility |
|-----------|------|----------------|
| API router / endpoints | `app/main.py` | `/health`, `/api/v1/upload`, `/api/v1/chat`, `/api/v1/admin/upload-official`, `/api/v1/documents`, `/api/v1/documents/{doc_id}`, static `/chat/` |
| Configuration | `app/config.py` | `Settings` singleton loaded from `.env` via `python-dotenv`; `get_settings()` cached |
| Auth | `app/security.py` | `X-API-Key` verification; PDF upload validation (magic bytes, size, page count) |
| Document registry | `app/document_registry.py` | SQLite metadata store; `DocumentRecord` (doc_id, user_id, filename, pages, is_official, status, ocr_used) |
| PDF loader | `app/pdf_loader.py` | Page-by-page PyMuPDF extraction; Tesseract OCR fallback for scanned pages; `OCR_THRESHOLD_CHARS` gate |
| OCR | `app/ocr.py` | Tesseract binary discovery, language-pack verification (`mar+eng`), image-to-text |
| Chunker | `app/chunker.py` | Page-aware character chunking (no cross-page chunks); `DocumentChunk` with `chunk_id="{doc_id}_{index}"` |
| Embedder | `app/embedder.py` | `E5Embedder` over SentenceTransformer; `passage:` / `query:` prefixes; 768-dim normalized vectors |
| Vector store | `app/vector_store.py` | `ChromaVectorStore` wrapper; collection `chunks`, cosine space; multi-tenant `user_id` filtering; add/query/delete/structural-chunks |
| RAG coordinator | `app/rag.py` | `answer_question()`: structural detection -> vector retrieval -> distance gatekeeper -> grounded LLM -> `NOT_FOUND` -> general fallback |
| Prompts | `app/prompts.py` | Grounded & general system instructions (EN/MR/HI), `build_grounded_prompt`, `build_general_prompt`, `NOT_FOUND_TOKEN` |
| LLM clients | `app/llm.py` | `BaseLLMClient` ABC; `GrokLLMClient`, `OpenRouterLLMClient`, `GeminiLLMClient`, `MockLLMClient` |
| Admin ingestion | `app/ingest_official.py` | Batched embedding/storage of official PDF; safe replacement; user_id=`official`, `is_official=1` |
| Frontend | `web/index.html`, `web/app.js`, `web/style.css` | Bilingual chat UI at `/chat/`; sends `X-API-Key`, `user_id`, `language` |

**Mermaid diagram.**
```mermaid
graph TD
    U[User / WhatsApp webview] -->|HTTPS JSON| W[Web UI /chat/ index.html app.js]
    W -->|X-API-Key, user_id, question, language| F[FastAPI app/main.py]
    F -->|auth| S[security.py verify_api_key]
    F -->|chat| R[rag.py answer_question]
    R -->|embed query| E[embedder.py E5Embedder]
    R -->|retrieve| V[vector_store.py ChromaVectorStore ChromaDB]
    V -->|cosine ANN| DB[(data/chroma chunks collection)]
    R -->|grounded prompt| P[prompts.py]
    R -->|generate| L[llm.py provider client]
    L -->|HTTP| LLM[(Remote LLM: Gemini/Grok/OpenRouter)]
    F -->|upload| PDF[pdf_loader.py PageData]
    PDF -->|OCR fallback| OCR[ocr.py Tesseract]
    PDF -->|chunk| C[chunker.py DocumentChunk]
    C -->|embed| E
    C -->|store| V

## 3. End-to-End Query Flow

Traced through `app/main.py` `chat()` and `app/rag.py` `answer_question()`:

1. **Client request.** Frontend POSTs `/api/v1/chat` with `{ user_id, doc_id?, question, language, history? }` and `X-API-Key` header (`main.py:246`).
2. **Authentication.** `verify_api_key` rejects missing/invalid key with 401 (`security.py:29`).
3. **Official document resolution.** `registry.get_active_official_document(settings.OFFICIAL_DOCUMENT_ID)` returns the `is_official=1, status='ready'` record, or the configured ID if set (`document_registry.py:177`).
4. **Target binding.**
   - If `doc_id` omitted: binds to official document; if no ready official doc exists, returns HTTP 503 (`main.py:292-298`).
   - If `doc_id` == official doc id: `target_user_id = "official"` (`main.py:276`).
   - If `doc_id` is a user-owned doc: `registry.get_document(doc_id, user_id)` enforces ownership; 404 if not found (`main.py:280-291`).
5. **RAG coordination.** `answer_question(question, user_id, doc_id, language, vector_store, embedder, llm)` (`rag.py:114`):
   - Resolves `top_k` (default config `TOP_K=4`) and `distance_threshold` (default `SIMILARITY_DISTANCE_THRESHOLD=0.45`) (`config.py:63,65`).
   - Flow 1: no `doc_id` → general assistant mode immediately (`rag.py:152-154`).
   - Step 1: detects structural queries (chapter/TOC/index keywords EN + Marathi); if structural, calls `vector_store.get_structural_chunks(...)` (`vector_store.py:186`).
   - Step 2: otherwise dense vector retrieval: `embedder.embed_query(question)` (prefix `query: `) → `vector_store.query(user_id, query_embedding, top_k, doc_id)` with `user_id`+`doc_id` metadata filter (`rag.py:174-181`, `vector_store.py:query`).
   - Flow 2: no chunks → general fallback (`rag.py:184-186`).
   - Step 3: distance gatekeeper — if `best_distance > threshold` → general fallback (`rag.py:188-194`).
   - Filters `relevant_results` to `distance <= threshold`; falls back to top-1 if none (`rag.py:197-199`).
   - Step 4: builds grounded prompt (`prompts.py:build_grounded_prompt`) → `llm.generate(...)` with grounded system instruction (`rag.py:215-220`).
   - Flow 3: if LLM returns `NOT_FOUND` token → general fallback (`rag.py:223-225`).
   - Builds `SourceSnippet` citations (first 180 chars per unique page) (`rag.py:228-234`).
   - Returns `RAGResponse(answer, mode, sources)`.
6. **Response.** `main.py` returns `ChatResponse(answer, mode, sources)` (`main.py:313-320`).
7. **General fallback** (`_run_general_fallback`, `rag.py:81`): general prompt + general system instruction; language-specific prefix note; `mode="general"`, `sources=[]`.

**Verified.** Each step above maps to an actual function in the inspected source. Empty/malformed `question` is rejected by Pydantic `min_length=1` (`schemas.py:80-85`). Empty `user_id` returns 400 (`main.py:262`).
    F -->|registry| DR[document_registry.py SQLite documents.sqlite3]
    DR -->|official doc status| R
```
- Evaluation: `python eval/run_eval.py` (isolated temp Chroma dir)

## 4. Ingestion and Vector Database

### How the existing vector database was created (verified from code + live DB)

**Official document (verified in live DB).**
- `data/documents.sqlite3` contains an active official record: `doc_id=d_official_a0c183bc77`, `user_id=official`, `filename=UDCPR_compressed_2.pdf`, `pages=492`, `is_official=1`, `status=ready`, `ocr_used=1`.
- Produced by `app/ingest_official.py` `ingest_official_pdf()` (admin CLI or `/api/v1/admin/upload-official`). Pipeline: validate PDF (admin limits 100 MB / 1000 pages) → `extract_text_from_pdf` (PyMuPDF + OCR fallback) → `chunk_pages` (page-aware, chunk_size/overlap from config) → `embedder.embed_passages` (prefix `passage: `, batch) → `vector_store.add_chunks(user_id="official", ...)` → registry `add_document` + `set_official_document(..., deactivate_others=True)` → `set_document_status("ready")`. On failure: purge partial vectors, mark `failed` (`ingest_official.py:250-260`).

**User-uploaded document (verified in live DB).**
- Record: `doc_id=d_63b0e7994409`, `user_id=u_j0m4wcit`, `filename=Technical Support Intern - JD (1).pdf`, `pages=2`, `is_official=0`, `status=ready`, `ocr_used=0`.
- Produced by `/api/v1/upload` (`main.py:148`): `validate_pdf_upload` (10 MB / 30 pages default; `.env` overrides MAX_PAGES=500) → extract → chunk → embed → `add_chunks(user_id=clean_user_id, ...)` → registry `add_document`. Rollback on error purges vectors + registry row (`main.py:223-235`).

**Vector store (verified in live DB).**
- Chroma collection `chunks`, 768-dimensional vectors, `hnsw:space=cosine`, embedding function type `known`/`default` (Chroma's default), consistent with `intfloat/multilingual-e5-base` 768-dim output (`chroma.sqlite3` `collections` row).
- Multi-tenant isolation enforced via `where` clause on `user_id` (and `doc_id` when scoped) in `query`/`delete_document`/`get_structural_chunks` (`vector_store.py:102-184`, `:220-226`, `:261-269`).

### Query pipeline compatibility with existing data

- Query-time `embed_query` uses `query: ` prefix; ingestion `embed_passages` uses `passage: ` prefix — consistent asymmetric E5 usage (`embedder.py:71`, `:95`). Compatible with existing vectors.
- Collection name default `chunks` matches the live collection (`vector_store.py:51`).
- Cosine distance matches live `hnsw:space=cosine`.
- `user_id`/`doc_id` metadata filters match stored metadata fields (`chunk.to_metadata()` produces `doc_id`, `page`, `chunk_index`, `ocr`; store adds `user_id`, `filename`).
- **Compatibility verified:** query pipeline is structurally compatible with the existing Chroma collection.

### Unverified assumptions about ingestion
- The exact CLI command and `.env` values used to create the 492-page official document cannot be confirmed from source alone (no ingestion log is retained). Confirmed only by the resulting DB records.
- Whether `data/sample_pdfs/sample_land_record.pdf` (2 pages, used by tests/eval) was ever indexed into the live Chroma collection is unknown; the live DB contains the 492-page official doc and a 2-page JD, not the sample PDF by name.

## 5. Component Inventory

| Module | Responsibility | Inputs | Outputs | Key dependencies | Status |
|--------|---------------|--------|---------|------------------|--------|
| `app/main.py` | FastAPI app, routes, static mount | HTTP requests | JSON responses | all app modules | Implemented |
| `app/config.py` | Typed settings singleton from `.env` | env vars | `Settings` object | python-dotenv | Implemented |
| `app/security.py` | API key auth, PDF validation | request headers/file | validated bytes / 401/400/413 | pymupdf | Implemented |
| `app/document_registry.py` | SQLite doc metadata CRUD | doc fields | `DocumentRecord` | sqlite3 | Implemented |
| `app/pdf_loader.py` | Page-by-page PDF text + OCR fallback | PDF path/bytes | `List[PageData]` | pymupdf, pytesseract | Implemented |
| `app/ocr.py` | Tesseract discovery, lang packs, image OCR | PIL Image | text | pytesseract, system Tesseract | Implemented (system binary required) |
| `app/chunker.py` | Page-aware character chunking | `List[PageData]`, doc_id | `List[DocumentChunk]` | — | Implemented |
| `app/embedder.py` | E5 embeddings (passage/query) | text list / query | 768-dim vectors | sentence-transformers, torch | Implemented |
| `app/vector_store.py` | ChromaDB wrapper, CRUD, isolation | chunks/embeddings / query | `List[SearchResult]` | chromadb | Implemented |
| `app/rag.py` | Three-tier RAG coordinator | question, user, doc | `RAGResponse` | embedder, vector_store, llm, prompts | Implemented |
| `app/prompts.py` | Prompt templates, context block | chunks, question, language | prompt strings | — | Implemented |
| `app/llm.py` | Provider-agnostic LLM clients | prompt, system_instruction | generated text | google-genai, httpx | Implemented |
| `app/ingest_official.py` | Admin ingestion CLI/API | PDF path | registry record | all RAG modules | Implemented |
| `web/*` | Bilingual chat frontend | user input | API calls | — | Implemented |
| `eval/run_eval.py` | Retrieval + generation eval | questions.json, PDF | metrics JSON | app modules | Implemented |
| `tests/*` | Unit + integration tests | — | pass/fail | pytest, fastapi TestClient | Implemented (87 passed per README) |

## 6. Configuration and Dependencies

**Environment variables (`.env`, `.env.example`).** Actual `.env` values contain real API keys and are deliberately not reproduced here. Relevant keys:
- `API_KEY`, `ALLOWED_ORIGINS`
- `LLM_PROVIDER`, `LLM_MODEL` (default `grok`/`grok-3-mini` in example; actual `.env` has duplicate keys — see Findings)
- `GEMINI_API_KEY`, `GROK_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, plus base URLs
- `EMBEDDING_MODEL` = `intfloat/multilingual-e5-base`
- `CHROMA_DIR` = `./data/chroma`
- `DOCS_DB_PATH` = `./data/documents.sqlite3`
- `CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`, `TOP_K=4`, `SIMILARITY_DISTANCE_THRESHOLD=0.45`
- `MAX_UPLOAD_MB=10`, `MAX_PAGES` (`.env` overrides to 500)
- `OFFICIAL_DOCUMENT_ID`, `ADMIN_MAX_UPLOAD_MB=100`, `ADMIN_MAX_PAGES=1000`
- `OCR_ENABLED=true`, `OCR_LANGS=mar+eng`, `OCR_THRESHOLD_CHARS=30`, `OCR_DPI=300`, `TESSERACT_CMD`

**Dependency manifest.** `requirements.txt` pins all packages listed in the stack above. `.gitignore` excludes `.env`, `data/chroma/`, `*.sqlite3`, PDFs, venv.

**Note:** `.env` is committed-tracked (modified per git status) and contains live API keys; `.env.example` is the safe template.

## 7. Error Handling and Testing

**Error handling.**
- API key: 401 on missing/invalid `X-API-Key` (`security.py`).
- PDF upload: 400 for non-PDF/corrupt/empty/oversize/page-exceeded; 413 for size; 422 when Tesseract missing for scanned uploads (`main.py`, `security.py`).
- Chat: 400 empty `user_id`; 404 unknown `doc_id`; 503 when official document not ready (`main.py`).
- RAG: graceful degradation to general fallback on no chunks, distance-threshold breach, or `NOT_FOUND` (`rag.py`).
- LLM clients: classify timeouts, connection errors, 401/403, 429, 4xx/5xx into `TimeoutError`/`ConnectionError`/`PermissionError`/`ValueError`/`RuntimeError` (`llm.py`).
- Ingestion: transactional rollback — purge partial vectors + mark `failed` on exception (`ingest_official.py:250-260`).
- OCR: `TesseractNotInstalledError`, `LanguagePackMissingError`, `OCRError` with diagnostic messages (`ocr.py`).
- Logging: `logging` used throughout; `ingest_official.py` configures `basicConfig`. No centralized request logging or structured logging observed.

**Testing.**
- Test suite under `tests/`: `test_api.py`, `test_chunker.py`, `test_embedder.py`, `test_eval.py`, `test_frontend.py`, `test_health.py`, `test_llm_grok.py`, `test_llm_openrouter.py`, `test_ocr.py`, `test_official_doc.py`, `test_pdf_loader.py`, `test_rag.py`, `test_vector_store.py`.
- README reports **87 passed, 2 warnings in ~12s**.
- Tests use `TestClient`, temp dirs, mock embedders/LLMs; OCR tests mock `extract_text_from_image`; Chroma stores isolated in temp dirs.
- Evaluation: `eval/questions.json` (23 questions: explicit/multi_page/paraphrased/absent), `eval/run_eval.py`, results in `eval/eval_results.json` (retrieval hit@1=1.0, hit@k=1.0, MRR=1.0; generation skipped).

## 8. Findings

### Confirmed issues (evidenced in code or live DB)

**F-001. Duplicate LLM_PROVIDER / LLM_MODEL keys in `.env`.** `.env` sets `LLM_PROVIDER=gemini` + `LLM_MODEL=gemini-3.8-flash`, then immediately redefines `LLM_PROVIDER=openrouter` + `LLM_MODEL=openrouter/free`. Last assignment wins in `os.getenv`, so the active provider is OpenRouter with model `openrouter/free`, not Gemini as the first block suggests. `config.py` defaults to `grok`/`grok-3-mini` when env is absent — three conflicting defaults across files. Evidence: `.env` (key values not reproduced).

**F-002. Admin upload always reports `chunks=0`.** `main.py:137` returns `UploadResponse(chunks=0, ...)` for `/api/v1/admin/upload-official` regardless of the actual number of chunks indexed. Evidence: `ingest_official_pdf` returns a `DocumentRecord` with no chunk count; endpoint hardcodes `0`.

**F-003. Conversation history accepted but ignored.** `ChatRequest.history` is defined (`schemas.py:86-89`) but `answer_question` never reads or uses it (`rag.py:114-154`). Multi-turn context is not passed to the LLM.

**F-004. No limit on grounded context length.** `build_context_block` joins all retrieved chunks with no token/character budget (`prompts.py:75-83`). With `top_k=4` and 500-char chunks this is small today, but unbounded for larger top_k or long chunks.

**F-005. `.env` contains live API keys.** Inspected `.env` holds non-empty `GEMINI_API_KEY` and `OPENROUTER_API_KEY` values. This is a secret-exposure risk; `.env` is gitignored but present on disk and reported as modified. Not reproduced in this report.

**F-006. Distance threshold 0.45 is permissive relative to observed distance overlap.** Eval `eval_results.json` shows answerable distances mean 0.193 and absent/off-topic distances mean 0.213 — substantial overlap. At threshold 0.45 essentially all questions pass the gate; the gate relies on the LLM `NOT_FOUND` token and fallback for absent questions. This is a design choice, not a bug, but the threshold provides little filtering value on this data.

**F-007. Evaluation dataset is built from a 2-page sample PDF, while production official doc is 492 pages.** `eval/questions.json` and `eval/eval_results.json` (`total_chunks: 2` in tuning table) reflect a tiny corpus; `multi_page_coverage=100%` and `hit_rate=100%` are not representative of retrieval quality on the 492-page official document. Evidence: `eval/eval_results.json` tuning entries; live DB `documents` table (492-page official).

### Potential / suspected issues (require diagnostics)

**S-001. Provider/model mismatch between eval and runtime.** `eval_results.json` generation was "skipped" citing `GEMINI_API_KEY is not configured`, yet `.env` contains a `GEMINI_API_KEY` and `LLM_PROVIDER=openrouter`. `run_eval.py` instantiates a client based on `settings.LLM_PROVIDER` (openrouter) using `OPENROUTER_API_KEY`. Unclear whether generation eval actually ran against OpenRouter or was skipped; the `eval_results.json` may be stale. Diagnostic: re-run `eval/run_eval.py` with current `.env` and inspect provider resolution.

**S-002. No retries/backoff on LLM calls.** `rag.py` calls `llm.generate(...)` once with no retry. Transient network/429 failures surface as 500. Diagnostic: check whether any retry wrapper exists (none observed in `llm.py` or `rag.py`).

**S-003. No rate limiting or request size guarding beyond PDF validation.** FastAPI app has no rate-limit middleware. Diagnostic: confirm absence (observed).

**S-004. Web frontend sends `X-API-Key` via URL query params and stores it in `sessionStorage`** (`web/app.js`, `web/README.md` security warning). Documented limitation; acceptable for local/dev, not for production WhatsApp webview without a BFF/session-token layer.

**S-005. `document_registry.add_document` uses `INSERT OR REPLACE`** — re-adding an existing `doc_id` silently overwrites the record. Could mask duplicate uploads. Diagnostic: confirm callers never reuse `doc_id` (upload generates UUIDv4 hex; official uses a fixed doc_id — replacement path handles this).

**S-006. Structural query results use hardcoded `distance=0.0`** (`vector_store.py:252`), which always passes the threshold gate. Intended behavior for structural queries, but worth noting.

**S-007. OCR requires system Tesseract binary + `mar.traineddata`.** Not guaranteed on all deployment hosts; digital PDFs work without it, scanned PDFs fail with 422 (`main.py:176-180`). Diagnostic: `is_tesseract_available()` / `check_language_support("mar+eng")` on target host.

**S-008. `MAX_PAGES` conflict.** `.env` sets `MAX_PAGES=500` while `config.py` default and `.env.example` say 30. End-user upload limit effectively 500 pages in current env. Diagnostic: confirm intended limit.

### Missing functionality
- No conversation-history persistence or session store.
- No observability (request logging, metrics, tracing).
- No health check beyond static `{"status":"ok"}` (does not verify Chroma/LLM reachability).
- No evaluation against the real 492-page official document.
- No WhatsApp adapter (out of current scope, but service boundary exists via REST).

## 9. Current Limitations

1. The system operates correctly as a local REST RAG service for the indexed official document and user uploads, verified by tests and live DB records.
2. Reliability is limited by: duplicated/conflicting LLM config in `.env`; history ignored; unbounded context; no LLM retries; no rate limiting; no centralized logging/health verification; eval not representative of the 492-page production doc; `.env` secret exposure.
3. OCR-dependent scanned ingestion requires system Tesseract + Marathi language pack, absent on some hosts.
4. WhatsApp integration is not implemented; the REST API is the intended future integration surface. No WhatsApp-specific code exists in the repo.

End of CURRENT_SYSTEM_ANALYSIS.md