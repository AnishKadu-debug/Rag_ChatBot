# IMPLEMENTATION_PLAN.md

Phased execution plan for the WhatsApp Land Records RAG Chatbot audit recommendations.
Repository: `c:\Users\Anish\Projects\WhatsAppRag` (branch `main`, commit `8be503b`).
Source: `RAG_IMPROVEMENT_RECOMMENDATIONS.md` (IDs RAG-001 … RAG-017) and `CURRENT_SYSTEM_ANALYSIS.md`.

Constraints carried over from audit:
- Do not rebuild the vector database by default; compatibility between ingestion and query was verified (Chroma collection `chunks`, 768-dim cosine, `multilingual-e5-base`).
- WhatsApp integration is out of current scope; only preserve a clean REST service boundary for a future adapter.
- No source/config changes are made by this document; it is a plan only.

---

## Phase 0: Resolve Unknowns

**Recommendation IDs:** S-001, S-007, S-008 (from `RAG_IMPROVEMENT_RECOMMENDATIONS.md`).

**Prerequisites:** None. Read-only diagnostics.

**Tasks (execution order):**
1. Re-run `eval/run_eval.py` with the current `.env` (resolved provider = OpenRouter per duplicate-key analysis) and confirm whether generation evaluation runs or is skipped; record which provider/model is actually used. (Addresses S-001.)
2. On the target local host, run `is_tesseract_available()` / `check_language_support("mar+eng")` to confirm Tesseract + Marathi pack presence for scanned ingestion. (Addresses S-007.)
3. Confirm the intended `MAX_PAGES` end-user limit (`.env` says 500; `config.py`/`.env.example` default 30) and align config accordingly. (Addresses S-008.)

## Phase 1: Fix Critical Issues

**Recommendation IDs:** RAG-001 (P0), RAG-002 (P0), RAG-003 (P1).

**Prerequisites:** Phase 0 complete (provider intent known).

**Tasks (execution order):**
1. Deduplicate `LLM_PROVIDER` / `LLM_MODEL` in `.env` so exactly one provider/model remains; align `.env.example`. Add startup log of resolved provider+model (no keys). (RAG-001; files: `.env`, `.env.example`, possibly `app/config.py`.)
2. Rotate any live keys present in `.env`; move real keys out of the repo (secret manager / host env); keep `.env.example` placeholder-only. (RAG-002; file: `.env`.)
3. Make `/api/v1/admin/upload-official` report the real indexed chunk count instead of hardcoded `0` (have `ingest_official_pdf` return chunk count or count via `vector_store`). (RAG-003; files: `app/main.py`, `app/ingest_official.py`.)

**Files/modules likely to change:** `.env`, `.env.example`, `app/main.py`, `app/ingest_official.py`, optionally `app/config.py`.

**Dependencies:** Phase 0 (to know intended provider).

**Acceptance criteria:**
- App starts with a single, intended LLM provider/model; logged at startup.
- `.env` contains no real API key values.
- Admin upload response `chunks` equals actual stored chunk count.

**Risks/rollback:** Rotating keys requires updating wherever they're deployed; keep old values available until clients are repointed. `.env` edits are reversible from git.


**Files/modules likely to change:** None in this phase (diagnostics only); `.env` / `app/config.py` may change after S-008 outcome.

**Dependencies:** None.

**Acceptance criteria:**
- Provider/model in use is unambiguous and matches `.env`.
- Tesseract/Marathi availability documented for the host.
- Intended `MAX_PAGES` value confirmed.

**Risks/rollback:** Read-only; no rollback needed.


## Phase 2: Establish a Reliable Baseline

**Recommendation IDs:** None new (applies all of Phase 1 fixes).

**Prerequisites:** Phase 1 complete.

**Tasks (execution order):**
1. Start the service locally: `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`.
2. Verify the complete local query flow end-to-end: `/health`, `/api/v1/upload` (user PDF), `/api/v1/chat` (grounded answer against official doc and against user doc), `/api/v1/documents`, `/api/v1/documents/{doc_id}`.
3. Run the full test suite (`pytest`) and confirm all tests pass.
4. Confirm multi-tenant isolation: user B cannot retrieve/delete user A's document (already tested; re-verify after Phase 1 changes).
5. Document the local runbook (start command, env requirements, how to ingest the official PDF).

**Files/modules likely to change:** None (verification only); runbook documentation added.

**Dependencies:** Phase 1.

**Acceptance criteria:**
- Service starts without errors.
- Chat against the official 492-page document returns a grounded answer with page citations.
- `pytest` suite green (target: all existing tests pass).
- Multi-tenant isolation holds.

**Risks/rollback:** None destructive; verification only.


## Phase 3: Improve Retrieval and Answer Accuracy

**Recommendation IDs:** RAG-004 (P1), RAG-005 (P1), RAG-006 (P2), RAG-007 (P2), RAG-008 (P2), RAG-009 (P2), RAG-010 (P2).

**Constraint:** Do not alter ingestion or rebuild the vector database unless evidence shows incompatibility (compatibility already verified in CURRENT_SYSTEM_ANALYSIS.md Section 4).

**Tasks (execution order):**
1. Wire `ChatRequest.history` through `answer_question` and include recent turns in the grounded/general prompt within a context budget. (RAG-004; files: `app/rag.py`, `app/prompts.py`, `app/schemas.py`.)
2. Run `eval/run_eval.py` against the real 492-page official document and record retrieval metrics (hit@k, MRR, multi-page coverage). (RAG-005; file: `eval/run_eval.py`, `eval/`.)
3. Re-tune `SIMILARITY_DISTANCE_THRESHOLD` on the real corpus; update config. (RAG-006; file: `.env`, `app/config.py`.)
4. Add a context budget (char/tokens) to `build_context_block` that truncates/drops lowest-ranked chunks while preserving citation metadata. (RAG-007; file: `app/prompts.py`.)
5. Add an ingestion audit record capturing pdf filename, doc_id, chunk count, embedding model, chunk_size/overlap, OCR flag, timestamp. (RAG-008; files: `app/ingest_official.py`, `app/document_registry.py`.)
6. Record embedding model name + vector dimension at ingestion and assert match at query startup; fail fast on mismatch. (RAG-009; files: `app/vector_store.py`, `app/embedder.py`, `app/document_registry.py`.)
7. Add a heuristic post-generation citation verification (cited pages present in retrieved set; key answer tokens in context) without a second LLM call. (RAG-010; file: `app/rag.py`.)

**Files/modules likely to change:** `app/rag.py`, `app/prompts.py`, `app/schemas.py`, `app/vector_store.py`, `app/embedder.py`, `app/ingest_official.py`, `app/document_registry.py`, `.env`, `app/config.py`.

**Dependencies:** Phase 2 (baseline working); Phase 0 (provider intent).

**Acceptance criteria:**
- Multi-turn follow-up questions retain context.
- Eval results based on real 492-page doc recorded.
- Threshold tuned with documented F1/recall trade-off.
- Context block bounded; no context-window overflow at higher `top_k`.
- Ingestion audit written; embedding mismatch detected at startup.
- Fabricated citations flagged.

**Risks/rollback:** Threshold change could increase fallback rate; tune conservatively and keep rollback to 0.45. Do not reindex unless evidence demands it.


## Phase 4: Improve Reliability and Performance

**Recommendation IDs:** RAG-011 (P1), RAG-012 (P2), RAG-013 (P2), RAG-015 (P2), RAG-016 (P2).

**Prerequisites:** Phase 3 complete.

**Tasks (execution order):**
1. Add retry with exponential backoff around `llm.generate` (honor 429/`Retry-After`), bounded attempts, fallback to general mode after exhaustion. (RAG-011; files: `app/llm.py`, `app/rag.py`.)
2. Add rate limiting (token bucket per API key / per user_id) on `/api/v1/chat` and `/api/v1/upload`. (RAG-012; file: `app/main.py` or middleware.)
3. Add in-memory cache keyed by normalized query text for `embed_query` (TTL/size cap). (RAG-013; file: `app/embedder.py`.)
4. Enhance `/health` to verify ChromaDB collection accessibility and optionally LLM reachability; add structured access logs (latency, mode) for chat/upload. (RAG-015; files: `app/main.py`, logging config.)
5. Make `add_document` raise/return a flag when overwriting an existing `doc_id`; callers choose insert vs replace explicitly. (RAG-016; file: `app/document_registry.py`.)

**Files/modules likely to change:** `app/llm.py`, `app/rag.py`, `app/main.py`, `app/embedder.py`, `app/document_registry.py`.

**Dependencies:** Phase 3.

**Acceptance criteria:**
- Transient 429/timeout on LLM results in retry then fallback, not 500.
- Burst requests from one API key are throttled after the limit.
- Repeated identical questions do not re-invoke the embedding model.
- `/health` reports degraded when ChromaDB is unavailable.
- `add_document` overwrite is explicit.

**Risks/rollback:** Rate limiting could block legitimate burst users; tune limits generously. Retry adds latency on failure; bounded attempts cap it.


## Phase 5: Testing and Evaluation

**Recommendation IDs:** RAG-014 (P1).

**Prerequisites:** Phases 1-4 complete.

**Tasks (execution order):**
1. Re-run generation evaluation with the resolved provider key and record results (non-skipped). (RAG-014a; file: `eval/run_eval.py`, `eval/eval_results.json`.)
2. Add retrieval tests against the 492-page document. (RAG-014b; file: `tests/`.)
3. Add a conversation-history unit test for `answer_question`. (RAG-014c; file: `tests/test_rag.py`.)
4. Re-run full `pytest` suite; confirm green including new tests. (RAG-014d.)

**Files/modules likely to change:** `eval/run_eval.py`, `eval/eval_results.json`, `tests/test_rag.py`, new test files.

**Dependencies:** Phases 1-4.

**Acceptance criteria:**
- Generation eval non-skipped and based on real corpus.
- Retrieval tests on 492-page doc pass.
- History test passes; full suite green.

**Risks/rollback:** Generation eval requires a valid provider key; if unavailable, clearly document skip reason and do not fake results with MockLLMClient.


## Phase 6: Prepare for Future Integration

**Recommendation IDs:** RAG-017 (P2).

**Prerequisites:** Phases 1-5 complete (RAG core stable).

**Tasks (execution order):**
1. Confirm no WhatsApp-specific imports exist in `app/` modules; RAG logic remains strictly in `app/rag.py` + `app/main.py` REST handlers. (RAG-017a.)
2. Document the integration contract: chat page URL format (`https://<host>/?user_id=<id>`), required headers (`X-API-Key`), endpoints (`/api/v1/chat`, `/health`), and auth pattern; place in `web/README.md` or a dedicated integration doc. (RAG-017b; files: `web/README.md` or new doc.)
3. Keep WhatsApp adapter logic (when later built) outside `app/`, calling the REST API or an internal service facade only. (RAG-017c; no `app/` changes.)

**Files/modules likely to change:** `web/README.md` (or new integration doc); no `app/` changes expected.

**Dependencies:** Phase 5 (core stable before defining contract).

**Acceptance criteria:**
- RAG core has no WhatsApp coupling.
- Integration contract documented for the future WhatsApp team.

**Risks/rollback:** None; documentation and boundary preservation only.

---

## Summary

**1. Recommended implementation order:**
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 (sequential; each phase depends on the prior).

**2. Prioritized first five actions:**
1. Deduplicate `LLM_PROVIDER`/`LLM_MODEL` in `.env` (RAG-001, P0).
2. Remove/rotate live API keys from `.env` (RAG-002, P0).
3. Fix admin upload to report real `chunks` count (RAG-003, P1).
4. Re-run evaluation against the real 492-page document (RAG-005, P1).
5. Wire conversation `history` into `answer_question` (RAG-004, P1).

**3. Dependencies between changes:**
- Phase 0 diagnostics must precede Phase 1 (provider intent).
- Phase 1 fixes must precede Phase 2 baseline verification.
- RAG-006 threshold tuning depends on RAG-005 real-corpus eval.
- RAG-004 history and RAG-007 context budget both touch prompt building; coordinate to avoid double-counting context.
- Phase 4 reliability (retries, rate limit, health) builds on the stable RAG core from Phase 3.
- Phase 6 contract depends on Phase 5 stability.

**4. Conditions before considering the local RAG service ready:**
- `.env` has a single, unambiguous LLM provider/model and no committed secrets.
- Service starts locally; `/health`, upload, chat, documents, delete all work.
- `pytest` suite green (including history test).
- Chat against the official 492-page document returns grounded answers with page citations.
- Eval results based on the real corpus are recorded.
- Transient LLM failures fall back gracefully (no unhandled 500).
- `/health` detects ChromaDB unavailability.

**5. Explicitly deferred work (out of current scope):**
- WhatsApp integration: no WhatsApp SDK, webhook, Twilio/Meta API, or phone-number handling will be added. The REST API at `/api/v1/chat` (with `X-API-Key`, `user_id`, `question`, `language`) is the integration surface for a future WhatsApp adapter.
- User accounts/login system (only opaque `user_id` from host page).
- Payments, wallet, orders, or other WhatsApp bot service flows.
- Fine-tuning models or complex agent frameworks.
- Rebuilding the vector database (not required; compatibility verified).

End of IMPLEMENTATION_PLAN.md