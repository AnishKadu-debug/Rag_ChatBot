# Maharashtra Land Records RAG Chatbot — Web Frontend

A clean, responsive, mobile-first web chat interface built using vanilla HTML5, CSS3, and JavaScript (ES6+). Connects to the FastAPI RAG backend to provide grounded Q&A on an official Maharashtra land-records PDF (approximately 500 pages, covering Digital 7/12, 8A holdings, and Ferfar mutation registers) with general assistant fallback.

---

## 1. Quick Start & Local Execution

There are two straightforward ways to run the frontend locally:

### Option A — Direct via FastAPI (Recommended)
The frontend is mounted as static files directly at `/chat/` in FastAPI:
```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Start the FastAPI server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
```
http://127.0.0.1:8000/chat/?api_key=dev-secret-key-12345&user_id=u123
```
*(Passing `?api_key=` and `?user_id=` in the URL automatically configures your local development session).*

### Option B — Standalone Static Web Server
Because the backend has CORS enabled for `http://localhost:3000` and `http://127.0.0.1:3000`, you can serve the `web/` directory with Python's built-in HTTP server:
```powershell
# In a separate terminal:
python -m http.server 3000 --directory web
```
Then visit:
```
http://localhost:3000/?api_key=dev-secret-key-12345&user_id=u123
```

---

## 2. Features & Architecture

1. **One Fixed Official PDF (Approx. 500 Pages):**
   - End users do not upload or select documents.
   - All user questions are automatically bound server-side to the verified active official document (`user_id="official"`).
   - If the official document is missing, indexing, or unavailable, the system displays a clear service-unavailable error (HTTP 503).
2. **English Default & Marathi Language Option:**
   - **English by default** for first-time visitors with no saved preference.
   - Prominent, touch-friendly `English | मराठी` toggle button in the header.
   - Complete UI localization for all strings, placeholders, badges, error alerts, and suggested questions.
   - Persists user language choice in `localStorage`.
   - Sends `language: "en"` or `language: "mr"` with every chat request, directing Gemini to answer in the selected language.
3. **Three-Tier Response & Fallback Handling:**
   - **Grounded Answer**: When supported by the official PDF, returns grounded response with verified page numbers (e.g. `(Page 5)`) and clickable source chips.
   - **General Fallback**: When the official PDF does not contain the answer, Gemini generates a general answer in the selected language without source citations or page numbers, and without claiming it came from the PDF.
   - **Service Unavailable (HTTP 503)**: When the official PDF is missing or still indexing, displays: *"Official land records document is temporarily unavailable. Please contact the administrator."*
4. **Mobile-First & WhatsApp Webview Optimization:**
   - Designed to run inside mobile webviews and phone browsers starting from 320px width.
   - Safe-area insets (`env(safe-area-inset-bottom)`) prevent mobile home indicators and keyboards from obscuring the message composer.
   - Touch targets $\ge 44\text{px}$ for accessibility.
   - Smooth auto-scrolling that does not interrupt reading.
5. **Clean Public UI (Document Management Removed):**
   - End-user upload buttons, paperclip icons, drag-and-drop, and document deletion modals are removed from public chat.
   - Administrative ingestion is handled through backend scripts or authenticated admin endpoints.

---

## 3. Configuration & Query Parameters

The frontend supports zero-touch configuration via URL parameters or the in-app Developer Settings modal (`⚙️` icon):

| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `user_id` | Random guest `u_...` | Tenant user identifier for multi-turn chat session isolation. |
| `api_key` or `key` | `""` | Dev API key passed to backend via `X-API-Key` header. Stored in `sessionStorage`. |
| `api_url` | Current origin / `http://127.0.0.1:8000` | Backend API base URL. |

---

## 4. How the Administrator Ingests the Official PDF

Administrators ingest or replace the official document without restarting the server:

### Method 1: Administrator CLI Tool (Recommended for 500+ Pages)
```powershell
# 1. Place the official PDF in data/sample_pdfs/ or any local folder
# 2. Run the batched ingestion tool:
python -m app.ingest_official "data/sample_pdfs/official_land_records.pdf"
```
Options:
- `--batch-size 64`: Process chunk embeddings in batches of 64 to minimize RAM usage.
- `--no-replace`: Keep previous official vectors instead of purging them.

### Method 2: Authenticated Admin HTTP Endpoint
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/admin/upload-official" \
  -H "X-API-Key: your-admin-api-key" \
  -F "file=@data/sample_pdfs/official_land_records.pdf" \
  -F "replace=true"
```

### Safe Replacement Guarantee:
When replacing the official PDF:
1. The new document is chunked and embedded in batches under status `indexing`.
2. Existing official vectors are purged from ChromaDB so duplicate or stale chunks are never retrieved.
3. The previous document is marked inactive (`is_official=0`) and the new document is marked `is_official=1`, `status="ready"`.
4. If ingestion fails midway, all new partial vectors are purged and the document is marked `failed`, preventing broken documents from going live.

---

## 5. Security Architecture & Production Requirements

> [!WARNING]
> **Client-Side API Key Security Limitation:**
> In development mode, protected endpoints require the `X-API-Key` header. In a browser client, any secret stored in JavaScript, `sessionStorage`, or URL parameters is visible to the end-user via browser DevTools.
> 
> Therefore, **the server `API_KEY` must never be hardcoded into frontend files or distributed to end-users.**

### What is Required Before Public Production Deployment?
When embedding this chat page into the WhatsApp webview flow (`?user_id=<user_id>`):
1. **Option 1 — WhatsApp Bot Signed Webview Session (Recommended):**
   - The WhatsApp bot backend generates a short-lived HMAC or JWT session token (e.g., valid for 15 minutes) bound to the `user_id`.
   - The webview opens `https://chat.example.com/?session_token=<token>&user_id=<user_id>`.
   - The backend validates the signed session token instead of requiring a static shared secret.
2. **Option 2 — Backend-for-Frontend (BFF) Proxy:**
   - Host the frontend on a server that injects the `X-API-Key` server-side into API requests, or issue HttpOnly session cookies.
3. **CORS Hardening:**
   - Restrict `ALLOWED_ORIGINS` in `.env` to only the company's verified webview domains.
