"""End-to-end Phase 1 pipeline demonstration script.

Run this from the project root (with .venv activated) using:

    python notebooks/demo_rag_pipeline.py

What this script does step by step:
  1. Load a sample land-record PDF and print extracted text per page.
  2. Chunk each page into overlapping segments and print them.
  3. Load the multilingual-e5-base embedding model.
  4. Embed all chunks (with 'passage: ' prefix) and a test question (with 'query: ' prefix).
  5. Store chunks + embeddings in a ChromaDB collection.
  6. Run a similarity search and print the top-k results with cosine distances.
  7. Build the grounded prompt and (optionally) call the Gemini LLM.
  8. Demonstrate the general-answer fallback path.

No API key is required to run steps 1–6.  Step 7 uses a mock LLM if GEMINI_API_KEY is unset.
"""

from pathlib import Path
import sys
import os

# Allow running from the project root without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Configuration ──────────────────────────────────────────────────────────────
PDF_PATH      = "data/sample_pdfs/sample_land_record.pdf"
DEMO_USER_ID  = "demo_user_001"
DEMO_DOC_ID   = "demo_doc_001"
DEMO_FILENAME = "sample_land_record.pdf"
CHROMA_DIR    = "data/chroma_demo"        # Separate from the real data folder
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 4

SEP = "=" * 65


# ── Step 1: PDF Extraction ─────────────────────────────────────────────────────
print(SEP)
print("STEP 1 — PDF Text Extraction")
print(SEP)

from app.pdf_loader import extract_text_from_pdf

pages = extract_text_from_pdf(PDF_PATH)
print(f"Extracted {len(pages)} page(s) from '{PDF_PATH}'")
for p in pages:
    print(f"\n--- Page {p.page_number} ({len(p.text)} chars) ---")
    print(p.text[:300].strip())
    if len(p.text) > 300:
        print("  [... truncated ...]")


# ── Step 2: Chunking ───────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 2 — Chunking with Overlap")
print(SEP)

from app.chunker import chunk_pages

chunks = chunk_pages(
    pages=pages,
    doc_id=DEMO_DOC_ID,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
print(f"Total chunks produced: {len(chunks)}")
for c in chunks:
    print(f"\n  [{c.chunk_id}] page={c.page} | {len(c.text)} chars")
    print(f"  '{c.text[:100].strip()}...'")


# ── Step 3: Embedding ──────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 3 — Embedding Chunks with multilingual-e5-base")
print(SEP)
print("Loading embedding model (downloads on first run, cached afterwards)...")

from app.embedder import E5Embedder

embedder = E5Embedder(model_name="intfloat/multilingual-e5-base", device="cpu")
chunk_texts   = [c.text for c in chunks]
embeddings    = embedder.embed_passages(chunk_texts)
print(f"Embedded {len(embeddings)} chunk(s).")
print(f"Embedding dimensionality: {len(embeddings[0])} floats per vector")


# ── Step 4: Vector Storage ─────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 4 — Storing Chunks in ChromaDB")
print(SEP)

from app.vector_store import ChromaVectorStore
import shutil

# Clear any stale demo data before running
demo_chroma_path = Path(CHROMA_DIR)
if demo_chroma_path.exists():
    shutil.rmtree(demo_chroma_path)

store = ChromaVectorStore(persist_dir=CHROMA_DIR, collection_name="demo_chunks")
store.add_chunks(
    user_id=DEMO_USER_ID,
    chunks=chunks,
    embeddings=embeddings,
    filename=DEMO_FILENAME,
)
print(f"Stored {len(chunks)} chunks for user='{DEMO_USER_ID}', doc='{DEMO_DOC_ID}'")


# ── Step 5: Retrieval & Similarity Search ──────────────────────────────────────
print(f"\n{SEP}")
print("STEP 5 — Similarity Search (vector retrieval)")
print(SEP)

questions = [
    "Who is the owner of the land?",
    "What is the mutation entry number?",
]

for question in questions:
    print(f"\nQuestion: '{question}'")
    query_vector = embedder.embed_query(question)
    results = store.query(
        user_id=DEMO_USER_ID,
        query_embedding=query_vector,
        top_k=TOP_K,
        doc_id=DEMO_DOC_ID,
    )
    print(f"  Retrieved {len(results)} chunk(s):")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] page={r.page} | distance={r.distance:.4f} | '{r.text[:80].strip()}...'")


# ── Step 6: Grounded Prompt ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 6 — Building the Grounded Prompt")
print(SEP)

from app.chunker import DocumentChunk
from app.prompts import (
    SYSTEM_INSTRUCTION_GROUNDED,
    build_grounded_prompt,
    build_context_block,
)

demo_question = "Who is the owner of survey number 45/2?"
query_vector  = embedder.embed_query(demo_question)
results       = store.query(
    user_id=DEMO_USER_ID, query_embedding=query_vector, top_k=2, doc_id=DEMO_DOC_ID
)

context_chunks = [
    DocumentChunk(
        chunk_id=r.chunk_id, doc_id=r.doc_id, page=r.page,
        chunk_index=r.chunk_index, text=r.text
    )
    for r in results
]

grounded_prompt = build_grounded_prompt(demo_question, context_chunks)
print("System instruction (safety & grounding rules):")
print(f"  {SYSTEM_INSTRUCTION_GROUNDED[:200]}...")
print("\nFull grounded prompt:")
print(grounded_prompt)


# ── Step 7: LLM Answer Generation ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 7 — LLM Answer Generation")
print(SEP)

gemini_key = os.environ.get("GEMINI_API_KEY", "")

if gemini_key:
    print("GEMINI_API_KEY found — calling real Gemini API...")
    from app.rag import answer_question
    response = answer_question(
        question=demo_question,
        user_id=DEMO_USER_ID,
        doc_id=DEMO_DOC_ID,
        vector_store=store,
        embedder=embedder,
    )
else:
    print("GEMINI_API_KEY not set — using MockLLMClient (no API call).")
    from app.llm import MockLLMClient
    from app.rag import answer_question
    mock_answer = "The land owner is Ramesh Vishnu Patil (Page 1)."
    response = answer_question(
        question=demo_question,
        user_id=DEMO_USER_ID,
        doc_id=DEMO_DOC_ID,
        vector_store=store,
        embedder=embedder,
        llm=MockLLMClient(canned_response=mock_answer),
    )

print(f"\nMode   : {response.mode}")
print(f"Answer : {response.answer}")
print(f"Sources:")
for s in response.sources:
    print(f"  - Page {s.page}: '{s.snippet[:80]}...'")


# ── Step 8: General Fallback Demo ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 8 — General Fallback Demonstration")
print(SEP)

from app.llm import MockLLMClient
from app.prompts import NOT_FOUND_TOKEN
from app.rag import answer_question

# ── Path A: Distance-based fallback ───────────────────────────────────────────
#
# In a large real-world database, an off-topic question like "biryani recipe"
# would naturally score cosine distance > 0.45 against land-record chunks.
#
# With our tiny 2-chunk demo database, normalized embedding vectors in
# high-dimensional space (768 dims) can make even unrelated questions score
# below 0.45 — because there are only two reference points in the space.
# This is precisely why Phase 4 (Evaluation & Tuning) tunes the threshold
# against a real set of documents and questions.
#
# For the demo we print the actual raw distances, then re-run with a tighter
# threshold (0.15) so you can SEE the fallback trigger visually.

off_topic_question = "Can you give me the recipe for chicken biryani?"
print(f"Off-topic question: '{off_topic_question}'")
print("  (Showing raw retrieval distances before applying threshold...)")

raw_query_vec = embedder.embed_query(off_topic_question)
raw_results   = store.query(
    user_id=DEMO_USER_ID,
    query_embedding=raw_query_vec,
    top_k=2,
    doc_id=DEMO_DOC_ID,
)
for i, r in enumerate(raw_results, 1):
    print(f"  [{i}] page={r.page} | distance={r.distance:.4f}")

print()
print("  Note: With only 2 chunks in this tiny demo database,")
print("  even unrelated queries can score below 0.45.")
print("  Real-world databases with hundreds of chunks produce correct scores.")
print("  The threshold will be tuned in Phase 4 (Evaluation & Tuning).")
print()
print("  Re-running with demo threshold=0.15 to show the distance fallback path:")

fallback_by_distance = answer_question(
    question=off_topic_question,
    user_id=DEMO_USER_ID,
    doc_id=DEMO_DOC_ID,
    distance_threshold=0.15,  # Tight threshold forces fallback on this tiny DB
    vector_store=store,
    embedder=embedder,
    llm=MockLLMClient(canned_response="Fry onions, add spices, rice and chicken, then steam."),
)
print(f"  Mode   : {fallback_by_distance.mode}   <- 'general' because distance > 0.15")
print(f"  Answer : {fallback_by_distance.answer}")
print(f"  Sources: {fallback_by_distance.sources}")

# ── Path B: NOT_FOUND fallback ─────────────────────────────────────────────────
#
# When the question IS related to the document domain (land records) but the
# specific fact isn't in the retrieved chunks, the grounded LLM returns
# NOT_FOUND and the pipeline falls back to general mode.

class _NotFoundThenFallbackLLM(MockLLMClient):
    """First call simulates LLM saying the context lacks the answer. Second call is the fallback."""
    def __init__(self):
        super().__init__()
        self._calls = 0
    def generate(self, prompt, system_instruction=None, temperature=0.2):
        self._calls += 1
        return NOT_FOUND_TOKEN if self._calls == 1 else "Land tax details are not in this document."

print()
fallback_by_not_found = answer_question(
    question="What is the outstanding land tax amount?",
    user_id=DEMO_USER_ID,
    doc_id=DEMO_DOC_ID,
    distance_threshold=0.45,
    vector_store=store,
    embedder=embedder,
    llm=_NotFoundThenFallbackLLM(),
)
print(f"NOT_FOUND fallback (fact missing from document):")
print(f"  Mode   : {fallback_by_not_found.mode}   <- 'general' because LLM returned NOT_FOUND")
print(f"  Answer : {fallback_by_not_found.answer}")
print(f"  Sources: {fallback_by_not_found.sources}")

print(f"\n{SEP}")
print("Phase 1 demo complete. All steps verified.")
print(SEP)

# Clean up demo chroma data
try:
    store.client._system.stop()
except Exception:
    pass
if demo_chroma_path.exists():
    shutil.rmtree(demo_chroma_path, ignore_errors=True)
