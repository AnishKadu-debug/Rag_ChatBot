"""Core RAG pipeline coordinator: retrieval, grounding, and general fallback logic.

How the Pipeline Connects the Pieces:
1. Query Resolution: Follow-up questions ("Show it in detail") are contextualized with the
   most recent user turns into a standalone retrieval query; non-follow-ups are used as-is.
   For a referenced chapter, its heading page is located and the chapter content is fetched
   in PDF order (bounded) instead of relying on fuzzy similarity alone.
2. Vector Retrieval: Query ChromaDB for the closest document chunks belonging to this user_id.
3. Distance Gatekeeper: Check the cosine distance of the best match against the threshold.
   - If distance > SIMILARITY_DISTANCE_THRESHOLD (e.g. 0.45): route to fallback handling.
4. Grounded LLM Call: If good context exists, build the grounded prompt and ask the LLM.
5. Fallback Rules: Questions that explicitly target the document (chapter, section, contents,
   index) NEVER go to the general-knowledge fallback. If their source text cannot be retrieved,
   a clear document-not-found answer is returned instead of invented legal content. Other
   questions keep the general fallback.
6. Mode & Source Integrity:
   - If answered from document: mode='document', with page citations and snippet sources.
   - If fallback: mode='general', sources=[] (empty). A general answer is NEVER passed off
     as a fact from the user's document.
   - If retrieved text exists but generation returns nothing, a clear error is returned
     (never an empty answer with citations).
"""

from dataclasses import dataclass
import logging
import re
import time
from typing import Dict, List, Optional

from app.chunker import DocumentChunk
from app.config import get_settings
from app.embedder import E5Embedder, get_embedder
from app.llm import BaseLLMClient, generate_with_retry, get_llm_client
from app.prompts import (
    NOT_FOUND_TOKEN,
    SYSTEM_INSTRUCTION_GENERAL,
    SYSTEM_INSTRUCTION_GROUNDED,
    HistoryMessage,
    build_general_prompt,
    build_grounded_prompt,
    get_general_system_instruction,
    get_grounded_system_instruction,
)
from app.vector_store import ChromaVectorStore, SearchResult

logger = logging.getLogger(__name__)

# Patterns identifying document structural / chapter-count / TOC queries
_STRUCTURAL_QUERY_PATTERNS = [
    r"\bchapter[s]?\b",
    r"\btable of contents\b",
    r"\bcontents\b",
    r"\bindex\b",
    r"\btoc\b",
    r"\bheadings\b",
    r"\bप्रकरण[े]?\b",
    r"\bअनुक्रमणिका\b",
]

# Follow-up detection: strong anaphora cannot define a topic on their own
_STRONG_ANAPHORA_PATTERN = re.compile(
    r"\b(it|its|this|that|these|those|them|they|same|above|previous|earlier|mentioned|"
    r"thereof|हे|हा|ती|तो|त्या)\b",
    re.IGNORECASE,
)

# Weak continuation words are only treated as follow-ups in short questions
_WEAK_CONTINUATION_PATTERN = re.compile(
    r"\b(detail|details|elaborate|expand|continue|further|more|deeper|"
    r"तपशील|सविस्तर|पुढे|अधिक)\b",
    re.IGNORECASE,
)

# Chapter references in user questions, e.g. "Chapter 1", "chapter-3", "प्रकरण १"
_CHAPTER_MENTION_PATTERN = re.compile(
    r"(?<![a-z-])(?:chapter|प्रकरण)\s*[-\u2013\u2014:.]?\s*(\d{1,2})(?!\d)",
    re.IGNORECASE,
)

# Dotted section references in user questions, e.g. "section 1.1", "Regulation No. 14.8.8"
_SECTION_MENTION_PATTERN = re.compile(
    r"\b(?:section|regulation|reg\.?|sec\.?|clause|para|नियम|कलम|विनियम)"
    r"\s*(?:no\.?\s*)?[-\u2013\u2014:.]?\s*(\d+(?:\.\d+)+)",
    re.IGNORECASE,
)

# Questions that ask for the table of contents / index of a chapter
_CONTENTS_QUERY_PATTERN = re.compile(
    r"\b(table of contents|contents|index|अनुक्रमणिका|सूची)\b",
    re.IGNORECASE,
)

# Actual chapter heading format on content pages, e.g. "UDCPR-2020 1 CHAPTER – 1 ADMINISTRATION".
# Verified against the indexed official PDF: heading chunks carry the running page header
# ("UDCPR-2020 <arabic page>"), while front-index entries mention chapters mid-chunk.
_CHAPTER_HEADING_PATTERN = re.compile(
    r"UDCPR\s*-\s*2020\s+\d+\s+CHAPTER\s*[-\u2013\u2014:.]?\s*(\d{1,2})(?!\d)",
    re.IGNORECASE,
)

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Bounded targeted-chapter context: never send a whole chapter/section of the PDF to the LLM
_MAX_CHAPTER_CONTEXT_CHUNKS: int = 8
_MAX_CHAPTER_INDEX_BLOCK_CHUNKS: int = 8
_CHAPTER_PAGE_SPAN_FALLBACK: int = 12
# Hard read guard on chunks scanned for a chapter overview before even sampling.
_MAX_CHAPTER_RANGE_SCAN_CHUNKS: int = 800
# Lookup bound for chapter-heading pages (used as the front-matter/content boundary).
_MAX_CHAPTER_HEADING_LOOKUP: int = 200

# Bounded targeted-section context (lexical heading lookup + PDF-order body walk)
_MAX_SECTION_CANDIDATES: int = 300
_MAX_SECTION_CONTEXT_CHUNKS: int = 8
_SECTION_PAGE_SPAN: int = 2

# Raw HTML tags are never acceptable in user-facing answers; <br> becomes a line break.
_HTML_LINE_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?\s*[a-zA-Z][^>]*>")

# Clear answers for document-scoped questions when the document cannot answer them
_DOCUMENT_NOT_FOUND_MESSAGES = {
    "en": "This information was not found in the official document.",
    "mr": "हे उत्तर अधिकृत दस्तऐवजात आढळले नाही.",
    "hi": "यह जानकारी आधिकारिक दस्तावेज़ में नहीं मिली।",
}

_GENERATION_FAILED_MESSAGES = {
    "en": "The document content was retrieved, but the answer could not be generated. "
    "Please try again.",
    "mr": "दस्तऐवजातील मजकूर मिळाला, परंतु उत्तर तयार करता आले नाही. कृपया पुन्हा प्रयत्न करा.",
    "hi": "दस्तावेज़ की सामग्री प्राप्त हुई, लेकिन उत्तर तैयार नहीं हो सका। कृपया पुनः प्रयास करें।",
}

# Bounded contextualized retrieval query built from the most recent user turns
_MAX_RESOLVED_QUERY_CHARS: int = 400
_RESOLVED_QUERY_TURNS: int = 2
_RESOLVED_TURN_CHARS: int = 200



@dataclass
class SourceSnippet:
    """Represents a source citation from the document."""

    page: int
    snippet: str

    def to_dict(self) -> Dict[str, object]:
        return {"page": self.page, "snippet": self.snippet}


@dataclass
class RAGResponse:
    """Standardized response from the RAG pipeline."""

    answer: str
    mode: str  # "document" or "general"
    sources: List[SourceSnippet]

    def to_dict(self) -> Dict[str, object]:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "sources": [s.to_dict() for s in self.sources],
        }


def _run_general_fallback(
    question: str,
    llm: BaseLLMClient,
    language: str = "en",
    prefix_note: bool = True,
    history: Optional[List[HistoryMessage]] = None,
) -> RAGResponse:
    """Execute the fallback path when document context cannot answer the question."""
    prompt = build_general_prompt(question, language=language, history=history)
    system_instruction = get_general_system_instruction(language=language)
    llm_started = time.perf_counter()
    raw_answer = generate_with_retry(
        llm,
        prompt=prompt,
        system_instruction=system_instruction,
    )
    logger.info(
        "RAG timing: general LLM call %.1f ms",
        (time.perf_counter() - llm_started) * 1000,
    )

    if prefix_note:
        lang_lower = (language or "en").lower().strip()
        if lang_lower == "mr":
            prefix = "दस्तऐवजात ही माहिती उपलब्ध नाही, परंतु येथे सामान्य माहिती दिली आहे:\n\n"
        elif lang_lower == "hi":
            prefix = "दस्तावेज़ में यह जानकारी उपलब्ध नहीं है, लेकिन यहाँ सामान्य जानकारी दी गई है:\n\n"
        else:
            prefix = "I couldn't find this information in your document, but here is a general answer:\n\n"
        answer_text = f"{prefix}{raw_answer}"
    else:
        answer_text = raw_answer

    return RAGResponse(
        answer=answer_text,
        mode="general",
        sources=[],
    )


def resolve_retrieval_query(
    question: str,
    history: Optional[List[HistoryMessage]] = None,
) -> str:
    """Build a standalone retrieval query, resolving follow-up references via history.

    Short or anaphoric questions ("Show it in detail.", "Tell me more") carry no
    retrievable topic on their own. For those, the most recent user questions are
    prepended so retrieval targets the same subject (e.g. "Chapter 1") instead of
    an uninformative fragment. The original question is what gets answered; this
    contextualized query is used for retrieval only and is bounded in length.
    """
    clean_question = (question or "").strip()
    if not history:
        return clean_question

    is_followup = (
        bool(_STRONG_ANAPHORA_PATTERN.search(clean_question))
        or (len(clean_question.split()) <= 4 and bool(_WEAK_CONTINUATION_PATTERN.search(clean_question)))
        or len(clean_question.split()) <= 2
    )
    if not is_followup:
        return clean_question

    previous_user_turns = [
        (msg.get("content") or "").strip()
        for msg in history
        if (msg.get("role") or "").strip().lower() == "user" and (msg.get("content") or "").strip()
    ]
    if not previous_user_turns:
        return clean_question

    prefix = " ".join(
        turn[:_RESOLVED_TURN_CHARS] for turn in previous_user_turns[-_RESOLVED_QUERY_TURNS:]
    )
    resolved = f"{prefix} {clean_question}"
    return resolved[:_MAX_RESOLVED_QUERY_CHARS]


def _detect_chapter_number(text: str) -> Optional[int]:
    """Extract a chapter number referenced in a query ("Chapter 1", "प्रकरण १")."""
    normalized = (text or "").translate(_DEVANAGARI_DIGITS)
    match = _CHAPTER_MENTION_PATTERN.search(normalized)
    if not match:
        return None
    return int(match.group(1))


def _detect_section_number(text: str) -> Optional[str]:
    """Extract a dotted section/regulation number ("section 1.1", "Regulation No. 14.8.8")."""
    normalized = (text or "").translate(_DEVANAGARI_DIGITS)
    match = _SECTION_MENTION_PATTERN.search(normalized)
    if not match:
        return None
    return match.group(1)


def _chapter_heading_pages(structural_results: List[SearchResult]) -> Dict[int, int]:
    """Map chapter number -> first physical page of its heading (content pages only)."""
    pages: Dict[int, int] = {}
    for result in structural_results:
        match = _CHAPTER_HEADING_PATTERN.search(result.text or "")
        if not match:
            continue
        chapter_num = int(match.group(1))
        if chapter_num not in pages or result.page < pages[chapter_num]:
            pages[chapter_num] = result.page
    return pages


def _sample_chapter_overview(
    chunks: List[SearchResult], max_chunks: int
) -> List[SearchResult]:
    """Sample one chunk per page, evenly spread across a chapter's page range.

    A chapter overview must reflect the chapter's actual body text across its whole
    range, not just its first pages. Taking the first chunk of evenly spaced pages
    keeps page-order coverage (first and last page included) while staying bounded.
    """
    if len(chunks) <= max_chunks:
        return list(chunks)

    first_chunk_by_page: Dict[int, SearchResult] = {}
    for chunk in chunks:  # already in (page, chunk_index) order
        if chunk.page not in first_chunk_by_page:
            first_chunk_by_page[chunk.page] = chunk

    pages = sorted(first_chunk_by_page)
    if len(pages) <= max_chunks:
        return [first_chunk_by_page[page] for page in pages]

    stride = len(pages) / max_chunks
    picked_indexes = sorted({int(i * stride) for i in range(max_chunks)})
    picked_indexes[-1] = len(pages) - 1  # always include the chapter's last page
    return [first_chunk_by_page[pages[i]] for i in picked_indexes]


def _build_chapter_results(
    store: object,
    user_id: str,
    doc_id: str,
    chapter_num: int,
    structural_results: List[SearchResult],
) -> List[SearchResult]:
    """Targeted retrieval for one chapter: body text sampled across its page range.

    Returns an empty list (caller falls back to generic retrieval) unless a verified
    chapter heading page exists for this chapter. Content is body text only — the
    front-index listing is used for contents questions, not for chapter explanations,
    because index titles carry the document's printed page numbers and no body text.
    """
    if not hasattr(store, "get_chunks_by_page_range"):
        return []

    heading_pages = _chapter_heading_pages(structural_results)
    start_page = heading_pages.get(chapter_num)
    if start_page is None:
        return []

    next_page = heading_pages.get(chapter_num + 1, start_page + _CHAPTER_PAGE_SPAN_FALLBACK + 1)
    end_page = max(start_page, next_page - 1)

    range_chunks = store.get_chunks_by_page_range(
        user_id=user_id,
        doc_id=doc_id,
        start_page=start_page,
        end_page=end_page,
        max_results=_MAX_CHAPTER_RANGE_SCAN_CHUNKS,
    )
    if not range_chunks:
        return []

    return _sample_chapter_overview(range_chunks, _MAX_CHAPTER_CONTEXT_CHUNKS)


def _build_chapter_contents_results(
    structural_results: List[SearchResult],
    chapter_num: int,
) -> List[SearchResult]:
    """Return the complete front-index block for one chapter, in PDF page order.

    The index block starts at the chunk containing the index's own "CHAPTER – N"
    heading line and ends at (and includes) the chunk containing the next chapter's
    heading line. Because the chunker uses overlapping windows, a chapter's entries
    can span several chunks (and a title can be cut at a chunk end), so the whole
    block is returned instead of a keyword-matched subset. Only pre-content pages
    are considered, and the block is bounded by chunk count.
    """
    heading_pages = _chapter_heading_pages(structural_results)
    if not heading_pages:
        return []
    content_start = min(heading_pages.values())

    chapter_line = re.compile(
        rf"^[ \t]*CHAPTER\s*[-\u2013\u2014:.]?\s*{chapter_num}[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    next_line = re.compile(
        rf"^[ \t]*CHAPTER\s*[-\u2013\u2014:.]?\s*{chapter_num + 1}[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )

    block: List[SearchResult] = []
    for result in structural_results:  # already sorted by (page, chunk_index)
        if result.page >= content_start:
            break
        if not block and not chapter_line.search(result.text or ""):
            continue
        block.append(result)
        if len(block) >= _MAX_CHAPTER_INDEX_BLOCK_CHUNKS or next_line.search(result.text or ""):
            break
    return block


def _build_section_results(
    store: object,
    user_id: str,
    doc_id: str,
    section_num: str,
) -> List[SearchResult]:
    """Retrieve the actual body text of one explicitly referenced section.

    Lexical heading lookup first (a line that is exactly the section number, e.g.
    "1.1" on its own line), then the heading chunk plus following chunks in PDF order,
    stopping after the chunk where the next section heading appears. Bounded by chunk
    count; returns an empty list when the section heading is not present in the body.

    Front-matter index entries list the same numbers ("4.7 \\nCommercial Zone \\n71")
    and are earlier in PDF order, so only matches at/after the first chapter-heading
    page (the content boundary) are accepted — otherwise a section question would be
    answered from its table-of-contents entry instead of its body.
    """
    if not (
        hasattr(store, "get_structural_chunks") and hasattr(store, "get_chunks_by_page_range")
    ):
        return []

    candidates = store.get_structural_chunks(
        user_id=user_id,
        doc_id=doc_id,
        keywords=[section_num],
        max_results=_MAX_SECTION_CANDIDATES,
    )

    content_start = 0
    chapter_structural = store.get_structural_chunks(
        user_id=user_id,
        doc_id=doc_id,
        keywords=["CHAPTER"],
        max_results=_MAX_CHAPTER_HEADING_LOOKUP,
    )
    heading_pages = _chapter_heading_pages(chapter_structural)
    if heading_pages:
        content_start = min(heading_pages.values())

    heading_re = re.compile(rf"^[ \t]*{re.escape(section_num)}[ \t]*$", re.MULTILINE)
    heading_chunk: Optional[SearchResult] = None
    for candidate in candidates:  # sorted by (page, chunk_index)
        if content_start and candidate.page < content_start:
            continue  # front-matter index entry, not the section body
        if heading_re.search(candidate.text or ""):
            heading_chunk = candidate
            break
    if heading_chunk is None:
        return []

    body_chunks = store.get_chunks_by_page_range(
        user_id=user_id,
        doc_id=doc_id,
        start_page=heading_chunk.page,
        end_page=heading_chunk.page + _SECTION_PAGE_SPAN,
        max_results=_MAX_SECTION_CONTEXT_CHUNKS,
    )
    if not body_chunks:
        return [heading_chunk]

    other_heading_re = re.compile(r"^[ \t]*(\d+(?:\.\d+)+)[ \t]*$", re.MULTILINE)
    results: List[SearchResult] = []
    started = False
    for chunk in body_chunks:  # sorted by (page, chunk_index)
        if chunk.chunk_id == heading_chunk.chunk_id:
            started = True
            results.append(chunk)
            continue
        if not started:
            continue
        results.append(chunk)
        headings = set(other_heading_re.findall(chunk.text or ""))
        if any(h != section_num and not h.startswith(section_num + ".") for h in headings):
            break  # Next section started; the appended chunk completed the target section.
    return results or [heading_chunk]


def _localized_message(messages: Dict[str, str], language: str = "en") -> str:
    """Pick a pre-written message for the requested language (falls back to English)."""
    return messages.get((language or "en").lower().strip(), messages["en"])


def _document_not_found_response(language: str = "en") -> RAGResponse:
    """Clear document-not-found answer for document-scoped questions with no source text.

    This response is never routed to the general-knowledge fallback, so likely legal
    content from the LLM's memory cannot be presented as if it came from the document.
    """
    logger.info(
        "Document-scoped question has no adequate source text; returning document-not-found response."
    )
    return RAGResponse(
        answer=_localized_message(_DOCUMENT_NOT_FOUND_MESSAGES, language),
        mode="document",
        sources=[],
    )


def _generation_failed_response(language: str = "en") -> RAGResponse:
    """Clear error when source text was retrieved but the LLM produced an empty answer."""
    logger.error("Grounded generation returned an empty answer; returning explicit error response.")
    return RAGResponse(
        answer=_localized_message(_GENERATION_FAILED_MESSAGES, language),
        mode="document",
        sources=[],
    )


def _sanitize_answer_text(text: str) -> str:
    """Return user-safe answer text: no raw HTML tags or HTML space entities.

    LLM providers occasionally emit markup (<br>, <b>, ...) inside otherwise plain
    answers. Line-break tags become newlines; all other tags are removed. This is
    deterministic so raw markup can never reach the client.
    """
    if not text:
        return text
    cleaned = _HTML_LINE_BREAK_RE.sub("\n", text)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ")
    return cleaned.strip()


def answer_question(
    question: str,
    user_id: str,
    doc_id: Optional[str] = None,
    top_k: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    language: str = "en",
    vector_store: Optional[ChromaVectorStore] = None,
    embedder: Optional[E5Embedder] = None,
    llm: Optional[BaseLLMClient] = None,
    history: Optional[List[HistoryMessage]] = None,
) -> RAGResponse:
    """Coordinate the complete RAG query flow with fallback handling.

    Args:
        question: User's query text.
        user_id: User identifier for multi-tenant isolation.
        doc_id: Target document ID (optional; if omitted, routes directly to general mode).
        top_k: Number of chunks to retrieve (defaults to config TOP_K).
        distance_threshold: Maximum allowable cosine distance (defaults to config).
        vector_store: Chroma vector store instance (injected or default).
        embedder: E5Embedder instance (injected or default).
        llm: LLM client instance (injected or default).
        history: Recent conversation turns (RAG-004) as [{"role", "content"}], oldest first.

    Returns:
        RAGResponse: Structured response with answer, mode, and source citations.
    """
    settings = get_settings()
    k = top_k or settings.TOP_K
    threshold = (
        distance_threshold
        if distance_threshold is not None
        else settings.SIMILARITY_DISTANCE_THRESHOLD
    )

    # Initialize dependencies if not injected
    active_llm = llm or get_llm_client()

    # Flow 1: If no document was specified, answer purely as a general assistant
    if not doc_id:
        logger.info("No doc_id supplied; running in general assistant mode.")
        return _run_general_fallback(question, active_llm, language=language, prefix_note=False, history=history)

    active_embedder = embedder or get_embedder()
    active_store = vector_store or ChromaVectorStore()

    # Step 1: Resolve follow-up references ("Show it in detail") into a standalone
    # retrieval query using recent history. The answered question itself is unchanged.
    retrieval_query = resolve_retrieval_query(question, history)
    if retrieval_query != question.strip():
        logger.info(
            "Retrieval query contextualized from history (length=%d chars).",
            len(retrieval_query),
        )

    # Step 2: Detect explicit references (chapter / section) and structural queries
    is_structural = any(
        re.search(p, retrieval_query, re.IGNORECASE) for p in _STRUCTURAL_QUERY_PATTERNS
    )
    chapter_num = _detect_chapter_number(retrieval_query)
    section_num = _detect_section_number(retrieval_query)
    # Contents detection uses the literal question so that a detail follow-up ("Show it in
    # detail") does not inherit "contents" from the previous user turn in history.
    is_contents = bool(_CONTENTS_QUERY_PATTERN.search(question or ""))
    # Questions that explicitly target the document must never be answered by the
    # general-knowledge fallback if retrieval fails (anti-hallucination guard).
    is_document_scoped = is_structural or chapter_num is not None or section_num is not None

    retrieval_started = time.perf_counter()
    search_results: List[SearchResult] = []
    retrieval_path = "dense"

    # Step 2a: Explicit section reference -> exact lexical heading lookup, no dense search.
    # If the requested section body is not present, abstain explicitly rather than
    # answering from an unrelated part of the document or from model memory.
    if section_num is not None:
        section_results = _build_section_results(active_store, user_id, doc_id, section_num)
        if not section_results:
            logger.info(
                "Section %s was requested but its body was not found in the document.",
                section_num,
            )
            return _document_not_found_response(language)
        search_results = section_results
        retrieval_path = f"section:{section_num}"

    # Step 2b: Structural queries (chapters, contents, index)
    if not search_results and is_structural and hasattr(active_store, "get_structural_chunks"):
        structural_results = active_store.get_structural_chunks(
            user_id=user_id,
            doc_id=doc_id,
            keywords=["CHAPTER", "INDEX", "प्रकरण"],
            max_results=60,
        )
        search_results = structural_results
        retrieval_path = "structural"

        if chapter_num is not None:
            if is_contents:
                # Contents answers come from the actual index entries only (complete
                # titles, index-stated page numbers) — no body chunks mixed in.
                contents_results = _build_chapter_contents_results(structural_results, chapter_num)
                if contents_results:
                    search_results = contents_results
                    retrieval_path = f"chapter-index:{chapter_num}"
            else:
                # Detail questions get the index block plus PDF-ordered body chunks.
                chapter_results = _build_chapter_results(
                    active_store, user_id, doc_id, chapter_num, structural_results
                )
                if chapter_results:
                    search_results = chapter_results
                    retrieval_path = f"chapter:{chapter_num}"

    # Step 3: Fall back to dense vector retrieval if no lexical/structural match
    if not search_results:
        embedding_started = time.perf_counter()
        query_vector = active_embedder.embed_query(retrieval_query)
        logger.info(
            "RAG timing: query embedding %.1f ms",
            (time.perf_counter() - embedding_started) * 1000,
        )
        search_results = active_store.query(
            user_id=user_id,
            query_embedding=query_vector,
            top_k=k,
            doc_id=doc_id,
        )
        retrieval_path = "dense"

    logger.info(
        "Retrieval: path=%s | query=%r | pages=%s | distances=%s | candidates=%d",
        retrieval_path,
        retrieval_query[:200],
        [r.page for r in search_results],
        [round(r.distance, 4) for r in search_results],
        len(search_results),
    )
    logger.info(
        "RAG timing: retrieval %.1f ms (%d candidates)",
        (time.perf_counter() - retrieval_started) * 1000,
        len(search_results),
    )

    # Flow 2: If no chunks exist in the vector store for this user/doc
    if not search_results:
        logger.info("No candidate chunks retrieved for user %s, doc %s.", user_id, doc_id)
        if is_document_scoped:
            return _document_not_found_response(language)
        return _run_general_fallback(question, active_llm, language=language, prefix_note=True, history=history)

    # Step 3: Check distance threshold of the best matching chunk
    best_distance = search_results[0].distance
    logger.info("Best retrieval distance: %.4f (threshold: %.4f)", best_distance, threshold)

    if best_distance > threshold:
        if is_document_scoped:
            logger.info("Best distance exceeds threshold for a document-scoped question.")
            return _document_not_found_response(language)
        logger.info("Best distance exceeds threshold; falling back to general assistant.")
        return _run_general_fallback(question, active_llm, language=language, prefix_note=True, history=history)

    # Keep only relevant chunks under the threshold
    relevant_results = [r for r in search_results if r.distance <= threshold]
    if not relevant_results:
        relevant_results = [search_results[0]]

    # Convert SearchResults to DocumentChunks for prompt building
    chunks_for_prompt = [
        DocumentChunk(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            page=r.page,
            chunk_index=r.chunk_index,
            text=r.text,
            ocr=r.ocr,
        )
        for r in relevant_results
    ]

    logger.info(
        "Grounded context: chunks_to_llm=%d pages=%s",
        len(chunks_for_prompt),
        sorted({c.page for c in chunks_for_prompt}),
    )

    # Step 4: Build grounded prompt and generate answer
    grounded_prompt = build_grounded_prompt(
        question, chunks_for_prompt, language=language, history=history
    )
    system_instruction = get_grounded_system_instruction(language=language)
    llm_started = time.perf_counter()
    raw_answer = generate_with_retry(
        active_llm,
        prompt=grounded_prompt,
        system_instruction=system_instruction,
    )
    logger.info(
        "RAG timing: grounded LLM call %.1f ms",
        (time.perf_counter() - llm_started) * 1000,
    )

    # Flow 3: If the grounded LLM indicates the context does not contain the answer
    if NOT_FOUND_TOKEN in raw_answer:
        logger.info("LLM responded with %s.", NOT_FOUND_TOKEN)
        if is_document_scoped:
            return _document_not_found_response(language)
        logger.info("Falling back to general mode.")
        return _run_general_fallback(question, active_llm, language=language, prefix_note=True, history=history)

    # Strip any raw HTML markup the model emitted so only plain text/Markdown is shown.
    answer_text = _sanitize_answer_text(raw_answer)

    # Flow 4: Retrieved text exists but generation produced nothing — return a clear
    # error instead of an empty answer with citations.
    if not answer_text:
        return _generation_failed_response(language)

    # Build source citations (page and short snippet)
    sources: List[SourceSnippet] = []
    seen_pages = set()
    for chunk in chunks_for_prompt:
        if chunk.page not in seen_pages:
            snippet = chunk.text[:180].strip() + ("..." if len(chunk.text) > 180 else "")
            sources.append(SourceSnippet(page=chunk.page, snippet=snippet))
            seen_pages.add(chunk.page)

    # RAG-010: heuristic citation verification — every page the LLM cites must exist in
    # the retrieved context. Mismatches are logged for review; no extra LLM call is made.
    cited_pages = {
        int(m) for m in re.findall(r"(?:Page|पान)\s*:?\s*(\d+)", answer_text, re.IGNORECASE)
    }
    known_pages = {chunk.page for chunk in chunks_for_prompt}
    unknown_citations = cited_pages - known_pages
    if unknown_citations:
        logger.warning(
            "Citation verification: answer cites page(s) %s not present in retrieved context %s.",
            sorted(unknown_citations),
            sorted(known_pages),
        )

    return RAGResponse(
        answer=answer_text,
        mode="document",
        sources=sources,
    )
