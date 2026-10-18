"""Unit tests for the RAG pipeline coordination and fallback logic."""

import pytest
from app.llm import MockLLMClient
from app.prompts import NOT_FOUND_TOKEN
from app.rag import RAGResponse, answer_question, resolve_retrieval_query
from app.vector_store import SearchResult


class MockEmbedder:
    """Mock embedder returning fixed dummy vectors without downloading weights."""

    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_passages(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class RecordingEmbedder(MockEmbedder):
    """Mock embedder recording every query passed to it."""

    def __init__(self):
        self.queries = []

    def embed_query(self, query: str):
        self.queries.append(query)
        return [0.1, 0.2, 0.3, 0.4]


class MockVectorStore:
    """Mock vector store allowing custom search result injection."""

    def __init__(self, search_results=None, structural_results=None, page_range_results=None):
        self._results = search_results if search_results is not None else []
        self._structural_results = structural_results if structural_results is not None else []
        self._page_range_results = page_range_results if page_range_results is not None else []
        self.page_range_calls = []

    def query(self, user_id, query_embedding, top_k=4, doc_id=None):
        return self._results

    def get_structural_chunks(self, user_id, doc_id=None, keywords=None, max_results=30):
        return self._structural_results

    def get_chunks_by_page_range(self, user_id, doc_id, start_page, end_page, max_results=10):
        self.page_range_calls.append((user_id, doc_id, start_page, end_page, max_results))
        return self._page_range_results



def test_rag_fallback_when_no_doc_id():
    """Verify that omitting doc_id directly triggers general mode."""
    mock_llm = MockLLMClient(canned_response="A 7/12 extract is a land revenue document.")
    response = answer_question(
        question="What is a 7/12 extract?",
        user_id="user_123",
        doc_id=None,
        llm=mock_llm,
    )

    assert response.mode == "general"
    assert response.sources == []
    assert "7/12 extract is a land revenue document" in response.answer


def test_rag_fallback_when_distance_exceeds_threshold():
    """Verify that chunks with cosine distance > threshold trigger fallback."""
    # Create search result with distance 0.65 (above 0.45 threshold)
    far_result = SearchResult(
        chunk_id="chunk_far",
        doc_id="doc_xyz",
        page=1,
        chunk_index=0,
        text="Irrigation canal maintenance rules.",
        distance=0.65,
    )
    mock_store = MockVectorStore(search_results=[far_result])
    mock_embedder = MockEmbedder()
    mock_llm = MockLLMClient(canned_response="General answer about weather.")

    response = answer_question(
        question="What will the weather be tomorrow?",
        user_id="user_123",
        doc_id="doc_xyz",
        distance_threshold=0.45,
        vector_store=mock_store,
        embedder=mock_embedder,
        llm=mock_llm,
    )

    assert response.mode == "general"
    assert response.sources == []
    assert "couldn't find this information in your document" in response.answer


def test_rag_fallback_when_llm_returns_not_found():
    """Verify that if the grounded LLM returns NOT_FOUND, fallback is triggered."""
    relevant_result = SearchResult(
        chunk_id="chunk_1",
        doc_id="doc_xyz",
        page=2,
        chunk_index=0,
        text="Survey 45 owner: Ramesh Patil.",
        distance=0.15,  # Close distance
    )
    mock_store = MockVectorStore(search_results=[relevant_result])
    mock_embedder = MockEmbedder()

    # Create a dynamic mock LLM: first call returns NOT_FOUND, second call returns fallback
    class FallbackMockLLM(MockLLMClient):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        def generate(self, prompt, system_instruction=None, temperature=0.2):
            self.call_count += 1
            if self.call_count == 1:
                return NOT_FOUND_TOKEN
            return "General guidance on mortgage loans."

    mock_llm = FallbackMockLLM()

    response = answer_question(
        question="What is the mortgage amount?",
        user_id="user_123",
        doc_id="doc_xyz",
        distance_threshold=0.45,
        vector_store=mock_store,
        embedder=mock_embedder,
        llm=mock_llm,
    )

    assert response.mode == "general"
    assert response.sources == []
    assert "General guidance on mortgage loans." in response.answer


def test_rag_grounded_document_success():
    """Verify that valid context and grounded LLM answer produce mode='document' with citations."""
    relevant_result = SearchResult(
        chunk_id="chunk_1",
        doc_id="doc_xyz",
        page=1,
        chunk_index=0,
        text="Land Owner: Ramesh Vishnu Patil. Survey Number: 45/2.",
        distance=0.12,
    )
    mock_store = MockVectorStore(search_results=[relevant_result])
    mock_embedder = MockEmbedder()
    mock_llm = MockLLMClient(
        canned_response="The land owner is Ramesh Vishnu Patil (Page 1)."
    )

    response = answer_question(
        question="Who is the land owner?",
        user_id="user_123",
        doc_id="doc_xyz",
        distance_threshold=0.45,
        vector_store=mock_store,
        embedder=mock_embedder,
        llm=mock_llm,
    )

    assert response.mode == "document"
    assert len(response.sources) == 1
    assert response.sources[0].page == 1
    assert "Ramesh Vishnu Patil" in response.sources[0].snippet
    assert "Ramesh Vishnu Patil (Page 1)" in response.answer


def test_rag_structural_chapter_query_retrieval_and_answer():
    """Verify that structural queries like 'How many chapters are there in this PDF?'
    retrieve chapter structure chunks and produce a grounded document answer with citations."""
    index_chunk_1 = SearchResult(
        chunk_id="chunk_idx_1",
        doc_id="doc_udcpr",
        page=5,
        chunk_index=20,
        text="INDEX\nCHAPTER – 1 ADMINISTRATION\n1.0 Short Title...",
        distance=0.0,
    )
    index_chunk_15 = SearchResult(
        chunk_id="chunk_idx_15",
        doc_id="doc_udcpr",
        page=14,
        chunk_index=66,
        text="CHAPTER – 15 REGULATIONS FOR SPECIAL ACTIVITIES\n15.1 Quarrying Operations\nAPPENDICES",
        distance=0.0,
    )

    mock_store = MockVectorStore(structural_results=[index_chunk_1, index_chunk_15])
    mock_embedder = MockEmbedder()
    mock_llm = MockLLMClient(
        canned_response="There are 15 chapters in this document, from Chapter 1 to Chapter 15 (Page 5, Page 14)."
    )

    response = answer_question(
        question="How many chapters are there in this PDF?",
        user_id="official",
        doc_id="doc_udcpr",
        vector_store=mock_store,
        embedder=mock_embedder,
        llm=mock_llm,
    )

    assert response.mode == "document"
    assert "15 chapters" in response.answer
    assert any(s.page == 5 for s in response.sources)
    assert any(s.page == 14 for s in response.sources)


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up resolution and targeted chapter retrieval
# ─────────────────────────────────────────────────────────────────────────────


def _chapter_1_history():
    return [
        {"role": "user", "content": "List the contents of Chapter 1."},
        {"role": "assistant", "content": "Chapter 1 is Administration (Page 17)."},
    ]


def test_resolve_retrieval_query_contextualizes_followup_from_history():
    """Follow-up references like 'Show it in detail' inherit the previous user topic."""
    history = _chapter_1_history()

    resolved = resolve_retrieval_query("Show it in detail.", history)
    assert "Chapter 1" in resolved
    assert "Show it in detail." in resolved

    # Standalone questions are never rewritten
    assert resolve_retrieval_query("What is potkharaba?", history) == "What is potkharaba?"
    # Without history there is nothing to resolve
    assert resolve_retrieval_query("Show it in detail.", None) == "Show it in detail."


def _chapter_1_index_chunk_20():
    # Real layout from the indexed PDF: the Chapter 1 index block ends mid-title
    # ("1.10 Removal of Di") because the chunker cuts at 500 chars.
    return SearchResult(
        chunk_id="idx_20",
        doc_id="doc_udcpr",
        page=5,
        chunk_index=20,
        text="UDCPR-2020 \nV \n \nINDEX \nRegulation \nNo. \nParticulars \nPage  No. \n \n"
        "CHAPTER – 1  \nADMINISTRATION \n \n1.0 \nShort Title, Extent and Commencement \n1 \n"
        "1.1 * \nExtent and Jurisdiction \n1 \n1.2 \nCommencement of Regulations \n1 \n"
        "1.4 \nApplicability of Regulations \n17 \n1.5 ** \nSaving \n18 \n"
        "1.9 \nMeanings as in Acts, Rules & Interpretations \n20 \n1.10 \nRemoval of Di",
        distance=0.0,
    )


def _chapter_1_index_chunk_21():
    # Overlapping continuation chunk: carries the remaining entries AND the next
    # chapter heading line that terminates the block.
    return SearchResult(
        chunk_id="idx_21",
        doc_id="doc_udcpr",
        page=5,
        chunk_index=21,
        text="er to Decide Charges \n20 \n1.9 \nMeanings as in Acts, Rules & Interpretations \n20 \n"
        "1.10 \nRemoval of Difficulties \n21 \n \nCHAPTER – 2 \nDEVELOPMENT PERMISSION AND "
        "COMMENCEMENT \nCERTIFICATE \n \n2.1 \nPermission from the Planning Authority is Mandatory \n23",
        distance=0.0,
    )


def _chapter_1_content_chunks():
    return [
        SearchResult(
            chunk_id="c_17",
            doc_id="doc_udcpr",
            page=17,
            chunk_index=79,
            text="UDCPR-2020 \n1 \n \nCHAPTER – 1 \nADMINISTRATION \n1.0 \nSHORT TITLE, EXTENT & "
            "COMMENCEMENT \nThese regulations shall be called as Unified Development Control "
            "and Promotion Regulations for Maharashtra.",
            distance=0.0,
        ),
        SearchResult(
            chunk_id="c_18",
            doc_id="doc_udcpr",
            page=18,
            chunk_index=85,
            text="1. Act – means the Maharashtra Regional & Town Planning Act, 1966.",
            distance=0.0,
        ),
    ]


def _chapter_1_heading_chunks():
    heading_ch1 = SearchResult(
        chunk_id="head_ch1",
        doc_id="doc_udcpr",
        page=17,
        chunk_index=79,
        text="UDCPR-2020 \n1 \n \nCHAPTER – 1 \nADMINISTRATION \n1.0 \nSHORT TITLE, EXTENT & "
        "COMMENCEMENT.",
        distance=0.0,
    )
    heading_ch2 = SearchResult(
        chunk_id="head_ch2",
        doc_id="doc_udcpr",
        page=39,
        chunk_index=234,
        text="UDCPR-2020 \n23 \n \nCHAPTER – 2 \nDEVELOPMENT PERMISSION AND COMMENCEMENT CERTIFICATE \n"
        "2.1 \nPERMISSION FROM THE PLANNING AUTHORITY IS MANDATORY",
        distance=0.0,
    )
    return [heading_ch1, heading_ch2]


def test_followup_detail_uses_targeted_chapter_retrieval():
    """A detail follow-up about Chapter 1 uses PDF-ordered chapter body text, not dense search.

    Chapter explanations must be grounded in the chapter's body text; the front-index
    block (titles and index-stated page numbers) is for contents questions only.
    """
    mock_store = MockVectorStore(
        structural_results=[
            _chapter_1_index_chunk_20(),
            _chapter_1_index_chunk_21(),
            *_chapter_1_heading_chunks(),
        ],
        page_range_results=_chapter_1_content_chunks(),
    )
    embedder = RecordingEmbedder()
    mock_llm = MockLLMClient(
        canned_response="Chapter 1 (Administration) includes the short title, extent, "
        "commencement and definitions (Page 17, Page 18)."
    )

    response = answer_question(
        question="Show it in detail.",
        user_id="official",
        doc_id="doc_udcpr",
        vector_store=mock_store,
        embedder=embedder,
        llm=mock_llm,
        history=_chapter_1_history(),
    )

    assert response.mode == "document"
    assert mock_store.page_range_calls == [("official", "doc_udcpr", 17, 38, 800)]
    assert embedder.queries == []  # Targeted retrieval needs no dense embedding
    assert [s.page for s in response.sources] == [17, 18]
    prompt = mock_llm.last_prompt or ""
    assert "Show it in detail." in prompt
    # Actual body text reaches the prompt …
    assert "These regulations shall be called" in prompt
    assert "Town Planning Act, 1966" in prompt
    # … while the front-index entries do not (they are not chapter body content).
    assert "Removal of Difficulties" not in prompt
    assert "Extent and Jurisdiction" not in prompt


def test_contents_question_uses_complete_index_block_only():
    """'List the contents of Chapter 1' must use the full index block (all entries,
    complete titles) and no body chunks, so index-stated page numbers stay intact."""
    mock_store = MockVectorStore(
        structural_results=[
            _chapter_1_index_chunk_20(),
            _chapter_1_index_chunk_21(),
            *_chapter_1_heading_chunks(),
        ],
        page_range_results=_chapter_1_content_chunks(),
    )
    embedder = RecordingEmbedder()
    mock_llm = MockLLMClient(
        canned_response="Chapter 1 – Administration: 1.0 Short Title ... 1.10 Removal of "
        "Difficulties (Page 5)."
    )

    response = answer_question(
        question="List the contents of Chapter 1.",
        user_id="official",
        doc_id="doc_udcpr",
        vector_store=mock_store,
        embedder=embedder,
        llm=mock_llm,
    )

    assert response.mode == "document"
    prompt = mock_llm.last_prompt or ""
    # Both index chunks (complete entries, no truncated titles) must be present
    assert "1.0" in prompt and "1.10" in prompt
    assert "Removal of Difficulties" in prompt
    # No body chunks and no dense embedding for a contents question
    assert mock_store.page_range_calls == []
    assert embedder.queries == []
    # Citations point at the index page only
    assert [s.page for s in response.sources] == [5]


def test_followup_missing_context_returns_document_not_found():
    """A document-scoped follow-up with no adequate source text returns a clear
    document-not-found answer — never a general-knowledge fallback."""
    unrelated = SearchResult(
        chunk_id="chunk_far",
        doc_id="doc_xyz",
        page=99,
        chunk_index=0,
        text="Unrelated content about a different subject.",
        distance=0.65,
    )
    mock_store = MockVectorStore(search_results=[unrelated])
    embedder = RecordingEmbedder()
    mock_llm = MockLLMClient(canned_response="General answer about the topic.")

    response = answer_question(
        question="Show it in detail.",
        user_id="official",
        doc_id="doc_xyz",
        distance_threshold=0.45,
        vector_store=mock_store,
        embedder=embedder,
        llm=mock_llm,
        history=_chapter_1_history(),
    )

    assert response.mode == "document"
    assert response.sources == []
    assert "was not found in the official document" in response.answer
    # The general LLM must never be called for a document-scoped question
    assert mock_llm.last_prompt is None
    # The history-contextualized query was used for retrieval
    assert embedder.queries
    assert "Chapter 1" in embedder.queries[0]


def test_missing_section_with_no_candidates_returns_document_not_found():
    """An explicit section reference that does not exist in the document must return
    a clear document-not-found answer, not general Maharashtra land-revenue content."""
    mock_store = MockVectorStore(search_results=[])
    embedder = RecordingEmbedder()
    mock_llm = MockLLMClient(canned_response="Generic Maharashtra land revenue answer.")

    response = answer_question(
        question="Explain section 9.9 in detail.",
        user_id="official",
        doc_id="doc_udcpr",
        vector_store=mock_store,
        embedder=embedder,
        llm=mock_llm,
    )

    assert response.mode == "document"
    assert response.sources == []
    assert "was not found in the official document" in response.answer
    assert mock_llm.last_prompt is None


def test_section_question_retrieves_actual_section_body():
    """'Explain section 1.1 in detail' must retrieve the section's real body text
    (heading chunk plus following chunks until the next heading), not the index entry."""
    decoy = SearchResult(
        chunk_id="decoy_p3",
        doc_id="doc_udcpr",
        page=3,
        chunk_index=6,
        text="Notification under Regulation No.1.10 of the Unified Development control and "
        "Promotion Regulations issued by State Government.",
        distance=0.0,
    )
    heading_chunk = SearchResult(
        chunk_id="sec_head_11",
        doc_id="doc_udcpr",
        page=17,
        chunk_index=79,
        text="UDCPR-2020 \n1 \n \nCHAPTER – 1 \nADMINISTRATION \n1.0 \nSHORT TITLE, EXTENT & "
        "COMMENCEMENT \n \n1.1 \nExtent and Jurisdiction \n \ni) \nThese regulations shall apply "
        "to the building activities and development works on lands",
        distance=0.0,
    )
    body_chunks = [
        heading_chunk,
        SearchResult(
            chunk_id="sec_body_80",
            doc_id="doc_udcpr",
            page=17,
            chunk_index=80,
            text="tivities and development works on lands \nwithin the jurisdiction of all Planning "
            "Authorities and Regional Plan areas except Municipal Corporation of Greater Mumbai.",
            distance=0.0,
        ),
        SearchResult(
            chunk_id="sec_body_81",
            doc_id="doc_udcpr",
            page=17,
            chunk_index=81,
            text="ii) \nThese Regulations shall also be applicable to the Town Planning Scheme area. \n"
            "1.2 \nCommencement of Regulations \n \nThese regulations shall come into force with "
            "effect from the date of publication of notification in the official Gazette.",
            distance=0.0,
        ),
    ]
    mock_store = MockVectorStore(
        structural_results=[decoy, heading_chunk],
        page_range_results=body_chunks,
    )
    embedder = RecordingEmbedder()
    mock_llm = MockLLMClient(
        canned_response="Section 1.1 (Extent and Jurisdiction) applies to building activities "
        "within the jurisdiction of all Planning Authorities (Page 17)."
    )

    response = answer_question(
        question="Explain section 1.1 in detail.",
        user_id="official",
        doc_id="doc_udcpr",
        vector_store=mock_store,
        embedder=embedder,
        llm=mock_llm,
    )

    assert response.mode == "document"
    assert mock_store.page_range_calls == [("official", "doc_udcpr", 17, 19, 8)]
    assert embedder.queries == []  # Lexical heading lookup needs no dense embedding
    prompt = mock_llm.last_prompt or ""
    assert "Extent and Jurisdiction" in prompt
    assert "These regulations shall apply" in prompt  # actual body text, not the index entry
    assert "Town Planning Scheme area" in prompt  # body continues to the next heading (1.2)
    assert "Explain section 1.1 in detail." in prompt
    assert [s.page for s in response.sources] == [17]


# ─────────────────────────────────────────────────────────────────────────────
# Section 4.7 regression: front-index TOC entries must not shadow section bodies
# ─────────────────────────────────────────────────────────────────────────────


def _section_47_index_entry():
    # Real front-index layout (physical page 8): an entry line that is exactly
    # "4.7" followed by the title and the document's printed page number (71).
    return SearchResult(
        chunk_id="idx_p8",
        doc_id="doc_udcpr",
        page=8,
        chunk_index=36,
        text="USE CLASSIFICATION AND PERMISSIBLE USES \n \n4.6 \nFuture Urbanisable Zone \n71 \n"
        "4.7 \nCommercial Zone \n71 \n4.8  \nIndustrial Zone \n72",
        distance=0.0,
    )


def _section_47_heading_chunk():
    # Real content layout (physical page 87 = printed page 71): heading + body text.
    return SearchResult(
        chunk_id="sec_head_47",
        doc_id="doc_udcpr",
        page=87,
        chunk_index=592,
        text="his land, shall be the liability of the owner. \n \n4.7 \nCOMMERCIAL ZONE \n \n"
        "In commercial zones, buildings or premises shall be used for the uses and purposes "
        "given below :- \ni) \nAny use permitted in residential R2 zone.",
        distance=0.0,
    )


def _section_47_body_chunk():
    return SearchResult(
        chunk_id="sec_body_593",
        doc_id="doc_udcpr",
        page=88,
        chunk_index=593,
        text="ii) \nAny commercial use of land or building permitted in the zone. \n \n4.8 \n"
        "INDUSTRIAL ZONE \n \nIn industrial zones, the following uses shall be permissible :-",
        distance=0.0,
    )


def _chapter_4_and_5_heading_chunks():
    return [
        SearchResult(
            chunk_id="head_ch4",
            doc_id="doc_udcpr",
            page=81,
            chunk_index=549,
            text="UDCPR-2020 \n65 \n \nCHAPTER -4 \nLAND USE CLASSIFICATION AND PERMISSIBLE "
            "USES \n4.1 \nGENERAL",
            distance=0.0,
        ),
        SearchResult(
            chunk_id="head_ch5",
            doc_id="doc_udcpr",
            page=107,
            chunk_index=731,
            text="UDCPR-2020 \n91 \n \nCHAPTER - 5 \nADDITIONAL PROVISIONS FOR REGIONAL PLAN "
            "AREAS \n5.0 \nGeneral",
            distance=0.0,
        ),
    ]


def test_section_question_skips_front_index_entry_and_retrieves_body():
    """'Tell me about chapter 4 section 4.7' must retrieve the real section body
    (physical pages 87-88), never the front-index TOC entry (physical page 8)."""
    index_entry = _section_47_index_entry()
    heading_chunk = _section_47_heading_chunk()
    body_chunk = _section_47_body_chunk()
    mock_store = MockVectorStore(
        structural_results=[index_entry, heading_chunk, *_chapter_4_and_5_heading_chunks()],
        page_range_results=[heading_chunk, body_chunk],
    )
    embedder = RecordingEmbedder()
    mock_llm = MockLLMClient(
        canned_response="Section 4.7 (Commercial Zone) permits residential R2 uses and "
        "commercial uses (Page 87, Page 88)."
    )

    response = answer_question(
        question="Tell me about chapter 4 section 4.7",
        user_id="official",
        doc_id="doc_udcpr",
        vector_store=mock_store,
        embedder=embedder,
        llm=mock_llm,
    )

    assert response.mode == "document"
    # The body walk starts at the verified content heading page, not the index page.
    assert mock_store.page_range_calls == [("official", "doc_udcpr", 87, 89, 8)]
    assert embedder.queries == []  # Lexical heading lookup needs no dense embedding
    prompt = mock_llm.last_prompt or ""
    assert "In commercial zones, buildings or premises shall be used" in prompt
    assert "Any commercial use of land or building permitted" in prompt
    # The front-index TOC text (printed page numbers) must not be in the context.
    assert "Future Urbanisable Zone" not in prompt
    assert "Commercial Zone \n71" not in prompt
    assert [s.page for s in response.sources] == [87, 88]


def test_followup_detail_resolves_previous_section_and_uses_section_body():
    """'Give me more detail' after a section question resolves to that section and
    answers from the section body, preserving the context in the prompt."""
    index_entry = _section_47_index_entry()
    heading_chunk = _section_47_heading_chunk()
    body_chunk = _section_47_body_chunk()
    mock_store = MockVectorStore(
        structural_results=[index_entry, heading_chunk, *_chapter_4_and_5_heading_chunks()],
        page_range_results=[heading_chunk, body_chunk],
    )
    embedder = RecordingEmbedder()
    mock_llm = MockLLMClient(
        canned_response="In addition to R2 residential uses, any commercial use of land "
        "or building is permitted in the Commercial Zone (Page 87, Page 88)."
    )
    history = [
        {"role": "user", "content": "Tell me about chapter 4 section 4.7"},
        {"role": "assistant", "content": "Section 4.7 is the Commercial Zone (Page 87)."},
    ]

    response = answer_question(
        question="Give me more detail",
        user_id="official",
        doc_id="doc_udcpr",
        vector_store=mock_store,
        embedder=embedder,
        llm=mock_llm,
        history=history,
    )

    assert response.mode == "document"
    # The resolved query still targets section 4.7, so the section body is retrieved.
    assert mock_store.page_range_calls == [("official", "doc_udcpr", 87, 89, 8)]
    prompt = mock_llm.last_prompt or ""
    assert "In commercial zones, buildings or premises shall be used" in prompt
    assert "Give me more detail" in prompt
    assert "Conversation so far:" in prompt
    assert [s.page for s in response.sources] == [87, 88]


def test_missing_section_body_abstains_without_dense_or_chapter_fallback():
    """An explicit section that exists only in the front index (no body heading)
    must abstain instead of answering from index entries or dense matches."""
    index_entry = _section_47_index_entry()
    dense_decoy = SearchResult(
        chunk_id="dense_decoy",
        doc_id="doc_udcpr",
        page=82,
        chunk_index=558,
        text="IV) Commercial Zone - Following other zones shall be treated as equivalent "
        "to Commercial Zone.",
        distance=0.10,
    )
    mock_store = MockVectorStore(
        structural_results=[index_entry, *_chapter_4_and_5_heading_chunks()],
        search_results=[dense_decoy],
    )
    embedder = RecordingEmbedder()
    mock_llm = MockLLMClient(canned_response="Commercial zones allow several uses (Page 82).")

    response = answer_question(
        question="Explain section 4.7 in detail",
        user_id="official",
        doc_id="doc_udcpr",
        vector_store=mock_store,
        embedder=embedder,
        llm=mock_llm,
    )

    assert response.mode == "document"
    assert response.sources == []
    assert "was not found in the official document" in response.answer
    assert embedder.queries == []  # never silently falls back to dense retrieval
    assert mock_llm.last_prompt is None


# ─────────────────────────────────────────────────────────────────────────────
# Chapter overview regression: sample the chapter's own body across its range
# ─────────────────────────────────────────────────────────────────────────────


def test_chapter_overview_samples_chapter_body_not_front_index():
    """'Explain chapter 2' must build the overview from body chunks spread across
    chapter 2's page range — not front-index entries and not other chapters."""
    index_ch2 = SearchResult(
        chunk_id="idx_ch2",
        doc_id="doc_udcpr",
        page=5,
        chunk_index=21,
        text="CHAPTER – 2 \nDEVELOPMENT PERMISSION AND COMMENCEMENT CERTIFICATE \n \n2.1 \n"
        "Permission from the Planning Authority is Mandatory \n23 \n2.2 \nDevelopment "
        "Permission \n31",
        distance=0.0,
    )
    heading_ch1 = SearchResult(
        chunk_id="head_ch1",
        doc_id="doc_udcpr",
        page=17,
        chunk_index=79,
        text="UDCPR-2020 \n1 \n \nCHAPTER – 1 \nADMINISTRATION \n1.0 \nSHORT TITLE",
        distance=0.0,
    )
    heading_ch2 = SearchResult(
        chunk_id="head_ch2",
        doc_id="doc_udcpr",
        page=39,
        chunk_index=390,
        text="UDCPR-2020 \n23 \n \nCHAPTER – 2 \nDEVELOPMENT PERMISSION AND COMMENCEMENT "
        "CERTIFICATE \n2.1 \nPERMISSION FROM THE PLANNING AUTHORITY IS MANDATORY",
        distance=0.0,
    )
    heading_ch3 = SearchResult(
        chunk_id="head_ch3",
        doc_id="doc_udcpr",
        page=59,
        chunk_index=590,
        text="UDCPR-2020 \n43 \n \nCHAPTER – 3 \nGENERAL LAND DEVELOPMENT REQUIREMENT",
        distance=0.0,
    )
    page_chunks = [
        SearchResult(
            chunk_id=f"ch2_p{page}",
            doc_id="doc_udcpr",
            page=page,
            chunk_index=page * 10,
            text=f"Body text for physical page {page}.",
            distance=0.0,
        )
        for page in range(39, 59)
    ]
    # First chunk of the chapter's first page is the heading chunk (same text as the
    # structural result) plus the page's body marker.
    page_chunks[0] = SearchResult(
        chunk_id="head_ch2",
        doc_id="doc_udcpr",
        page=39,
        chunk_index=390,
        text=heading_ch2.text + " \n \nBody text for physical page 39.",
        distance=0.0,
    )

    mock_store = MockVectorStore(
        structural_results=[index_ch2, heading_ch1, heading_ch2, heading_ch3],
        page_range_results=page_chunks,
    )
    embedder = RecordingEmbedder()
    mock_llm = MockLLMClient(
        canned_response="Chapter 2 covers development permission and commencement "
        "certificate procedures across the chapter (Page 39, Page 58)."
    )

    response = answer_question(
        question="Explain chapter 2",
        user_id="official",
        doc_id="doc_udcpr",
        vector_store=mock_store,
        embedder=embedder,
        llm=mock_llm,
    )

    assert response.mode == "document"
    assert mock_store.page_range_calls == [("official", "doc_udcpr", 39, 58, 800)]
    prompt = mock_llm.last_prompt or ""
    # Coverage spans the whole chapter: first and last pages are both present …
    assert "Body text for physical page 39." in prompt
    assert "Body text for physical page 58." in prompt
    # … with exactly the bounded number of sampled chunks …
    assert prompt.count("[Page ") == 8
    # … and no front-index entries or other-chapter content.
    assert "Permission from the Planning Authority is Mandatory" not in prompt
    assert [s.page for s in response.sources] == [39, 41, 44, 46, 49, 51, 54, 58]


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting: raw HTML markup must never reach the user
# ─────────────────────────────────────────────────────────────────────────────


def test_answer_html_markup_is_sanitized_before_return():
    """Raw HTML tags and HTML space entities are stripped from grounded answers."""
    relevant_result = SearchResult(
        chunk_id="chunk_setback",
        doc_id="doc_xyz",
        page=2,
        chunk_index=0,
        text="Setback rules: front setback minimum 3 m.",
        distance=0.12,
    )
    mock_store = MockVectorStore(search_results=[relevant_result])
    mock_llm = MockLLMClient(
        canned_response="Setback:<br>Front 3 m.<b> Mandatory</b>&nbsp;always.<br/>See above."
    )

    response = answer_question(
        question="What are the setback rules?",
        user_id="user_123",
        doc_id="doc_xyz",
        distance_threshold=0.45,
        vector_store=mock_store,
        embedder=MockEmbedder(),
        llm=mock_llm,
    )

    assert response.mode == "document"
    assert response.answer == "Setback:\nFront 3 m. Mandatory always.\nSee above."
    assert "<br" not in response.answer.lower()
    assert "<b>" not in response.answer
    assert "&nbsp;" not in response.answer


def test_markup_only_answer_returns_generation_failed():
    """An answer that consists only of markup sanitizes to empty -> explicit error."""
    relevant_result = SearchResult(
        chunk_id="chunk_setback",
        doc_id="doc_xyz",
        page=2,
        chunk_index=0,
        text="Setback rules: front setback minimum 3 m.",
        distance=0.12,
    )
    mock_store = MockVectorStore(search_results=[relevant_result])
    mock_llm = MockLLMClient(canned_response="<br/>")

    response = answer_question(
        question="What are the setback rules?",
        user_id="user_123",
        doc_id="doc_xyz",
        distance_threshold=0.45,
        vector_store=mock_store,
        embedder=MockEmbedder(),
        llm=mock_llm,
    )

    assert response.mode == "document"
    assert response.sources == []
    assert "could not be generated" in response.answer


def test_grounded_system_instruction_states_page_and_format_rules():
    """The grounded system prompt must pin the physical-page convention and ban markup."""
    from app.prompts import get_grounded_system_instruction

    instruction = get_grounded_system_instruction("en")
    assert "physical PDF page numbers" in instruction
    assert "printed numbering" in instruction
    assert "Never emit HTML tags" in instruction
    assert "never emit Markdown pipe tables" in instruction


def test_empty_grounded_answer_returns_error_without_citations():
    """If retrieved text exists but generation returns nothing, return a clear error
    instead of an empty answer with citations."""
    relevant_result = SearchResult(
        chunk_id="chunk_1",
        doc_id="doc_xyz",
        page=1,
        chunk_index=0,
        text="Land Owner: Ramesh Vishnu Patil. Survey Number: 45/2.",
        distance=0.12,
    )
    mock_store = MockVectorStore(search_results=[relevant_result])
    mock_embedder = MockEmbedder()
    mock_llm = MockLLMClient(canned_response="")

    response = answer_question(
        question="Who is the land owner?",
        user_id="user_123",
        doc_id="doc_xyz",
        distance_threshold=0.45,
        vector_store=mock_store,
        embedder=mock_embedder,
        llm=mock_llm,
    )

    assert response.answer.strip() != ""
    assert "could not be generated" in response.answer
    assert response.sources == []
    # The grounded generation was attempted with the retrieved context
    assert mock_llm.last_prompt is not None
    assert "Ramesh Vishnu Patil" in (mock_llm.last_prompt or "")

