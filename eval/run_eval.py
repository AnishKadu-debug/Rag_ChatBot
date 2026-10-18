"""Evaluation and tuning harness for the WhatsApp Land Records RAG pipeline.

Phase 4 Objectives:
1. Evaluate Retrieval Quality:
   - Hit rate @ 1 and Hit rate @ k across answerable questions.
   - Multi-page coverage (retrieving all required pages).
   - Mean Reciprocal Rank (MRR).
   - Cosine distance distributions and distance threshold gatekeeping analysis.
2. Evaluate Real LLM Answer Quality:
   - Grounded generation when GEMINI_API_KEY is available.
   - Verification of cited page numbers against expected pages.
   - Detection of incorrect, unsupported, or incomplete answers.
   - Verification that absent/out-of-scope questions trigger fallback mode.
   - Cleanly skips LLM evaluation when GEMINI_API_KEY is not available (never
     fakes real quality using MockLLMClient).
3. Parameter Tuning Analysis:
   - Evaluates chunk size, overlap, top-k, and similarity distance threshold.
   - Changes defaults only with empirical evidence.
4. Tenant and Data Safety:
   - Uses an isolated temporary ChromaDB directory for evaluation.
   - Never touches or mutates production/user data in data/chroma.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding for Devanagari / Marathi characters
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.chunker import chunk_pages
from app.config import get_settings
from app.embedder import E5Embedder, get_embedder
from app.llm import (
    BaseLLMClient,
    GeminiLLMClient,
    GrokLLMClient,
    OpenRouterLLMClient,
    get_llm_client,
)
from app.pdf_loader import extract_text_from_pdf
from app.rag import RAGResponse, answer_question
from app.vector_store import ChromaVectorStore

logger = logging.getLogger("eval")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalQuestion:
    """Represents a single test question in the evaluation dataset."""

    id: str
    question: str
    category: str  # explicit | multi_page | paraphrased | absent
    is_answerable: bool
    expected_pages: List[int]
    expected_answer: str
    expected_keywords: List[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvalQuestion:
        return cls(
            id=str(data.get("id", "")),
            question=str(data.get("question", "")),
            category=str(data.get("category", "explicit")),
            is_answerable=bool(data.get("is_answerable", True)),
            expected_pages=[int(p) for p in data.get("expected_pages", [])],
            expected_answer=str(data.get("expected_answer", "")),
            expected_keywords=[str(k) for k in data.get("expected_keywords", [])],
            notes=str(data.get("notes", "")),
        )


@dataclass
class RetrievalItemResult:
    """Evaluation output for a single question's retrieval."""

    question_id: str
    question: str
    category: str
    is_answerable: bool
    expected_pages: List[int]
    retrieved_pages: List[int]
    retrieved_distances: List[float]
    hit_at_1: bool
    hit_at_k: bool
    all_pages_hit: bool
    reciprocal_rank: float
    min_distance: float
    passes_threshold: bool


@dataclass
class RetrievalEvalReport:
    """Aggregated retrieval metrics across all evaluation questions."""

    total_questions: int
    answerable_count: int
    absent_count: int
    top_k: int
    distance_threshold: float
    hit_rate_at_1: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    multi_page_coverage: float
    distance_stats_answerable: Dict[str, float]
    distance_stats_absent: Dict[str, float]
    threshold_analysis: Dict[str, Any]
    items: List[RetrievalItemResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "answerable_count": self.answerable_count,
            "absent_count": self.absent_count,
            "top_k": self.top_k,
            "distance_threshold": self.distance_threshold,
            "hit_rate_at_1": round(self.hit_rate_at_1, 4),
            "hit_rate_at_k": round(self.hit_rate_at_k, 4),
            "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 4),
            "multi_page_coverage": round(self.multi_page_coverage, 4),
            "distance_stats_answerable": {k: round(v, 4) for k, v in self.distance_stats_answerable.items()},
            "distance_stats_absent": {k: round(v, 4) for k, v in self.distance_stats_absent.items()},
            "threshold_analysis": self.threshold_analysis,
            "items": [asdict(item) for item in self.items],
        }


@dataclass
class GenerationItemResult:
    """Evaluation output for a single question's generated answer."""

    question_id: str
    question: str
    category: str
    is_answerable: bool
    expected_pages: List[int]
    actual_mode: str
    actual_answer: str
    cited_pages: List[int]
    citations_correct: bool
    keyword_coverage: float
    missing_keywords: List[str]
    classification: str  # correct | incomplete | unsupported | fallback_correct | fallback_false_negative | fallback_false_positive
    reason: str = ""


@dataclass
class GenerationEvalReport:
    """Aggregated generation metrics across all evaluation questions."""

    skipped: bool
    skip_reason: str = ""
    total_evaluated: int = 0
    correct_count: int = 0
    incomplete_count: int = 0
    unsupported_count: int = 0
    incorrect_count: int = 0
    accuracy: float = 0.0
    citation_accuracy: float = 0.0
    absent_fallback_accuracy: float = 0.0
    real_api_calls: int = 0
    items: List[GenerationItemResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "total_evaluated": self.total_evaluated,
            "correct_count": self.correct_count,
            "incomplete_count": self.incomplete_count,
            "unsupported_count": self.unsupported_count,
            "incorrect_count": self.incorrect_count,
            "accuracy": round(self.accuracy, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "absent_fallback_accuracy": round(self.absent_fallback_accuracy, 4),
            "real_api_calls": self.real_api_calls,
            "items": [asdict(item) for item in self.items],
        }


@dataclass
class TuningResult:
    """Results from evaluating a single configuration preset."""

    chunk_size: int
    chunk_overlap: int
    top_k: int
    distance_threshold: float
    total_chunks: int
    hit_rate_at_1: float
    hit_rate_at_k: float
    mrr: float
    multi_page_coverage: float
    precision: float
    recall: float
    f1_score: float


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset & Document Indexing
# ─────────────────────────────────────────────────────────────────────────────

def load_questions(json_path: Path) -> List[EvalQuestion]:
    """Load and parse evaluation questions from a JSON file.

    Args:
        json_path: Path to questions.json file.

    Returns:
        List[EvalQuestion]: Parsed list of evaluation questions.
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Evaluation questions file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Evaluation questions JSON root must be a list.")

    questions: List[EvalQuestion] = []
    for entry in data:
        q = EvalQuestion.from_dict(entry)
        questions.append(q)

    return questions


def index_document_for_eval(
    pdf_path: Path,
    embedder: E5Embedder,
    persist_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    user_id: str = "eval_user",
    doc_id: str = "eval_doc_001",
    collection_name: str = "eval_chunks",
) -> Tuple[ChromaVectorStore, str, str, int]:
    """Index a PDF into an isolated vector store instance for evaluation.

    Args:
        pdf_path: Path to target PDF file.
        embedder: E5Embedder instance.
        persist_dir: Isolated folder for ChromaDB storage.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap characters between chunks.
        user_id: Evaluation user identifier.
        doc_id: Evaluation document identifier.
        collection_name: Unique collection name.

    Returns:
        Tuple[ChromaVectorStore, str, str, int]:
            (vector_store, user_id, doc_id, total_chunks)
    """
    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(
        pages=pages,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(f"No chunks extracted from {pdf_path}")

    embeddings = embedder.embed_passages([c.text for c in chunks])

    store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    store.add_chunks(
        user_id=user_id,
        chunks=chunks,
        embeddings=embeddings,
        filename=pdf_path.name,
    )

    return store, user_id, doc_id, len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Retrieval Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_retrieval(
    questions: List[EvalQuestion],
    vector_store: ChromaVectorStore,
    embedder: E5Embedder,
    user_id: str,
    doc_id: str,
    top_k: int = 4,
    distance_threshold: float = 0.45,
) -> RetrievalEvalReport:
    """Measure retrieval performance across answerable and absent questions.

    Args:
        questions: List of evaluation test questions.
        vector_store: Vector store containing indexed document chunks.
        embedder: E5Embedder instance for encoding queries.
        user_id: Owner user ID.
        doc_id: Target document ID.
        top_k: Number of retrieved results to inspect.
        distance_threshold: Configured cosine distance cutoff.

    Returns:
        RetrievalEvalReport: Comprehensive metrics on hit rates, coverage, and distances.
    """
    items: List[RetrievalItemResult] = []
    answerable_distances: List[float] = []
    absent_distances: List[float] = []

    answerable_questions = [q for q in questions if q.is_answerable]
    multi_page_questions = [q for q in questions if q.is_answerable and len(q.expected_pages) > 1]
    absent_questions = [q for q in questions if not q.is_answerable]

    for q in questions:
        query_vec = embedder.embed_query(q.question)
        results = vector_store.query(
            user_id=user_id,
            query_embedding=query_vec,
            top_k=top_k,
            doc_id=doc_id,
        )

        retrieved_pages = [r.page for r in results]
        retrieved_distances = [r.distance for r in results]
        min_distance = retrieved_distances[0] if retrieved_distances else 1.0
        passes_threshold = min_distance <= distance_threshold

        if q.is_answerable:
            answerable_distances.append(min_distance)
            hit_at_1 = bool(retrieved_pages and retrieved_pages[0] in q.expected_pages)
            hit_at_k = any(p in q.expected_pages for p in retrieved_pages)
            all_pages_hit = all(p in retrieved_pages for p in q.expected_pages)

            # Calculate Reciprocal Rank of first relevant page
            rr = 0.0
            for rank, page in enumerate(retrieved_pages, start=1):
                if page in q.expected_pages:
                    rr = 1.0 / rank
                    break
            reciprocal_rank = rr
        else:
            absent_distances.append(min_distance)
            hit_at_1 = False
            hit_at_k = False
            all_pages_hit = False
            reciprocal_rank = 0.0

        items.append(
            RetrievalItemResult(
                question_id=q.id,
                question=q.question,
                category=q.category,
                is_answerable=q.is_answerable,
                expected_pages=q.expected_pages,
                retrieved_pages=retrieved_pages,
                retrieved_distances=retrieved_distances,
                hit_at_1=hit_at_1,
                hit_at_k=hit_at_k,
                all_pages_hit=all_pages_hit,
                reciprocal_rank=reciprocal_rank,
                min_distance=min_distance,
                passes_threshold=passes_threshold,
            )
        )

    # Aggregate Answerable Metrics
    ans_items = [it for it in items if it.is_answerable]
    hit_rate_at_1 = (sum(1 for it in ans_items if it.hit_at_1) / len(ans_items)) if ans_items else 0.0
    hit_rate_at_k = (sum(1 for it in ans_items if it.hit_at_k) / len(ans_items)) if ans_items else 0.0
    mean_rr = (sum(it.reciprocal_rank for it in ans_items) / len(ans_items)) if ans_items else 0.0

    mp_items = [it for it in items if it.is_answerable and len(it.expected_pages) > 1]
    multi_page_coverage = (
        (sum(1 for it in mp_items if it.all_pages_hit) / len(mp_items)) if mp_items else 1.0
    )

    # Distance distributions
    def _calc_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
        }

    stats_answerable = _calc_stats(answerable_distances)
    stats_absent = _calc_stats(absent_distances)

    # Threshold analysis across candidate cutoffs
    candidate_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    threshold_analysis: Dict[str, Any] = {}

    for thresh in candidate_thresholds:
        tp = sum(1 for d in answerable_distances if d <= thresh)
        fn = sum(1 for d in answerable_distances if d > thresh)
        tn = sum(1 for d in absent_distances if d > thresh)
        fp = sum(1 for d in absent_distances if d <= thresh)

        total = len(questions)
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        threshold_analysis[f"{thresh:.2f}"] = {
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    return RetrievalEvalReport(
        total_questions=len(questions),
        answerable_count=len(ans_items),
        absent_count=len(absent_questions),
        top_k=top_k,
        distance_threshold=distance_threshold,
        hit_rate_at_1=hit_rate_at_1,
        hit_rate_at_k=hit_rate_at_k,
        mean_reciprocal_rank=mean_rr,
        multi_page_coverage=multi_page_coverage,
        distance_stats_answerable=stats_answerable,
        distance_stats_absent=stats_absent,
        threshold_analysis=threshold_analysis,
        items=items,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM Answer Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page_citations(answer_text: str, sources: List[Any]) -> List[int]:
    """Extract all referenced page numbers from text citations and sources list."""
    pages: Set[int] = set()

    # From explicit source objects
    for s in sources:
        p = getattr(s, "page", None)
        if isinstance(p, int):
            pages.add(p)
        elif isinstance(s, dict) and "page" in s:
            pages.add(int(s["page"]))

    # From regex citations like (Page 1) or [Page 2] or Page 1
    matches = re.findall(r"(?:Page|पान)\s*[:#]?\s*(\d+)", answer_text, flags=re.IGNORECASE)
    for m in matches:
        pages.add(int(m))

    return sorted(pages)


def classify_answer(
    question: EvalQuestion,
    response: RAGResponse,
) -> Tuple[str, bool, float, List[str], str]:
    """Classify the generated answer for correctness, citation integrity, and grounding.

    Returns:
        Tuple[classification, citations_correct, keyword_coverage, missing_keywords, reason]
    """
    answer_text = response.answer
    mode = response.mode
    cited_pages = _extract_page_citations(answer_text, response.sources)

    # 1. Handling of Absent Questions (should fallback to general mode)
    if not question.is_answerable:
        if mode == "general":
            return "fallback_correct", True, 1.0, [], "Correctly identified as absent; answered via general fallback."
        else:
            return (
                "fallback_false_positive",
                False,
                0.0,
                [],
                "Failed to reject absent question: hallucinated document answer instead of falling back.",
            )

    # 2. Handling of Answerable Questions
    # If the pipeline returned general mode for an answerable question:
    if mode == "general":
        return (
            "fallback_false_negative",
            False,
            0.0,
            question.expected_keywords,
            "Erroneously fell back to general mode for an answerable document question.",
        )

    # Verify Citations
    citations_correct = True
    if question.expected_pages:
        # Cited pages must intersect with expected pages, and must not cite nonexistent pages
        has_expected = any(p in question.expected_pages for p in cited_pages)
        has_invalid = any(p not in question.expected_pages for p in cited_pages)
        if not has_expected:
            citations_correct = False
        elif has_invalid and len(cited_pages) > len(question.expected_pages):
            citations_correct = False

    # Check Keyword / Fact Coverage
    missing_keywords: List[str] = []
    lower_answer = answer_text.lower()
    for kw in question.expected_keywords:
        if kw.lower() not in lower_answer:
            missing_keywords.append(kw)

    keyword_coverage = (
        (len(question.expected_keywords) - len(missing_keywords)) / len(question.expected_keywords)
        if question.expected_keywords
        else 1.0
    )

    # Final Classification
    if not citations_correct:
        return (
            "unsupported",
            citations_correct,
            keyword_coverage,
            missing_keywords,
            f"Incorrect or missing page citations: cited {cited_pages}, expected {question.expected_pages}.",
        )

    if missing_keywords:
        return (
            "incomplete",
            citations_correct,
            keyword_coverage,
            missing_keywords,
            f"Answer is missing key facts: {missing_keywords}.",
        )

    return "correct", citations_correct, keyword_coverage, [], "Answer is grounded, factually complete, and cites the correct page(s)."


def evaluate_generation(
    questions: List[EvalQuestion],
    vector_store: ChromaVectorStore,
    embedder: E5Embedder,
    llm: Optional[BaseLLMClient],
    user_id: str,
    doc_id: str,
    top_k: int = 4,
    distance_threshold: float = 0.45,
) -> GenerationEvalReport:
    """Evaluate real LLM answers against expected answers and citations.

    Provider-agnostic: uses whichever LLM provider is configured (Grok, Gemini, etc.).
    If no real LLM client with a valid API key is available, marks evaluation as skipped.
    Never uses MockLLMClient to claim real quality.

    Args:
        questions: Test question dataset.
        vector_store: Vector store with indexed document.
        embedder: E5Embedder instance.
        llm: BaseLLMClient instance (must be a real provider client, or None).
        user_id: Evaluation user ID.
        doc_id: Evaluation document ID.
        top_k: Number of retrieved chunks.
        distance_threshold: Configured distance threshold.

    Returns:
        GenerationEvalReport: Structured answer evaluation report.
    """
    settings = get_settings()

    # Check if a real LLM client with a valid API key is available
    if llm is None:
        # Try to auto-detect the configured provider and instantiate
        provider = settings.LLM_PROVIDER.lower()
        if provider == "grok":
            if not settings.GROK_API_KEY.strip():
                return GenerationEvalReport(
                    skipped=True,
                    skip_reason=(
                        "GROK_API_KEY is not configured. Real LLM answer generation evaluation was skipped. "
                        "(MockLLMClient is not used to claim real answer quality per project requirements)."
                    ),
                )
            try:
                llm = get_llm_client(provider="grok")
            except Exception as exc:
                return GenerationEvalReport(
                    skipped=True,
                    skip_reason=f"Failed to initialize GrokLLMClient: {exc}",
                )
        elif provider == "gemini":
            if not settings.GEMINI_API_KEY.strip():
                return GenerationEvalReport(
                    skipped=True,
                    skip_reason=(
                        "GEMINI_API_KEY is not configured. Real LLM answer generation evaluation was skipped. "
                        "(MockLLMClient is not used to claim real answer quality per project requirements)."
                    ),
                )
            try:
                llm = get_llm_client(provider="gemini")
            except Exception as exc:
                return GenerationEvalReport(
                    skipped=True,
                    skip_reason=f"Failed to initialize GeminiLLMClient: {exc}",
                )
        elif provider == "openrouter":
            if not settings.OPENROUTER_API_KEY.strip():
                return GenerationEvalReport(
                    skipped=True,
                    skip_reason=(
                        "OPENROUTER_API_KEY is not configured. Real LLM answer generation evaluation was skipped. "
                        "(MockLLMClient is not used to claim real answer quality per project requirements)."
                    ),
                )
            try:
                llm = get_llm_client(provider="openrouter")
            except Exception as exc:
                return GenerationEvalReport(
                    skipped=True,
                    skip_reason=f"Failed to initialize OpenRouterLLMClient: {exc}",
                )
        else:
            return GenerationEvalReport(
                skipped=True,
                skip_reason=(
                    f"LLM_PROVIDER='{provider}' is not supported for evaluation. "
                    "Supported providers: 'openrouter', 'grok', 'gemini'."
                ),
            )

    # Verify the client is not a mock claiming real quality
    if llm.__class__.__name__ == "MockLLMClient":
        return GenerationEvalReport(
            skipped=True,
            skip_reason="MockLLMClient detected. Mock clients are forbidden for real answer evaluation.",
        )

    # Track real API calls
    real_api_calls = 0

    items: List[GenerationItemResult] = []
    for q in questions:
        try:
            response = answer_question(
                question=q.question,
                user_id=user_id,
                doc_id=doc_id,
                top_k=top_k,
                distance_threshold=distance_threshold,
                vector_store=vector_store,
                embedder=embedder,
                llm=llm,
            )
            # Each answer_question call with a real LLM makes 1-2 API calls
            # (1 for grounded, possibly +1 for fallback)
            if q.is_answerable:
                real_api_calls += 1  # grounded call
            else:
                real_api_calls += 2  # grounded (NOT_FOUND) + fallback call
        except PermissionError as exc:
            real_api_calls += 1
            logger.error("Authentication failed during answer generation: %s", exc)
            return GenerationEvalReport(
                skipped=True,
                skip_reason=(
                    f"LLM authentication failed: {exc}. "
                    "Evaluation aborted to prevent unnecessary API calls. "
                    "Please update GROK_API_KEY in your .env file with a valid key."
                ),
                real_api_calls=real_api_calls,
            )
        except ConnectionError as exc:
            real_api_calls += 1
            logger.error("Network connectivity error during answer generation: %s", exc)
            return GenerationEvalReport(
                skipped=True,
                skip_reason=(
                    f"Network error connecting to LLM provider: {exc}. "
                    "Evaluation aborted to prevent unnecessary API retries."
                ),
                real_api_calls=real_api_calls,
            )
        except Exception as exc:
            logger.error("Error generating answer for question '%s': %s", q.id, exc)
            real_api_calls += 1  # Count the failed attempt
            items.append(
                GenerationItemResult(
                    question_id=q.id,
                    question=q.question,
                    category=q.category,
                    is_answerable=q.is_answerable,
                    expected_pages=q.expected_pages,
                    actual_mode="error",
                    actual_answer=f"Error: {exc}",
                    cited_pages=[],
                    citations_correct=False,
                    keyword_coverage=0.0,
                    missing_keywords=q.expected_keywords,
                    classification="incorrect",
                    reason=f"Pipeline exception during generation: {exc}",
                )
            )
            continue

        cited_pages = _extract_page_citations(response.answer, response.sources)
        classification, citations_correct, kw_cov, missing_kw, reason = classify_answer(q, response)

        items.append(
            GenerationItemResult(
                question_id=q.id,
                question=q.question,
                category=q.category,
                is_answerable=q.is_answerable,
                expected_pages=q.expected_pages,
                actual_mode=response.mode,
                actual_answer=response.answer,
                cited_pages=cited_pages,
                citations_correct=citations_correct,
                keyword_coverage=kw_cov,
                missing_keywords=missing_kw,
                classification=classification,
                reason=reason,
            )
        )

    # Metrics
    total = len(items)
    correct_count = sum(1 for it in items if it.classification in ("correct", "fallback_correct"))
    incomplete_count = sum(1 for it in items if it.classification == "incomplete")
    unsupported_count = sum(1 for it in items if it.classification == "unsupported")
    incorrect_count = sum(
        1 for it in items if it.classification in ("incorrect", "fallback_false_negative", "fallback_false_positive")
    )

    accuracy = correct_count / total if total > 0 else 0.0

    ans_items = [it for it in items if it.is_answerable]
    citation_accuracy = (
        sum(1 for it in ans_items if it.citations_correct) / len(ans_items) if ans_items else 0.0
    )

    abs_items = [it for it in items if not it.is_answerable]
    fallback_accuracy = (
        sum(1 for it in abs_items if it.classification == "fallback_correct") / len(abs_items)
        if abs_items
        else 1.0
    )

    logger.info("Real LLM API calls made during evaluation: %d", real_api_calls)

    return GenerationEvalReport(
        skipped=False,
        skip_reason="",
        total_evaluated=total,
        correct_count=correct_count,
        incomplete_count=incomplete_count,
        unsupported_count=unsupported_count,
        incorrect_count=incorrect_count,
        accuracy=accuracy,
        citation_accuracy=citation_accuracy,
        absent_fallback_accuracy=fallback_accuracy,
        items=items,
        real_api_calls=real_api_calls,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Parameter Tuning Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_tuning_grid(
    questions: List[EvalQuestion],
    pdf_path: Path,
    embedder: E5Embedder,
) -> List[TuningResult]:
    """Run a systematic tuning experiment across chunk sizes, overlaps, top-k, and thresholds.

    Evaluates:
      - Chunk sizes: 250, 500
      - Overlaps: 50, 100
      - Top-k: 2, 4
      - Distance thresholds: 0.35, 0.45, 0.55

    Args:
        questions: Evaluation question dataset.
        pdf_path: Path to sample PDF.
        embedder: Pre-initialized E5Embedder instance.

    Returns:
        List[TuningResult]: Sorted performance summary per configuration.
    """
    results: List[TuningResult] = []
    temp_dir = tempfile.mkdtemp(prefix="rag_tuning_")

    try:
        # Test configurations
        chunk_configs = [
            (500, 100),  # Baseline default
            (250, 50),   # Finer granularity
        ]
        retrieval_configs = [
            (2, 0.45),
            (4, 0.45),  # Baseline default
            (4, 0.35),  # Stricter threshold
            (4, 0.55),  # Looser threshold
        ]

        for chunk_size, chunk_overlap in chunk_configs:
            col_name = f"tune_{chunk_size}_{chunk_overlap}"
            store, user_id, doc_id, total_chunks = index_document_for_eval(
                pdf_path=pdf_path,
                embedder=embedder,
                persist_dir=temp_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection_name=col_name,
            )

            for top_k, threshold in retrieval_configs:
                report = evaluate_retrieval(
                    questions=questions,
                    vector_store=store,
                    embedder=embedder,
                    user_id=user_id,
                    doc_id=doc_id,
                    top_k=top_k,
                    distance_threshold=threshold,
                )

                t_stats = report.threshold_analysis.get(f"{threshold:.2f}", {})
                precision = float(t_stats.get("precision", 0.0))
                recall = float(t_stats.get("recall", 0.0))
                f1 = float(t_stats.get("f1", 0.0))

                results.append(
                    TuningResult(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k,
                        distance_threshold=threshold,
                        total_chunks=total_chunks,
                        hit_rate_at_1=report.hit_rate_at_1,
                        hit_rate_at_k=report.hit_rate_at_k,
                        mrr=report.mean_reciprocal_rank,
                        multi_page_coverage=report.multi_page_coverage,
                        precision=precision,
                        recall=recall,
                        f1_score=f1,
                    )
                )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Report Formatting & CLI Output
# ─────────────────────────────────────────────────────────────────────────────

def print_retrieval_report(report: RetrievalEvalReport, verbose: bool = False) -> None:
    """Print a clean, structured retrieval report to stdout."""
    sep = "=" * 76
    sub_sep = "-" * 76

    print(f"\n{sep}")
    print("PHASE 4: RAG RETRIEVAL EVALUATION REPORT")
    print(sep)
    print(f"Total Questions Evaluated : {report.total_questions}")
    print(f"  - Answerable Questions  : {report.answerable_count} (explicit, paraphrased, multi-page)")
    print(f"  - Absent Questions      : {report.absent_count} (unanswerable / off-topic)")
    print(f"Retrieval Settings        : Top-k = {report.top_k} | Distance Threshold = {report.distance_threshold:.2f}")
    print(sub_sep)
    print("CORE RETRIEVAL METRICS (Answerable Questions):")
    print(f"  * Top-1 Hit Rate (P@1)  : {report.hit_rate_at_1 * 100:.1f}%")
    print(f"  * Top-k Hit Rate (R@k)  : {report.hit_rate_at_k * 100:.1f}%")
    print(f"  * Mean Reciprocal Rank  : {report.mean_reciprocal_rank:.4f}")
    print(f"  * Multi-page Coverage   : {report.multi_page_coverage * 100:.1f}%")
    print(sub_sep)
    print("COSINE DISTANCE DISTRIBUTIONS:")
    print("  Answerable Questions    : "
          f"min={report.distance_stats_answerable['min']:.4f}, "
          f"mean={report.distance_stats_answerable['mean']:.4f}, "
          f"max={report.distance_stats_answerable['max']:.4f}")
    print("  Absent / Off-topic Qs   : "
          f"min={report.distance_stats_absent['min']:.4f}, "
          f"mean={report.distance_stats_absent['mean']:.4f}, "
          f"max={report.distance_stats_absent['max']:.4f}")
    print(sub_sep)
    print("DISTANCE THRESHOLD GATEKEEPER ANALYSIS:")
    print(f"{'Threshold':<11} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)
    for thresh_str, metrics in report.threshold_analysis.items():
        print(
            f"{thresh_str:<11} | "
            f"{metrics['accuracy'] * 100:>6.1f}%    | "
            f"{metrics['precision'] * 100:>6.1f}%    | "
            f"{metrics['recall'] * 100:>6.1f}%    | "
            f"{metrics['f1'] * 100:>6.1f}%"
        )
    print(sep)

    if verbose:
        print("\nDETAILED PER-QUESTION RETRIEVAL BREAKDOWN:")
        for it in report.items:
            status = "HIT" if it.hit_at_k else ("REJECTED (Correct)" if not it.is_answerable else "MISS")
            print(f"[{it.question_id}] ({it.category:<11}) [{status:<18}] dist={it.min_distance:.4f}")
            print(f"    Q   : {it.question}")
            print(f"    Exp : Pages {it.expected_pages} | Ret: Pages {it.retrieved_pages}")


def print_generation_report(report: GenerationEvalReport, verbose: bool = False) -> None:
    """Print the LLM answer evaluation report to stdout."""
    sep = "=" * 76
    sub_sep = "-" * 76

    print(f"\n{sep}")
    print("PHASE 4: REAL LLM ANSWER GENERATION REPORT")
    print(sep)

    if report.skipped:
        print("[LLM EVALUATION SKIPPED]")
        print(f"Reason: {report.skip_reason}")
        print("Note: MockLLMClient was NOT used to fake answer quality.")
        print(sep)
        return

    settings = get_settings()
    provider_name = settings.LLM_PROVIDER.upper()
    model_name = settings.LLM_MODEL or "(default)"

    print(f"LLM Provider              : {provider_name} ({model_name})")
    print(f"Total Answers Evaluated   : {report.total_evaluated}")
    print(f"Real API Calls Made       : {report.real_api_calls}")
    print(f"  * Grounded / Correct    : {report.correct_count} ({report.accuracy * 100:.1f}%)")
    print(f"  * Incomplete Facts      : {report.incomplete_count}")
    print(f"  * Unsupported/Bad Cite  : {report.unsupported_count}")
    print(f"  * Incorrect / False Neg : {report.incorrect_count}")
    print(sub_sep)
    print("KEY QUALITY RATES:")
    print(f"  * Overall Answer Accuracy    : {report.accuracy * 100:.1f}%")
    print(f"  * Page Citation Accuracy     : {report.citation_accuracy * 100:.1f}%")
    print(f"  * Absent Fallback Accuracy   : {report.absent_fallback_accuracy * 100:.1f}%")
    print(sep)

    if verbose:
        print("\nDETAILED PER-QUESTION LLM BREAKDOWN:")
        for it in report.items:
            print(f"[{it.question_id}] Classification: {it.classification.upper()}")
            print(f"    Q        : {it.question}")
            print(f"    Mode     : {it.actual_mode} | Cited: {it.cited_pages} (Exp: {it.expected_pages})")
            print(f"    Answer   : {it.actual_answer[:120]}...")
            if it.reason:
                print(f"    Feedback : {it.reason}")


def print_tuning_report(results: List[TuningResult]) -> None:
    """Print the parameter tuning grid comparison table."""
    sep = "=" * 88
    print(f"\n{sep}")
    print("PHASE 4: PARAMETER TUNING GRID COMPARISON")
    print(sep)
    header = (
        f"{'Chunk/Ovlp':<12} | {'Chunks':<6} | {'Top-K':<5} | {'Thresh':<6} | "
        f"{'Hit@1':<7} | {'Hit@K':<7} | {'MRR':<7} | {'Coverage':<8} | {'F1':<6}"
    )
    print(header)
    print("-" * 88)
    for r in results:
        cfg = f"{r.chunk_size}/{r.chunk_overlap}"
        print(
            f"{cfg:<12} | {r.total_chunks:<6} | {r.top_k:<5} | {r.distance_threshold:<6.2f} | "
            f"{r.hit_rate_at_1 * 100:>5.1f}% | {r.hit_rate_at_k * 100:>5.1f}% | "
            f"{r.mrr:>6.4f} | {r.multi_page_coverage * 100:>6.1f}%  | {r.f1_score * 100:>5.1f}%"
        )
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    questions_path: Path,
    pdf_path: Path,
    top_k: int = 4,
    distance_threshold: float = 0.45,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    retrieval_only: bool = False,
    run_tuning: bool = False,
    api_key: Optional[str] = None,
    output_json: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[RetrievalEvalReport, Optional[GenerationEvalReport], Optional[List[TuningResult]]]:
    """Execute the complete Phase 4 evaluation workflow."""
    questions = load_questions(questions_path)
    embedder = get_embedder()

    # Create isolated temp folder for Chroma evaluation storage
    temp_dir = tempfile.mkdtemp(prefix="rag_eval_chroma_")

    try:
        # Index target document in isolated store
        store, user_id, doc_id, total_chunks = index_document_for_eval(
            pdf_path=pdf_path,
            embedder=embedder,
            persist_dir=temp_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 1. Retrieval Evaluation
        retrieval_report = evaluate_retrieval(
            questions=questions,
            vector_store=store,
            embedder=embedder,
            user_id=user_id,
            doc_id=doc_id,
            top_k=top_k,
            distance_threshold=distance_threshold,
        )
        print_retrieval_report(retrieval_report, verbose=verbose)

        # 2. Generation Evaluation (unless retrieval_only is set)
        generation_report: Optional[GenerationEvalReport] = None
        if not retrieval_only:
            llm_client: Optional[BaseLLMClient] = None
            settings = get_settings()
            provider = settings.LLM_PROVIDER.lower()

            # If an explicit API key override was provided, try to instantiate
            # the configured provider with that key.
            if api_key and api_key.strip():
                try:
                    if provider == "openrouter":
                        llm_client = OpenRouterLLMClient(api_key=api_key)
                    elif provider == "grok":
                        llm_client = GrokLLMClient(api_key=api_key)
                    elif provider == "gemini":
                        llm_client = GeminiLLMClient(api_key=api_key)
                    else:
                        logger.warning("Unsupported provider '%s' for --api-key override.", provider)
                except Exception as exc:
                    logger.warning("Could not instantiate %s LLM client: %s", provider, exc)

            # If no override, evaluate_generation will auto-detect from .env
            generation_report = evaluate_generation(
                questions=questions,
                vector_store=store,
                embedder=embedder,
                llm=llm_client,
                user_id=user_id,
                doc_id=doc_id,
                top_k=top_k,
                distance_threshold=distance_threshold,
            )
            print_generation_report(generation_report, verbose=verbose)

        # 3. Parameter Tuning Grid (if requested)
        tuning_results: Optional[List[TuningResult]] = None
        if run_tuning:
            print("\nRunning systematic parameter tuning sweep across configurations...")
            tuning_results = run_tuning_grid(
                questions=questions,
                pdf_path=pdf_path,
                embedder=embedder,
            )
            print_tuning_report(tuning_results)

        # 4. Optional JSON Export
        if output_json:
            export_data: Dict[str, Any] = {
                "retrieval": retrieval_report.to_dict(),
                "generation": generation_report.to_dict() if generation_report else None,
                "tuning": [asdict(t) for t in tuning_results] if tuning_results else None,
            }
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"\nFull evaluation report saved to: {output_json}")

        return retrieval_report, generation_report, tuning_results

    finally:
        # Safe cleanup of temporary Chroma persistence files
        try:
            store.client._system.stop()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: RAG Evaluation and Tuning Runner for WhatsApp Land Records",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "eval" / "questions.json",
        help="Path to evaluation questions JSON file (default: eval/questions.json)",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_pdfs" / "sample_land_record.pdf",
        help="Path to sample PDF (default: data/sample_pdfs/sample_land_record.pdf)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=get_settings().TOP_K,
        help=f"Number of retrieved chunks (default: {get_settings().TOP_K})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=get_settings().SIMILARITY_DISTANCE_THRESHOLD,
        help=f"Cosine distance threshold (default: {get_settings().SIMILARITY_DISTANCE_THRESHOLD})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=get_settings().CHUNK_SIZE,
        help=f"Chunk character size (default: {get_settings().CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=get_settings().CHUNK_OVERLAP,
        help=f"Chunk overlap character size (default: {get_settings().CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run only retrieval evaluation, skipping LLM answer evaluation",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run parameter tuning grid sweep across chunk sizes, overlaps, and thresholds",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API Key override for the configured provider (never logged)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to output complete JSON results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose question-by-question breakdown",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        questions_path=args.questions,
        pdf_path=args.pdf,
        top_k=args.top_k,
        distance_threshold=args.threshold,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retrieval_only=args.retrieval_only,
        run_tuning=args.tune,
        api_key=args.api_key,
        output_json=args.output,
        verbose=args.verbose,
    )
