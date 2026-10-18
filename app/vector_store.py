"""Vector store wrapper for ChromaDB.

What is a Vector Database?
Traditional databases (like SQL or MongoDB) search text by exact keywords or regex.
A Vector Database (like ChromaDB) stores high-dimensional embeddings and indexes them
with Approximate Nearest Neighbor (ANN) algorithms. This allows us to search by semantic
meaning rather than identical keywords.

Why Cosine Distance?
Cosine distance measures the geometric angle between two vector directions:
- Distance = 0.0: Perfect match (vectors point in the exact same direction).
- Distance ~0.2 to 0.4: Highly relevant match.
- Distance > 0.45: Marginally relevant or off-topic.

Multi-Tenant Security & Per-User Isolation:
Land records contain sensitive personal data. Every write, query, and delete MUST include
the user_id in the metadata filter. A query for user A will NEVER return chunks belonging
to user B.
"""

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunker import DocumentChunk
from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single retrieved chunk match with similarity distance."""

    chunk_id: str
    doc_id: str
    page: int
    chunk_index: int
    text: str
    distance: float
    ocr: bool = False


class ChromaVectorStore:
    """ChromaDB persistent client wrapper managing document chunk storage and retrieval."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "chunks",
    ):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.CHROMA_DIR

        # Ensure persistence directory exists on disk
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Create or fetch collection configured for cosine distance.
        # RAG-009: stamp the embedding model into collection metadata so future
        # query-time startups can fail fast on a model/ingestion mismatch.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": settings.EMBEDDING_MODEL,
            },
        )

        # RAG-009: fail fast if the collection was built with a different embedding model.
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != settings.EMBEDDING_MODEL:
            raise RuntimeError(
                f"Embedding model mismatch: collection '{collection_name}' was built with "
                f"'{stored_model}' but EMBEDDING_MODEL is configured as "
                f"'{settings.EMBEDDING_MODEL}'. Re-index the documents with the configured "
                "model or revert EMBEDDING_MODEL."
            )
        if not stored_model:
            # Legacy collection created before stamping existed: never block startup,
            # just note that the model-match check cannot be enforced for it.
            logger.info(
                "Collection '%s' has no embedding-model stamp (legacy); "
                "skipping embedding-model match check.",
                collection_name,
            )

    def add_chunks(
        self,
        user_id: str,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        filename: str = "",
    ) -> None:
        """Store document chunks and their precomputed embeddings with user metadata.

        Args:
            user_id: Owner user ID for access isolation.
            chunks: List of DocumentChunk objects to store.
            embeddings: Corresponding normalized vector embeddings.
            filename: Original PDF filename for reference.

        Raises:
            ValueError: If chunks and embeddings lengths do not match.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "user_id": user_id,
                    "doc_id": chunk.doc_id,
                    "filename": filename,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "ocr": chunk.ocr,
                }
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 4,
        doc_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Retrieve the top-k most similar chunks for a user.

        Strictly enforces user isolation via metadata filtering.

        Args:
            user_id: Identifier of the requesting user.
            query_embedding: Vector representation of the user's question.
            top_k: Maximum number of closest chunks to retrieve.
            doc_id: Optional document ID filter (if querying a specific document).

        Returns:
            List[SearchResult]: List of matches sorted by ascending distance (closest first).
        """
        # Build user isolation filter
        if doc_id:
            where_clause = {
                "$and": [
                    {"user_id": user_id},
                    {"doc_id": doc_id},
                ]
            }
        else:
            where_clause = {"user_id": user_id}

        # Query ChromaDB collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        search_results: List[SearchResult] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) else {}
            dist = float(distances[i]) if i < len(distances) else 1.0
            doc_text = docs[i] if i < len(docs) else ""

            search_results.append(
                SearchResult(
                    chunk_id=ids[i],
                    doc_id=str(meta.get("doc_id", "")),
                    page=int(meta.get("page", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc_text,
                    distance=dist,
                    ocr=bool(meta.get("ocr", False)),
                )
            )

        return search_results

    def get_structural_chunks(
        self,
        user_id: str,
        doc_id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        max_results: int = 30,
    ) -> List[SearchResult]:
        """Retrieve document structure chunks (e.g. INDEX or CHAPTER headings) for structural queries.

        Strictly enforces multi-tenant isolation and document scoping.

        Args:
            user_id: Requesting user identifier.
            doc_id: Optional document ID filter.
            keywords: Target structural keywords (default: ['CHAPTER', 'INDEX']).
            max_results: Maximum structural chunks to retrieve.

        Returns:
            List[SearchResult]: Chronologically sorted structural chunks.
        """
        search_terms = keywords or ["CHAPTER", "INDEX"]
        seen_ids = set()
        structural_results: List[SearchResult] = []

        if doc_id:
            where_clause = {
                "$and": [
                    {"user_id": user_id},
                    {"doc_id": doc_id},
                ]
            }
        else:
            where_clause = {"user_id": user_id}

        for term in search_terms:
            try:
                results = self.collection.get(
                    where=where_clause,
                    where_document={"$contains": term},
                    include=["documents", "metadatas"],
                )
            except Exception:
                continue

            if not results or not results.get("ids"):
                continue

            ids = results["ids"]
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            for i in range(len(ids)):
                cid = ids[i]
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                meta = metas[i] if i < len(metas) else {}
                doc_text = docs[i] if i < len(docs) else ""

                structural_results.append(
                    SearchResult(
                        chunk_id=cid,
                        doc_id=str(meta.get("doc_id", "")),
                        page=int(meta.get("page", 1)),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        text=doc_text,
                        distance=0.0,
                        ocr=bool(meta.get("ocr", False)),
                    )
                )

        # Sort chronologically by physical page number and chunk index
        structural_results.sort(key=lambda r: (r.page, r.chunk_index))
        return structural_results[:max_results]

    def get_chunks_by_page_range(
        self,
        user_id: str,
        doc_id: str,
        start_page: int,
        end_page: int,
        max_results: int = 10,
    ) -> List[SearchResult]:
        """Retrieve chunks within an inclusive physical page range in PDF reading order.

        Used for targeted chapter retrieval: once a chapter heading page is known, its
        content can be fetched deterministically instead of via semantic similarity.
        Strictly enforces multi-tenant isolation and document scoping.

        Args:
            user_id: Requesting user identifier.
            doc_id: Document ID to scope the retrieval.
            start_page: First physical page of the range (inclusive).
            end_page: Last physical page of the range (inclusive).
            max_results: Maximum number of chunks returned, in page order.

        Returns:
            List[SearchResult]: Page-ordered chunks (distance 0.0: exact range match).
        """
        if end_page < start_page:
            return []

        where_clause = {
            "$and": [
                {"user_id": user_id},
                {"doc_id": doc_id},
                {"page": {"$gte": start_page}},
                {"page": {"$lte": end_page}},
            ]
        }

        try:
            results = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("Page-range chunk lookup failed: %s", exc)
            return []

        if not results or not results.get("ids"):
            return []

        ids = results["ids"]
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        search_results: List[SearchResult] = []
        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) else {}
            doc_text = docs[i] if i < len(docs) else ""
            search_results.append(
                SearchResult(
                    chunk_id=ids[i],
                    doc_id=str(meta.get("doc_id", "")),
                    page=int(meta.get("page", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc_text,
                    distance=0.0,
                    ocr=bool(meta.get("ocr", False)),
                )
            )

        search_results.sort(key=lambda r: (r.page, r.chunk_index))
        return search_results[:max_results]

    def delete_document(self, user_id: str, doc_id: str) -> None:
        """Delete all chunks belonging to a specific document and user."""
        where_clause = {
            "$and": [
                {"user_id": user_id},
                {"doc_id": doc_id},
            ]
        }
        self.collection.delete(where=where_clause)



@lru_cache()
def get_vector_store() -> ChromaVectorStore:
    """Return a cached singleton instance of ChromaVectorStore."""
    return ChromaVectorStore()
