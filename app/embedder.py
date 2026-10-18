"""Text embedding module using multilingual-e5-base and sentence-transformers.

What is an Embedding?
An embedding converts raw human text into a dense mathematical vector (a list of numbers,
typically 768 floats for e5-base). In this high-dimensional vector space, pieces of text
with similar semantic meaning are placed close to each other, even if they use different
phrases or different languages (e.g., Marathi 'जमीन मालक' and English 'land owner').

Why the 'passage: ' and 'query: ' Prefixes Matter:
The multilingual-e5 family of models is trained as an asymmetric retrieval model:
1. 'passage: ': Tells the embedding model that the text is an indexed piece of document
   knowledge to be retrieved.
2. 'query: ': Tells the embedding model that the text is an inquiry/question looking
   for answers.
Tuning these two different roles allows the model to match questions with the right facts,
rather than just matching words that sound alike.
"""

from functools import lru_cache
import logging
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

# Maximum number of distinct user questions kept in the per-process embedding cache
QUERY_EMBEDDING_CACHE_SIZE: int = 256


class E5Embedder:
    """Wrapper around SentenceTransformer with multilingual-e5 asymmetric prefixes."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL

        # Select hardware device: GPU if available, otherwise CPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "Loading embedding model '%s' on device '%s'...",
            self.model_name,
            self.device,
        )
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_passages(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> List[List[float]]:
        """Compute normalized vector embeddings for document text chunks.

        Prepends the required 'passage: ' prefix to each text chunk before encoding.
        Processes in batches to prevent memory exhaustion on large documents (e.g. 500+ pages).

        Args:
            texts: List of document chunk text strings.
            batch_size: Batch size for encoding chunks (default: 64).

        Returns:
            List[List[float]]: List of 768-dimensional normalized embedding vectors.
        """
        if not texts:
            return []

        # Prepend 'passage: ' prefix as required by multilingual-e5 models
        prefixed_texts = [f"passage: {t.strip()}" for t in texts]

        # normalize_embeddings=True ensures cosine similarity can be computed via dot product
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Compute a normalized vector embedding for a user question.

        Prepends the required 'query: ' prefix before encoding. Results for
        repeated questions are served from a bounded in-process LRU cache.

        Args:
            query: User's search question.

        Returns:
            List[float]: 768-dimensional normalized embedding vector.
        """
        clean_query = query.strip()
        return list(self._embed_query_cached(clean_query))

    @lru_cache(maxsize=QUERY_EMBEDDING_CACHE_SIZE)
    def _embed_query_cached(self, clean_query: str) -> tuple:
        """Encode a cleaned question once and memoize the result.

        embed_query returns a list copy so callers can never mutate the cached
        tuple. The cache is bounded to avoid unbounded memory growth.
        """
        prefixed_query = f"query: {clean_query}"

        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return tuple(embedding.tolist())


@lru_cache()
def get_embedder() -> E5Embedder:
    """Return a cached singleton instance of the E5Embedder.

    Loading neural network weights is expensive; caching the instance ensures
    the model is loaded into memory only once per process.
    """
    return E5Embedder()
