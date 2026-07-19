"""
ManufacturingIQ Agentic AI - Retriever  (H-3)

FAISS-backed semantic retriever with hybrid scoring over the knowledge corpus.

H-3 improvements over original:
  - Hybrid retrieval: final score = α × semantic_score + (1−α) × keyword_score
  - Minimum confidence threshold: documents below MIN_SCORE filtered out
  - Citation and tags passthrough for source attribution
  - Index invalidation via reset() for corpus updates
  - Persistent embeddings/index saved to disk for fast startup
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from knowledge.corpus import get_all_documents
from state.schema import RetrievedDocument

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_HYBRID_ALPHA = float(os.getenv("RETRIEVER_HYBRID_ALPHA", "0.75"))
_MIN_SCORE = float(os.getenv("RETRIEVER_MIN_SCORE", "0.15"))

CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_DIR", "models/embeddings"))

_embedder = None
_preloaded = False


def _get_embedder():
    """Singleton SentenceTransformer loader with offline cache support."""
    global _embedder
    if _embedder is not None:
        return _embedder

    # Enforce local-only mode to eliminate Hugging Face HTTP requests
    import os as _os
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from sentence_transformers import SentenceTransformer
    # Try offline loading first (NO HTTP REQUESTS)
    cache_path = CACHE_DIR / f"{_EMBEDDING_MODEL.replace('/', '_')}_offline"
    if cache_path.exists():
        logger.info("Loading SentenceTransformer from offline cache: %s", cache_path)
        _embedder = SentenceTransformer(str(cache_path), cache_folder=str(CACHE_DIR), local_files_only=True)
    else:
        logger.info("Loading SentenceTransformer from cache: %s", _EMBEDDING_MODEL)
        _embedder = SentenceTransformer(_EMBEDDING_MODEL, cache_folder=str(CACHE_DIR), local_files_only=True)
    return _embedder


def _embed(texts: List[str]) -> np.ndarray:
    return _get_embedder().encode(texts, normalize_embeddings=True)


def _corpus_hash() -> str:
    """Generate hash of corpus content for cache invalidation."""
    docs = get_all_documents()
    content = json.dumps([d.get("text", "") for d in docs], sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Keyword scoring (BM25-inspired, no external deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _keyword_score(query_tokens: set[str], doc: Dict[str, Any]) -> float:
    """
    Soft term-overlap score against document title + tags.

    Score = (matched_terms) / sqrt(query_terms * doc_terms)
    Capped at 1.0.  Returns 0.0 if either set is empty.
    """
    doc_tokens = _tokenize(doc.get("title", ""))
    for tag in doc.get("tags", []):
        doc_tokens |= _tokenize(tag)

    if not query_tokens or not doc_tokens:
        return 0.0

    overlap = len(query_tokens & doc_tokens)
    denom = (len(query_tokens) * len(doc_tokens)) ** 0.5
    return min(1.0, overlap / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# FaissRetriever
# ---------------------------------------------------------------------------

class FaissRetriever:
    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.index = None
        self._embeddings: Optional[np.ndarray] = None
        self._built = False

    # ------------------------------------------------------------------
    def _cache_paths(self) -> tuple[Path, Path, Path]:
        """Get paths for embeddings, index, and corpus hash."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return (
            CACHE_DIR / f"embeddings_{_corpus_hash()}.npy",
            CACHE_DIR / f"faiss_{_corpus_hash()}.index",
            CACHE_DIR / f"corpus_hash.txt",
        )

    # ------------------------------------------------------------------
    def preload(self) -> "FaissRetriever":
        """
        Pre-initialize embeddings and FAISS index during FastAPI startup.
        Called once at application launch instead of lazily on first request.
        """
        global _preloaded
        if _preloaded:
            logger.debug("Retriever already preloaded, skipping")
            return self

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        emb_path, idx_path, hash_path = self._cache_paths()

        # Load from cache if exists
        if emb_path.exists() and idx_path.exists() and hash_path.exists():
            try:
                self._embeddings = np.load(emb_path)
                corpus_hash = _corpus_hash()
                with open(hash_path, "r") as f:
                    cached_hash = f.read().strip()
                if cached_hash == corpus_hash:
                    import faiss
                    self.index = faiss.read_index(str(idx_path))
                    self.documents = get_all_documents()
                    self._built = True
                    _preloaded = True
                    logger.info("Loaded retriever from cache: %d documents", len(self.documents))
                    return self
            except Exception as exc:
                logger.warning("Cache load failed, rebuilding: %s", exc)

        # Build fresh
        self.build_index()
        self.save_index()
        _preloaded = True
        return self

    # ------------------------------------------------------------------
    def build_index(self) -> None:
        """Build embeddings and FAISS index from corpus."""
        docs = get_all_documents()
        texts = [d["text"] for d in docs]
        self.documents = docs
        embeddings = _embed(texts)

        try:
            import faiss
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype("float32"))
            self.index = index
            self._embeddings = embeddings  # Keep for fallback
            logger.info("FAISS index built: %d documents, dim=%d", len(docs), dim)
        except Exception as exc:
            logger.warning("FAISS not available, using brute-force cosine fallback: %s", exc)
            self.index = None
            self._embeddings = embeddings

        self._built = True

    # ------------------------------------------------------------------
    def save_index(self) -> None:
        """Save embeddings and FAISS index to disk for fast reloads."""
        if self._embeddings is None:
            return
        emb_path, idx_path, hash_path = self._cache_paths()
        np.save(emb_path, self._embeddings)
        with open(hash_path, "w") as f:
            f.write(_corpus_hash())
        try:
            import faiss
            if self.index is not None:
                faiss.write_index(self.index, str(idx_path))
                logger.info("Saved retriever cache to %s", CACHE_DIR)
        except Exception as exc:
            logger.warning("Failed to save FAISS index: %s", exc)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Force index rebuild on next retrieve() call (useful after corpus updates)."""
        global _preloaded
        self.documents = []
        self.index = None
        self._embeddings = None
        self._built = False
        _preloaded = False

    # ------------------------------------------------------------------
    def _ensure_built(self) -> None:
        if self._built:
            return
        self.preload()  # Use preload instead of rebuild

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = _MIN_SCORE,
        alpha: float = _HYBRID_ALPHA,
    ) -> List[RetrievedDocument]:
        """
        Hybrid semantic + keyword retrieval.

        Parameters
        ----------
        query : str
            Natural-language query string (built from prediction context).
        top_k : int
            Maximum number of documents to return.
        min_score : float
            Minimum hybrid score [0, 1] — documents below this are discarded.
        alpha : float
            Semantic weight in [0, 1]; keyword weight is (1 - alpha).

        Returns
        -------
        List[RetrievedDocument]
            Sorted by hybrid score descending, at most top_k documents with
            score >= min_score.
        """
        self._ensure_built()

        q_emb = _embed([query])                         # shape (1, dim)
        q_tokens = _tokenize(query)
        n_docs = len(self.documents)

        # --- Semantic scores -------------------------------------------
        if self.index is not None:
            import faiss
            # Retrieve more than top_k so hybrid re-ranking can reorder
            k = min(n_docs, max(top_k * 3, 10))
            sem_scores_arr, idxs_arr = self.index.search(q_emb.astype("float32"), k)
            sem_scores = {int(idx): float(score)
                          for idx, score in zip(idxs_arr[0], sem_scores_arr[0])
                          if 0 <= idx < n_docs}
        else:
            embs = self._embeddings
            if embs is None:
                return []
            sims = (embs @ q_emb[0]).tolist()
            sem_scores = {i: float(sims[i]) for i in range(n_docs)}

        # Normalise semantic scores to [0, 1] (inner-product with normalised
        # vectors is already in [-1, 1]; shift and scale to [0, 1])
        if sem_scores:
            max_s = max(sem_scores.values())
            min_s = min(sem_scores.values())
            rng = max(max_s - min_s, 1e-6)
            sem_scores = {i: (v - min_s) / rng for i, v in sem_scores.items()}

        # --- Hybrid scoring -------------------------------------------
        candidates = []
        for idx, sem in sem_scores.items():
            kw = _keyword_score(q_tokens, self.documents[idx])
            hybrid = alpha * sem + (1.0 - alpha) * kw
            candidates.append((hybrid, idx))

        candidates.sort(reverse=True)

        # --- Build results with min-score filter ----------------------
        results: List[RetrievedDocument] = []
        for hybrid_score, idx in candidates:
            if len(results) >= top_k:
                break
            if hybrid_score < min_score:
                # Sorted descending — once below threshold, all remaining are too
                break
            doc = self.documents[idx]
            results.append(
                RetrievedDocument(
                    title=doc.get("title", ""),
                    source=doc.get("source", ""),
                    section=doc.get("section"),
                    excerpt=doc.get("text", ""),
                    confidence=round(hybrid_score, 3),
                    citation=doc.get("citation"),          # H-3: source attribution
                    tags=doc.get("tags", []),              # H-3: keyword metadata
                )
            )

        if not results:
            logger.info(
                "Retriever: no documents met min_score=%.2f for query %r; "
                "top raw hybrid score was %.3f",
                min_score,
                query[:80],
                candidates[0][0] if candidates else 0.0,
            )

        return results


# Module-level singleton
retriever = FaissRetriever()