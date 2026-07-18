"""Smoke test for the RAG retrieval logic.

Reimplements the small, dependency-light core of the retrieval step (cosine
similarity + top-k + threshold) and checks its shape/threshold behavior, so the
retrieval contract is covered without loading a transformer model or network.
"""
from __future__ import annotations

import numpy as np


def cosine_top_k(
    query_vec: np.ndarray,
    corpus: np.ndarray,
    top_k: int = 3,
    threshold: float = 0.05,
) -> list[tuple[int, float]]:
    """Return up to top_k (index, similarity) pairs above threshold, ranked desc.

    Mirrors the behavior in CreateModel/advanced_agricultural_rag_chatbot.py:
    top_k=3, minimum similarity threshold 0.05.
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
    sims = c @ q
    order = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in order if sims[i] > threshold]


def test_returns_at_most_top_k() -> None:
    corpus = np.eye(10, dtype=float)
    query = np.ones(10, dtype=float)
    hits = cosine_top_k(query, corpus, top_k=3)
    assert len(hits) <= 3


def test_ranked_descending() -> None:
    corpus = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query = np.array([1.0, 0.0])
    hits = cosine_top_k(query, corpus, top_k=3, threshold=0.0)
    sims = [s for _, s in hits]
    assert sims == sorted(sims, reverse=True)
    assert hits[0][0] == 0  # exact match ranks first


def test_threshold_filters_low_similarity() -> None:
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]])
    query = np.array([1.0, 0.0])
    # second vector is orthogonal (similarity 0) -> filtered by threshold 0.05
    hits = cosine_top_k(query, corpus, top_k=3, threshold=0.05)
    assert all(sim > 0.05 for _, sim in hits)
    assert len(hits) == 1
