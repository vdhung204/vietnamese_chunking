"""Max–Min (Kiss, Nagy & Szilagyi, 2025), with the shared Vietnamese inputs.

Reference implementation: https://github.com/hsdslab/MaxMinChunking/blob/
0caba41d2ef30114729f95e5c788173e4f40920e/maxmin_chunker.py
Paper: https://doi.org/10.1007/s10791-025-09638-7
The published code's sigmoid(size - 1) and strict comparison are preserved.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentence_splitter import split_sentences


def process_sentences(
    sentences: list[str],
    embeddings: np.ndarray,
    fixed_threshold: float = 0.6,
    c: float = 0.9,
    init_constant: float = 1.5,
) -> list[list[str]]:
    """Cluster consecutive sentences using the authors' Max–Min decisions.

No overlap or length policy is applied. Empty input is supported in addition
to the reference implementation. Similarity state is stored as scalar floats.
"""
    if not sentences:
        return []
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sentences):
        raise ValueError("embeddings must have shape (number of sentences, dimensions)")

    clusters = []
    start = 0
    minimum = float("-inf")
    for index in range(1, len(sentences)):
        similarities = cosine_similarity(
            embeddings[index:index + 1], embeddings[start:index]
        )[0]
        size = index - start
        if size == 1:
            minimum = float(similarities[0])
            score = init_constant * minimum
            adaptive_threshold = 0.0
        else:
            # Use the existing cluster minimum BEFORE including the candidate.
            sigmoid = 1.0 / (1.0 + np.exp(-(size - 1)))
            adaptive_threshold = minimum * c * sigmoid
            score = float(np.max(similarities))
            minimum = min(minimum, float(np.min(similarities)))

        if score > max(fixed_threshold, adaptive_threshold):
            continue
        clusters.append(sentences[start:index])
        start = index
        minimum = float("-inf")

    clusters.append(sentences[start:])
    return clusters


def maxmin_semantic_chunk(text: str) -> list[str]:
    """Split text with the project splitter and shared MiniLM model instance."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    # Lazy import avoids initializing the model for empty inputs/helper tests.
    from semantic_similarity import model

    embeddings = model.encode(
        sentences, convert_to_numpy=True, show_progress_bar=False
    )
    return [" ".join(cluster) for cluster in process_sentences(sentences, embeddings)]
