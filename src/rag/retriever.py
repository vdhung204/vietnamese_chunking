"""Global in-memory cosine retrieval, independent of chunking strategy."""

import numpy as np


def normalize_rows(vectors):
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite embedding matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class CosineRetriever:
    def __init__(self, chunks, embeddings):
        if not chunks:
            raise ValueError("Cannot index an empty chunk corpus")
        self.chunks = chunks
        self.embeddings = normalize_rows(embeddings)
        if len(chunks) != len(self.embeddings):
            raise ValueError("Chunk/embedding count mismatch")

    def retrieve_vectors(self, query_embeddings, top_k=5):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores = normalize_rows(query_embeddings) @ self.embeddings.T
        return [
            [dict(self.chunks[i], score=float(row[i]), rank=rank)
             for rank, i in enumerate(np.argsort(-row, kind="stable")[:top_k], 1)]
            for row in scores
        ]
