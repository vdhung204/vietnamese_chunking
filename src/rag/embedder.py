"""Reuse the chunkers' existing SentenceTransformer instance."""

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SharedEmbedder:
    def __init__(self, model=None):
        if model is None:
            from semantic_similarity import model
        self.model = model

    def encode_texts(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_query(self, query):
        return self.encode_texts([query])[0]
