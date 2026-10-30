"""Official LlamaIndex splitter using the project's existing embedding instance."""

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@lru_cache(maxsize=1)
def get_parser():
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from semantic_similarity import model
    from sentence_splitter import split_sentences

    class SharedEmbedding(BaseEmbedding):
        def _get_text_embedding(self, text):
            return model.encode(text, convert_to_numpy=True).tolist()

        def _get_query_embedding(self, query):
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query):
            return self._get_query_embedding(query)

        def _get_text_embeddings(self, texts):
            return model.encode(texts, convert_to_numpy=True,
                                show_progress_bar=False).tolist()

    # LlamaIndex concatenates sentence strings without inserting separators.
    def sentences_with_spaces(text):
        sentences = split_sentences(text)
        return [s + (" " if i < len(sentences) - 1 else "")
                for i, s in enumerate(sentences)]

    return SemanticSplitterNodeParser(
        embed_model=SharedEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        sentence_splitter=sentences_with_spaces,
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        include_metadata=False,
        include_prev_next_rel=False,
    )


def llamaindex_semantic_chunk(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    from llama_index.core.schema import Document

    nodes = get_parser().get_nodes_from_documents([Document(text=text)])
    return [node.text.strip() for node in nodes if node.text.strip()]
