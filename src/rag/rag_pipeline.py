"""One indexing/retrieval/generation pipeline for all injected chunkers."""

from time import perf_counter

from .retriever import CosineRetriever
from evaluation.rag_retrieval import build_rag_prompt


class RAGPipeline:
    def __init__(self, chunker, embedder, llm=None):
        self.chunker = chunker
        self.embedder = embedder
        self.llm = llm
        self.retriever = None

    def index_documents(self, documents):
        chunks = []
        seen = set()
        start = perf_counter()
        for document in documents:
            doc_id = document['id']
            if doc_id in seen or not document['text'].strip():
                raise ValueError("Documents require unique IDs and nonempty text")
            seen.add(doc_id)
            output = self.chunker(document['text'])
            if not isinstance(output, list) or not output or any(
                not isinstance(chunk, str) or not chunk.strip() for chunk in output
            ):
                raise ValueError("Chunker must return nonempty list[str] for nonempty text")
            chunks.extend(dict(document_id=doc_id, chunk_id=f"{doc_id}:{i}",
                               chunk_index=i, text=text) for i, text in enumerate(output))
        self.chunking_time = perf_counter() - start
        start = perf_counter()
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        embeddings = self.embedder.encode_texts([chunk['text'] for chunk in chunks])
        self.retriever = CosineRetriever(chunks, embeddings)
        self.embedding_time = perf_counter() - start
        self.chunks = chunks
        return self

    def retrieve(self, question, top_k=5):
        if self.retriever is None:
            raise RuntimeError("Call index_documents first")
        return self.retriever.retrieve_vectors(
            [self.embedder.encode_query(question)], top_k)[0]

    def answer(self, question, top_k=5):
        if self.llm is None:
            raise RuntimeError("Configure a shared LLMClient before generating answers")
        retrieved = self.retrieve(question, top_k)
        prompt = build_rag_prompt(question, retrieved)
        return dict(answer=self.llm.generate(prompt), prompt=prompt, retrieved=retrieved)
