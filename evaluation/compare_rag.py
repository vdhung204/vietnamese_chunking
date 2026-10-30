"""Controlled Max–Min versus Ours benchmark; run from any working directory."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from evaluation.evaluate_retrieval import evaluate_retrieval, validate_benchmark
from evaluation.rag_retrieval import build_rag_prompt
from rag.embedder import SharedEmbedder, MODEL_NAME
from rag.rag_pipeline import RAGPipeline


def compare_rag(documents, questions, embedder, chunkers, top_k=5, llm=None):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 5:
        raise ValueError('top_k must be at least 5 to compute Recall@5')
    validate_benchmark(documents, questions)
    start = perf_counter()
    query_vectors = embedder.encode_texts([q['question'] for q in questions])
    query_embedding_time = perf_counter() - start
    results = []
    for name, chunker in chunkers:
        pipeline = RAGPipeline(chunker, embedder, llm).index_documents(documents)
        start = perf_counter()
        rankings = pipeline.retriever.retrieve_vectors(query_vectors, top_k)
        retrieval_time = perf_counter() - start
        metrics = evaluate_retrieval(questions, rankings, pipeline.chunks)
        generation_time = 0.
        for detail in metrics['details']:
            # Neither evidence nor gold answers enter this prompt.
            detail['prompt'] = build_rag_prompt(detail['question'], detail['retrieved'])
            if llm is not None:
                start = perf_counter()
                detail['answer'] = llm.generate(detail['prompt'])
                generation_time += perf_counter() - start
        results.append(dict(
            method=name, num_documents=len(documents), num_questions=len(questions),
            num_chunks=len(pipeline.chunks),
            avg_chunk_words=sum(len(c['text'].split()) for c in pipeline.chunks) / len(pipeline.chunks),
            chunking_time=pipeline.chunking_time, embedding_time=pipeline.embedding_time,
            indexing_time=pipeline.chunking_time + pipeline.embedding_time,
            retrieval_time=retrieval_time, generation_time=generation_time,
            chunks=pipeline.chunks, **metrics))
    return dict(
        configuration=dict(embedding_model=MODEL_NAME, top_k=top_k, metric='cosine',
                           tie_break='stable corpus order',
                           evidence_match='NFC + casefold + whitespace normalization + substring, same document',
                           length_unit='whitespace-separated units, not model tokens',
                           llm=asdict(llm.config) if llm is not None else None,
                           prompt_template=build_rag_prompt('{question}', [])),
        query_embedding_time=query_embedding_time, results=results)


def write_results(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'rag_comparison.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = [key for key, value in report['results'][0].items() if not isinstance(value, list)]
    with (output_dir / 'rag_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['results'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', type=Path, default=ROOT / 'data/test_case.json')
    parser.add_argument('--questions', type=Path, default=ROOT / 'data/benchmark/questions.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'evaluation/results')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--ollama', action='store_true', help='Generate answers with local Ollama')
    parser.add_argument('--ollama-url', default='http://localhost:11434')
    args = parser.parse_args()
    if args.top_k < 5:
        parser.error('--top-k must be >= 5 for Recall@5')
    documents = json.loads(args.documents.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    validate_benchmark(documents, questions)

    from baselines.maxmin_semantic_chunker import maxmin_semantic_chunk
    from semantic_chunker import semantic_chunk
    from semantic_similarity import _encode_sentence

    embedder = SharedEmbedder()
    embedder.encode_texts(['Khởi động mô hình.'])
    _encode_sentence.cache_clear()
    llm = None
    if args.ollama:
        from rag.llm_client import OllamaClient
        llm = OllamaClient(base_url=args.ollama_url)
    report = compare_rag(documents, questions, embedder,
                         [('Max-Min', maxmin_semantic_chunk), ('Ours', semantic_chunk)], args.top_k, llm)
    if llm is not None:
        report['configuration']['ollama_url'] = llm.base_url
    report['provenance'] = dict(
        python=platform.python_version(),
        packages={name: importlib.metadata.version(name)
                  for name in ('numpy', 'sentence-transformers', 'torch', 'scikit-learn')},
        inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (args.documents, args.questions)},
        chunking_sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [*sorted((ROOT / 'src').glob('*.py')),
                                       ROOT / 'baselines/maxmin_semantic_chunker.py']},
        timing='Single warm-model run, initialization excluded; query encoding shared once; '
               'retrieval_time is ranking only; embedding_time includes index normalization; '
               'method order Max-Min then Ours; not a statistical speed comparison.')
    write_results(report, args.output_dir)
    print('===== RAG COMPARISON =====')
    for result in report['results']:
        print(f"\nMethod: {result['method']}")
        for field in ('num_documents', 'num_questions', 'num_chunks', 'recall_at_3',
                      'recall_at_5', 'mrr_at_5', 'chunking_time', 'embedding_time',
                      'indexing_time', 'retrieval_time'):
            print(f'{field}: {result[field]}')
    print(f'\nResults: {args.output_dir}')
    print('Answers saved; QA Accuracy not scored.' if llm else
          'LLM not configured: retrieval metrics and prompts only; no QA Accuracy.')


if __name__ == '__main__':
    main()
