"""Fair corpus-level comparison; unannotated boundaries are never negatives."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def normalize(text):
    return " ".join(text.split())


def calculate_boundary_metrics(records):
    tp = fp = fn = ignored = labeled = 0
    for record in records:
        predicted = set(record["predicted_cuts"])
        positive = set(record["gold_boundaries"])
        negative = set(record["negative_boundaries"])
        tp += len(predicted & positive)
        fp += len(predicted & negative)
        fn += len(positive - predicted)
        ignored += len(predicted - positive - negative)
        labeled += len(positive | negative)
    precision = tp / (tp + fp) if tp + fp else (0.0 if fn else None)
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn,
                ignored_predictions=ignored, labeled_boundaries=labeled)


def calculate_dependency_violation_rate(records):
    annotated = [r for r in records if r["dependency_boundaries"] is not None]
    total = sum(len(set(r["dependency_boundaries"])) for r in annotated)
    violations = sum(len(set(r["predicted_cuts"]) & set(r["dependency_boundaries"]))
                     for r in annotated)
    return dict(violations=violations, total_dependencies=total,
                rate=violations / total if total else None,
                annotated_documents=len(annotated))


def retrieval_metrics(chunks, questions, model):
    if not questions:
        return {"recall_at_3": None, "recall_at_5": None, "questions": 0, "details": []}
    from evaluation.rag_retrieval import DenseRetriever, build_rag_prompt

    rankings = DenseRetriever(chunks, model).retrieve([q["question"] for q in questions])
    details = []
    for question, ranked in zip(questions, rankings):
        def contains_evidence(chunk):
            return (chunk["document_id"] == question["document_id"]
                    and normalize(question["gold_evidence"]) in normalize(chunk["text"]))

        hits = {}
        for k in (3, 5):
            hits[f"hit_at_{k}"] = any(contains_evidence(c) for c in ranked[:k])
        evidence_preserved = any(contains_evidence(c) for c in chunks)
        details.append(dict(question=question, top_5=ranked,
                            evidence_preserved=evidence_preserved,
                            failure_reason=None if hits["hit_at_5"] else (
                                "ranking_miss" if evidence_preserved else "evidence_split"),
                            rag_prompt_at_5=build_rag_prompt(question["question"], ranked),
                            context_words_at_5=sum(len(c["text"].split()) for c in ranked),
                            **hits))
    return {"recall_at_3": sum(d["hit_at_3"] for d in details) / len(details),
            "recall_at_5": sum(d["hit_at_5"] for d in details) / len(details),
            "evidence_preservation_rate": sum(d["evidence_preserved"] for d in details) / len(details),
            "mean_context_words_at_5": sum(d["context_words_at_5"] for d in details) / len(details),
            "questions": len(details), "details": details}


def validate_documents(documents, dependency_annotations):
    from sentence_splitter import split_sentences
    ids = set()
    for doc in documents:
        if doc["id"] in ids or not doc["text"].strip():
            raise ValueError("Document IDs must be unique and text nonempty")
        ids.add(doc["id"])
        valid = set(range(len(split_sentences(doc["text"])) - 1))
        expected = doc.get("expected", {})
        complete = "gold_boundaries" in doc
        positive = set(doc.get("gold_boundaries", expected.get("must_cut_after", [])))
        negative = valid - positive if complete else set(expected.get("must_not_cut_after", []))
        dependencies = doc.get("dependency_boundaries", dependency_annotations.get(doc["id"]))
        positions = positive | negative | set(dependencies or [])
        if any(type(p) is not int for p in positions) or not positions <= valid or positive & negative:
            raise ValueError(f"Invalid boundary annotation: {doc['id']}")
        if positive & set(dependencies or []):
            raise ValueError(f"Gold cut conflicts with dependency: {doc['id']}")
        doc.update(gold_boundaries=sorted(positive), negative_boundaries=sorted(negative),
                   dependency_boundaries=dependencies, complete_annotation=complete)
    if not documents:
        raise ValueError("Corpus must not be empty")


def validate_questions(questions, documents):
    texts = {d["id"]: normalize(d["text"]) for d in documents}
    ids = [q["id"] for q in questions if "id" in q]
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique")
    for question in questions:
        if (not question["question"].strip() or not normalize(question["gold_evidence"])
                or question["document_id"] not in texts
                or normalize(question["gold_evidence"]) not in texts[question["document_id"]]):
            raise ValueError(f"Invalid question or evidence: {question}")


def compare(documents, questions):
    from baselines.llamaindex_semantic_chunker import get_parser, llamaindex_semantic_chunk
    from evaluation.evaluate_chunking import extract_cut_positions
    from semantic_chunker import semantic_chunk
    from semantic_similarity import model, _encode_sentence

    get_parser()  # initialization excluded from timing for both methods
    model.encode("Khởi động mô hình.", show_progress_bar=False)
    results = {}
    for name, chunker in [("LlamaIndex SemanticSplitter", llamaindex_semantic_chunk),
                          ("Ours", semantic_chunk)]:
        _encode_sentence.cache_clear()
        records, corpus = [], []
        elapsed = 0.0
        for doc in documents:
            start = perf_counter()
            chunks = chunker(doc["text"])
            elapsed += perf_counter() - start
            cuts = extract_cut_positions(doc["text"], chunks)
            if len(cuts) != len(chunks) - 1:
                raise ValueError(f"Cannot align chunks: {name}, {doc['id']}")
            # Existing helper can be ambiguous when sentence text repeats.
            from sentence_splitter import split_sentences
            sentences = split_sentences(doc["text"])
            cursor = 0
            for i, chunk in enumerate(chunks):
                end = cuts[i] + 1 if i < len(cuts) else len(sentences)
                chunk_sentences = split_sentences(chunk)
                core = sentences[cursor:end]
                if not core or chunk_sentences[-len(core):] != core:
                    raise ValueError(f"Ambiguous sentence alignment: {doc['id']}")
                cursor = end
            records.append(dict(id=doc["id"], predicted_cuts=cuts,
                                gold_boundaries=doc["gold_boundaries"],
                                negative_boundaries=doc["negative_boundaries"],
                                dependency_boundaries=doc["dependency_boundaries"]))
            corpus.extend(dict(document_id=doc["id"], chunk_index=i, text=text)
                          for i, text in enumerate(chunks))
        results[name] = dict(boundary=calculate_boundary_metrics(records),
                             dependency=calculate_dependency_violation_rate(records),
                             retrieval=retrieval_metrics(corpus, questions, model),
                             chunking_seconds=elapsed, chunk_count=len(corpus),
                             documents=records, chunks=corpus)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=ROOT / "data/test_case.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/benchmark/questions.json")
    parser.add_argument("--annotations", type=Path, help="Optional dependency annotation sidecar")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/results")
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    documents, questions = read(args.documents), read(args.questions)
    # Default annotations apply only to the original corpus, never to custom IDs.
    if args.annotations is None and args.documents.resolve() == (ROOT / "data/test_case.json").resolve():
        args.annotations = ROOT / "data/benchmark/annotations.json"
    annotations = read(args.annotations)["dependency_boundaries"] if args.annotations else {}
    validate_documents(documents, annotations)
    validate_questions(questions, documents)
    print("===== CHUNKER COMPARISON =====", flush=True)
    print("Boundary metrics: labeled positions only; unlabeled predictions are reported separately.")
    if not questions:
        print("Retrieval: N/A (no question/gold evidence annotations).", flush=True)
    results = compare(documents, questions)
    metadata = dict(model=MODEL_NAME, buffer_size=1, breakpoint_percentile_threshold=95,
                    retrieval_protocol="Corpus-wide cosine; top-3/5; full gold span in one chunk of labeled document",
                    generation="Not executed; identical RAG prompt template exported for each method",
                    python=platform.python_version(), documents=len(documents),
                    complete_boundary_documents=sum(d["complete_annotation"] for d in documents),
                    timing="Single warm-model run; initialization and retrieval excluded; sentence cache cleared per method",
                    packages={p: importlib.metadata.version(p) for p in
                              ("llama-index-core", "sentence-transformers", "torch", "numpy")},
                    input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (args.documents, args.questions, args.annotations) if p is not None})
    rows = []
    for name, result in results.items():
        rows.append(dict(method=name, boundary_precision=result["boundary"]["precision"],
                         boundary_recall=result["boundary"]["recall"], boundary_f1=result["boundary"]["f1"],
                         dependency_violation_rate=result["dependency"]["rate"],
                         recall_at_3=result["retrieval"]["recall_at_3"],
                         recall_at_5=result["retrieval"]["recall_at_5"],
                         chunking_seconds=result["chunking_seconds"]))
    print("\n===== COMPARISON =====")
    print(f"{'Metric':30} {'LlamaIndex':>14} {'Ours':>14}")
    for key in list(rows[0])[1:]:
        values = ["N/A" if r[key] is None else f"{r[key]:.4f}" for r in rows]
        print(f"{key:30} {values[0]:>14} {values[1]:>14}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "chunker_comparison.json").write_text(
        json.dumps(dict(metadata=metadata, results=results), ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "chunker_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "rag_questions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["method", "question_id", "document_id", "category", "question", "hit_at_3",
                  "hit_at_5", "evidence_preserved", "failure_reason", "context_words_at_5"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            for index, detail in enumerate(result["retrieval"]["details"]):
                q = detail["question"]
                writer.writerow(dict(method=name, question_id=q.get("id", str(index)),
                                     document_id=q["document_id"], category=q.get("category", ""),
                                     question=q["question"],
                                     **{key: detail[key] for key in fields[5:]}))


if __name__ == "__main__":
    main()
