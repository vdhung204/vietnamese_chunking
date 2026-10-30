"""Paired retrieval summary with source-article cluster bootstrap intervals.

Reads existing results only. Does not run or tune chunkers, retrieval or LLMs.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

METRICS = {'recall_at_3': 'hit_at_3', 'recall_at_5': 'hit_at_5',
           'mrr_at_5': 'reciprocal_rank_at_5'}


def summarize(report, questions, repetitions=5000, seed=2026):
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    labels = {q['id']: q for q in questions}
    methods = {result['method']: result for result in report['results']}
    if len(methods) != 2:
        raise ValueError('Exactly two methods required for paired differences')
    names = list(methods)
    records = [{d['question_id']: d for d in methods[name]['details']} for name in names]
    if any(set(record) != set(labels) for record in records):
        raise ValueError('Question IDs must match both reports exactly')
    ids = sorted(labels)
    titles = sorted({labels[qid]['article_id'] for qid in ids})
    groups = [np.array([i for i, qid in enumerate(ids) if labels[qid]['article_id'] == title])
              for title in titles]
    counts = np.array([len(g) for g in groups])
    # Resample source titles jointly for both methods, keeping all questions of
    # each sampled title together. Questions within a title are not independent.
    sampled = np.random.default_rng(seed).integers(0, len(groups), size=(repetitions, len(groups)))
    denominators = counts[sampled].sum(axis=1)
    output = dict(method_order=names, difference=f'{names[1]} minus {names[0]}',
                  questions=len(ids), source_titles=len(titles), bootstrap_repetitions=repetitions,
                  bootstrap_seed=seed, interval='95% paired source-title cluster bootstrap percentile',
                  note='Exploratory retrieval intervals, not QA accuracy or proof of generalization.',
                  metrics={}, by_domain={}, failure_counts={})
    for metric, field in METRICS.items():
        values = [np.array([float(record[qid][field]) for qid in ids]) for record in records]
        delta = values[1] - values[0]
        sums = np.array([delta[group].sum() for group in groups])
        samples = sums[sampled].sum(axis=1) / denominators
        output['metrics'][metric] = dict(
            means={name: float(value.mean()) for name, value in zip(names, values)},
            delta=float(delta.mean()),
            delta_ci95=[float(x) for x in np.quantile(samples, [.025, .975])],
            source_title_macro={name: float(np.mean([value[group].mean() for group in groups]))
                                for name, value in zip(names, values)})
        if metric == 'recall_at_5':
            output['paired_hit_counts_at_5'] = dict(
                both_hit=int(np.sum((values[0] == 1) & (values[1] == 1))),
                only_first=int(np.sum((values[0] == 1) & (values[1] == 0))),
                only_second=int(np.sum((values[0] == 0) & (values[1] == 1))),
                neither=int(np.sum((values[0] == 0) & (values[1] == 0))))
    for domain in sorted({q['domain'] for q in questions}):
        selected = [qid for qid in ids if labels[qid]['domain'] == domain]
        output['by_domain'][domain] = dict(questions=len(selected), methods={
            name: {metric: float(np.mean([record[qid][field] for qid in selected]))
                   for metric, field in METRICS.items()} for name, record in zip(names, records)})
    for name, record in zip(names, records):
        output['failure_counts'][name] = dict(Counter(d['failure_reason'] or 'hit' for d in record.values()))
    # Sensitivity check: each normalized question gets one vote. Keep the first
    # ID lexicographically, determined independently of either method's score.
    unique_ids, seen = [], set()
    for qid in ids:
        key = ' '.join(labels[qid]['question'].casefold().split())
        if key not in seen:
            seen.add(key)
            unique_ids.append(qid)
    output['deduplicated_sensitivity'] = dict(
        questions=len(unique_ids), policy='first lexicographic question ID per casefold/whitespace-normalized question',
        methods={name: {metric: float(np.mean([record[qid][field] for qid in unique_ids]))
                        for metric, field in METRICS.items()} for name, record in zip(names, records)})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(json.loads(args.report.read_text(encoding='utf-8')),
                        json.loads(args.questions.read_text(encoding='utf-8')))
    output = args.report.with_name('rag_statistical_summary.json')
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary['metrics'], ensure_ascii=False, indent=2))
    print(f'Saved: {output}')
