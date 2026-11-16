"""Recompute study rankings and published numeric evidence; never retune on test.

Run after evaluation/parameter_sensitivity.py. Writes only parameter_study/.
"""

import contextlib
import csv
import hashlib
import json
import gzip
import os
from functools import lru_cache
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'evaluation/results/parameter_study'


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def csv_read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(left, right):
    assert abs(float(left) - float(right)) < 1e-10, (left, right)


def main():
    os.environ['OPENBLAS_NUM_THREADS']='4'
    os.environ['OMP_NUM_THREADS']='4'
    import numpy as np
    from evaluation.parameter_sensitivity import frozen_hashes, write_json, write_csv
    from evaluation.evaluate_retrieval import evaluate_retrieval
    from evaluation.summarize_rag import summarize
    from rag.retriever import CosineRetriever
    from scripts.build_xquad_benchmark import SOURCE_HASH, convert, describe
    import evaluation.evaluate_retrieval as retrieval_evaluation
    retrieval_evaluation.normalize=lru_cache(maxsize=None)(retrieval_evaluation.normalize)

    summary = read(OUT/'summary.json')
    protocol = read(OUT/'protocol.json')
    details = read(OUT/'details.json.gz')
    corpora = read(OUT/'corpora.json.gz')
    vectors = np.load(OUT/'embeddings.npz')
    rows = csv_read(OUT/'sensitivity.csv')
    aggregates = csv_read(OUT/'aggregate.csv')
    before = frozen_hashes()
    study_sources = summary['provenance']['source_sha256_before']
    changed_since_study = sorted(name for name in set(before) | set(study_sources)
                                 if before.get(name) != study_sources.get(name))
    text_index = {text: i for i, text in enumerate(corpora['texts'])}
    question_sets = {view: read(ROOT/f'data/benchmark/xquad_vi/{view}/dev/questions.json')
                     for view in ('paragraph','article')}
    measured = {}
    for index, (key, signature) in enumerate(corpora['mapping'].items(),1):
        config_id, view = key.split('/')
        if (view, signature) not in measured:
            chunks = corpora['corpora'][signature]
            chunk_vectors = np.array([vectors['chunk_vectors'][text_index[c['text']]] for c in chunks])
            rankings = CosineRetriever(chunks, chunk_vectors).retrieve_vectors(vectors['query_vectors'], 5)
            measured[(view, signature)] = evaluate_retrieval(question_sets[view], rankings, chunks)
        metrics = measured[(view, signature)]
        row = next(r for r in rows if r['config_id']==config_id and r['view']==view)
        for field in ('recall_at_3','recall_at_5','mrr_at_5','evidence_preservation_rate','mean_context_words'):
            close(metrics[field], row[field])
        saved = details[key]['questions']
        for actual, original in zip(metrics['details'], saved, strict=True):
            for field in ('question_id','hit_at_3','hit_at_5','reciprocal_rank_at_5','failure_reason','context_words'):
                assert actual[field] == original[field]
            assert [c['chunk_id'] for c in actual['retrieved']] == original['retrieved_ids']
        if index % 60 == 0:
            print(f'Checked {index}/{len(rows)} configuration/view results.',flush=True)
    for row in aggregates:
        pair = [r for r in rows if r['config_id'] == row['config_id']]
        close(row['mean_recall_at_5'], sum(float(r['recall_at_5']) for r in pair)/2)
        close(row['mean_mrr_at_5'], sum(float(r['mrr_at_5']) for r in pair)/2)
    print(f'Verified all {len(rows)} dev configuration/view results from saved real vectors.', flush=True)

    # Rebuild expected dataset objects in memory; never overwrite annotations.
    data_root = ROOT/'data/benchmark/xquad_vi'
    raw = (data_root/'source/xquad.vi.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_HASH
    paragraphs, articles, questions, long_questions, duplicates = convert(json.loads(raw))
    manifest = read(data_root/'manifest.json')
    dataset_counts = {}
    for view, docs, qs in (('paragraph',paragraphs,questions),('article',articles,long_questions)):
        for split in ('full','dev','test'):
            selected_docs = [d for d in docs if split=='full' or d['split']==split]
            selected_qs = [q for q in qs if split=='full' or q['split']==split]
            assert selected_docs == read(data_root/view/split/'documents.json')
            assert selected_qs == read(data_root/view/split/'questions.json')
            assert describe(selected_docs,selected_qs) == manifest['summaries'][f'{view}/{split}']
            dataset_counts[f'{view}/{split}'] = dict(documents=len(selected_docs),questions=len(selected_qs),
                titles=len({d['article_id'] for d in selected_docs}))
    assert duplicates == read(data_root/'duplicate_question_audit.json')
    for path, expected in manifest['artifact_sha256'].items():
        assert hashlib.sha256((data_root/path).read_bytes()).hexdigest() == expected

    historical = {}
    bootstrap_rows = []
    for view in ('paragraph','article'):
        base=ROOT/f'evaluation/results/xquad_vi_{view}_test'
        report=read(base/'rag_comparison.json')
        qs=read(data_root/view/'test/questions.json')
        recalculated=summarize(report,qs)
        assert recalculated == read(base/'rag_statistical_summary.json')
        historical[view] = recalculated
        for repetitions in (1000,5000,10000):
            result = summarize(report,qs,repetitions=repetitions,seed=2026)
            for metric, values in result['metrics'].items():
                bootstrap_rows.append(dict(view=view,repetitions=repetitions,seed=2026,metric=metric,
                    delta=values['delta'],ci95_low=values['delta_ci95'][0],ci95_high=values['delta_ci95'][1]))
    write_csv(OUT/'bootstrap_stability.csv',bootstrap_rows)
    print('Verified dataset counts/splits/checksums and historical 5,000-repeat bootstrap.', flush=True)

    # Run the existing read-only numerical analysis into a NEW folder. This
    # recomputes tokens, 512/1024 budgets, source metrics and dependency counts.
    from scripts import audit_report_evidence
    numeric_out = OUT/'numeric_audit'
    numeric_out.mkdir(parents=True,exist_ok=True)
    audit_report_evidence.OUT=numeric_out
    with (numeric_out/'run.log').open('w',encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        audit_report_evidence.main()
    fresh=read(numeric_out/'audit.json')
    old=read(ROOT/'evaluation/results/report_audit/audit.json')
    for field in ('original_metrics_verified','tokenizer_stats','budget_results','dependency_diagnostic'):
        assert fresh[field] == old[field], field
    print('Recomputed historical retrieval, tokenizer, budget and dependency numbers: exact match.', flush=True)

    # List EVERY numeric mention in the Markdown report for manual traceability.
    # This inventory is not a claim that a regex proves the surrounding sentence.
    mentions=[]
    heading=''
    for line_number,line in enumerate((ROOT/'BAO_CAO.md').read_text(encoding='utf-8').splitlines(),1):
        if line.startswith('#'):
            heading=line.lstrip('# ')
        if line.startswith('#') or 'http' in line:
            continue
        for match in re.finditer(r'(?<![\w])\d+(?:[.,]\d+)*(?:%)?',line):
            mentions.append(dict(line=line_number,section=heading,value=match.group(),
                                 context=line[max(0,match.start()-65):match.end()+100]))
    write_csv(OUT/'report_numeric_mentions.csv',mentions)

    # Paired source-title bootstrap of NEW dev differences. Selection on the
    # same dev makes these exploratory intervals, not confirmatory p-values.
    labels=question_sets['paragraph']
    ids=[q['id'] for q in labels]
    titles=sorted({q['article_id'] for q in labels})
    groups=[np.array([i for i,q in enumerate(labels) if q['article_id']==title]) for title in titles]
    samples=np.random.default_rng(2026).integers(0,len(groups),size=(5000,len(groups)))
    denominator=np.array([len(g) for g in groups])[samples].sum(axis=1)
    default_id=summary['default_config_id']
    comparisons={}
    candidates={'best_tested':summary['best']['config_id']}
    candidates.update({family:value['best_config_id'] for family,value in summary['by_family'].items()})
    for name,config_id in candidates.items():
        data={}
        for field in ('hit_at_5','reciprocal_rank_at_5'):
            delta=np.zeros(len(labels))
            for view in ('paragraph','article'):
                original={r['question_id']:r for r in details[f'{default_id}/{view}']['questions']}
                candidate={r['question_id']:r for r in details[f'{config_id}/{view}']['questions']}
                delta+=np.array([float(candidate[qid][field])-float(original[qid][field]) for qid in ids])/2
            sums=np.array([delta[g].sum() for g in groups])
            values=sums[samples].sum(axis=1)/denominator
            data[field]=dict(delta=float(delta.mean()),ci95=[float(v) for v in np.quantile(values,[.025,.975])])
        comparisons[name]=dict(config_id=config_id,**data)
    write_json(OUT/'dev_exploratory_intervals.json',dict(
        warning='Selected comparisons on the same dev; no multiple-search correction or independent confirmation. '
                'Resample 10 source titles jointly across both views; never count views as independent questions.',
        comparisons=comparisons))

    family_ties={}
    for family, item in summary['by_family'].items():
        eligible=[r for r in aggregates if family in r['families'].split(';') or r['config_id']==default_id]
        family_ties[family]=[r['config_id'] for r in eligible if
            abs(float(r['mean_recall_at_5'])-item['best_mean_recall_at_5'])<1e-12 and
            abs(float(r['mean_mrr_at_5'])-item['best_mean_mrr_at_5'])<1e-12]
    assert frozen_hashes() == before
    verification=dict(study_configurations=len(protocol['grid']),study_view_results_verified=len(rows),
        distinct_rankings_recomputed=len(measured),dataset_counts=dataset_counts,
        source_questions=len(questions),unique_normalized_questions=manifest['unique_normalized_questions'],
        duplicate_question_groups=len(duplicates),report_numeric_mentions=len(mentions),
        original_bootstrap_exact_match=True,historical_numeric_audit_exact_match=True,
        production_sources_unchanged_during_verification=True,
        current_sources_match_study_snapshot=not changed_since_study,
        sources_changed_since_study=changed_since_study,
        family_tied_best_config_ids=family_ties,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(OUT/'verification.json',verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
