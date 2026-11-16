"""Check the active CLI/GUI chunker against the dev-selected weight artifact."""

import csv
from dataclasses import asdict
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'evaluation/results/parameter_study'


def main():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['OPENBLAS_NUM_THREADS']='4'
    print('Loading cached model to verify active weights...',flush=True)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    import semantic_similarity
    semantic_similarity.model.to('cpu')
    semantic_similarity._encode_sentence=lru_cache(maxsize=None)(semantic_similarity._encode_sentence.__wrapped__)
    from semantic_chunker import semantic_chunk
    from evaluation.parameter_sensitivity import production_settings, frozen_hashes
    from evaluation.evaluate_retrieval import evaluate_retrieval
    import evaluation.evaluate_retrieval as scoring
    scoring.normalize=lru_cache(maxsize=None)(scoring.normalize)
    from evaluation.evaluate_chunking import evaluate_case
    from sentence_splitter import split_sentences
    from dynamic_overlap import determine_overlap_count
    from gui.chunkers import configured_ours, OursSettings
    from rag.retriever import CosineRetriever

    config=production_settings()
    with (OUT/'aggregate.csv').open(encoding='utf-8-sig',newline='') as stream:
        aggregate=list(csv.DictReader(stream))
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    selected=[key for key,value in protocol['grid'].items() if value['parameters']==asdict(config)]
    assert len(selected)==1, 'Active production parameters must match one tested configuration'
    config_id=selected[0]
    row=next(r for r in aggregate if r['config_id']==config_id)
    best_recall=max(float(r['mean_recall_at_5']) for r in aggregate if 'weights' in r['families'].split(';'))
    assert abs(float(row['mean_recall_at_5'])-best_recall)<1e-12
    with gzip.open(OUT/'corpora.json.gz','rt',encoding='utf-8') as stream:
        saved=json.load(stream)
    embeddings=np.load(OUT/'embeddings.npz')
    lookup={text:i for i,text in enumerate(saved['texts'])}
    gui_chunker=configured_ours(OursSettings())
    results={}
    before=frozen_hashes()
    for view in ('paragraph','article'):
        base=ROOT/f'data/benchmark/xquad_vi/{view}/dev'
        docs=json.loads((base/'documents.json').read_text(encoding='utf-8'))
        questions=json.loads((base/'questions.json').read_text(encoding='utf-8'))
        expected=saved['corpora'][saved['mapping'][f'{config_id}/{view}']]
        chunks=[]
        for doc in docs:
            output=semantic_chunk(doc['text'])
            assert output==gui_chunker(doc['text']), 'GUI and CLI disagree'
            chunks.extend(dict(document_id=doc['id'],chunk_index=i,chunk_id=f"{doc['id']}:{i}",text=text)
                          for i,text in enumerate(output))
        assert chunks==expected, f'Production output differs from selected experiment: {view}'
        vectors=np.array([embeddings['chunk_vectors'][lookup[c['text']]] for c in chunks])
        ranked=CosineRetriever(chunks,vectors).retrieve_vectors(embeddings['query_vectors'],5)
        metrics=evaluate_retrieval(questions,ranked,chunks)
        results[view]={key:value for key,value in metrics.items() if key!='details'}
        results[view]['documents']=len(docs)
        results[view]['chunks']=len(chunks)
        print(view,results[view],flush=True)
    for field in ('recall_at_5','mrr_at_5'):
        mean=sum(r[field] for r in results.values())/2
        assert abs(mean-float(row['mean_'+field]))<1e-10
    pilot=json.loads((ROOT/'data/test_case.json').read_text(encoding='utf-8'))
    regressions=[evaluate_case(case) for case in pilot if int(case['id'].split('_')[1])>=4]
    failures=[r['id'] for r in regressions if not r['passed']]
    overlap_checks=[]
    originals={case['id']:case for case in pilot}
    for result in regressions:
        sentences=split_sentences(originals[result['id']]['text'])
        previous=-1
        expected=[]
        for cut in result['predicted_cuts']:
            expected.append(min(cut-previous,determine_overlap_count(sentences[cut],sentences[cut+1])))
            previous=cut
        assert expected==result['predicted_overlaps'], 'Output overlap violates the unchanged cosine thresholds'
        overlap_checks.append(dict(id=result['id'],overlap_matches_current_boundaries=True))
    # The old fixture expects two sentences at the OLD selected boundary.
    # Keep that historical label intact; explicitly record the one known change.
    unexpected=[r['id'] for r in regressions if not r['passed'] and not (
        r['id']=='case_010' and r['expected_overlap']==2 and r['predicted_overlaps']==[1]
        and r['must_cut_pass'] and r['must_not_cut_pass'] and r['min_cuts_pass'])]
    assert frozen_hashes()==before
    summary=dict(active_parameters=asdict(config),selected_config_id=config_id,
        selection='Highest mean dev Recall@5 in the 54-weight grid; risk=0.6 retained among tied penalties. '
                  'Other parameters unchanged. Not the best full configuration in the 180-config study.',
        mean_recall_at_5=float(row['mean_recall_at_5']),mean_mrr_at_5=float(row['mean_mrr_at_5']),
        cli_gui_match_on_60_dev_documents=True,production_matches_saved_selected_corpora=True,
        results=results,regression_cases=len(regressions),regression_failures=failures,
        unexpected_regression_failures=unexpected,overlap_rule_checks=overlap_checks,
        known_snapshot_change='case_010: new weights select a different boundary; overlap becomes 1 instead of historical 2. Original fixture retained.',
        regression_details=regressions,source_sha256=before,
        historical_test_configuration='Original 0.4 semantic / 0.4 size / 0.6 risk; old tables were not rerun.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'active_configuration.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Regression:',len(regressions),'cases; failures:',failures,flush=True)
    if unexpected:
        raise SystemExit('Unexpected regression: inspect active_configuration.json')


if __name__=='__main__':
    main()
