"""Evaluate frozen production weights for the Vietnamese scientific manuscript."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/results/scientific_paper_current'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['OPENBLAS_NUM_THREADS']='4'
    print('Loading cached model for current-weight paper experiments...',flush=True)
    import torch
    import numpy as np
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    np.random.seed(2026)
    from evaluation import compare_rag
    from evaluation.summarize_rag import summarize
    from evaluation.evaluate_retrieval import evaluate_retrieval,contains_evidence
    import evaluation.evaluate_retrieval as retrieval_scoring
    retrieval_scoring.normalize=lru_cache(maxsize=None)(retrieval_scoring.normalize)
    from evaluation.parameter_sensitivity import production_settings,write_json,write_csv,frozen_hashes
    from semantic_similarity import model
    model.to('cpu')
    model.eval()
    parameters=asdict(production_settings())
    assert (parameters['semantic_weight'],parameters['size_weight'],parameters['risk_weight'])==(.5,.3,.6)
    before=frozen_hashes()
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',dict(
        created_utc=datetime.now(timezone.utc).isoformat(),parameters=parameters,split='test',
        model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',max_seq_length=model.max_seq_length,
        top_k=5,device='cpu',threads=4,seed=2026,
        selection='Frozen after dev weight grid; no parameter updates during test runs.',
        limitation='Previous test results were already examined before the new study. '
                   'This is a new evaluation of the selected configuration on a previously inspected test set, '
                   'not independent confirmation on unseen data.',
        baseline='Original Max-Min defaults 0.6/0.9/1.5; shared model, splitter, retrieval and labels.',
        source_sha256=before,llm=None))
    token_rows,budget_rows,metric_rows=[],[],[]

    @lru_cache(maxsize=None)
    def token_length(text,special=False):
        return len(model.tokenizer(text,add_special_tokens=special,truncation=False)['input_ids'])

    for view in ('paragraph','article'):
        destination=OUT/f'xquad_vi_{view}_test'
        documents=ROOT/f'data/benchmark/xquad_vi/{view}/test/documents.json'
        questions_path=documents.with_name('questions.json')
        print(f'Running {view}: current Ours versus Max-Min...',flush=True)
        sys.argv=['compare_rag.py','--documents',str(documents),'--questions',str(questions_path),
                  '--output-dir',str(destination)]
        compare_rag.main()
        report=read(destination/'rag_comparison.json')
        report['configuration']['ours_parameters']=parameters
        report['configuration']['device']='cpu'
        report['configuration']['threads']=4
        report['configuration']['max_seq_length']=model.max_seq_length
        report['provenance']['evaluated_utc']=datetime.now(timezone.utc).isoformat()
        compare_rag.write_results(report,destination)
        questions=read(questions_path)
        summary=summarize(report,questions)
        write_json(destination/'rag_statistical_summary.json',summary)
        for result in report['results']:
            method=result['method']
            fresh=evaluate_retrieval(questions,[r['retrieved'] for r in result['details']],result['chunks'])
            for field in ('recall_at_3','recall_at_5','mrr_at_5','evidence_preservation_rate','mean_context_words'):
                assert abs(fresh[field]-result[field])<1e-10
            lens=[token_length(c['text'],True) for c in result['chunks']]
            token_rows.append(dict(view=view,method=method,chunks=len(lens),mean=statistics.mean(lens),
                median=statistics.median(lens),maximum=max(lens),over_128=sum(n>128 for n in lens),
                fraction_over_128=sum(n>128 for n in lens)/len(lens)))
            metric_rows.append(dict(view=view,method=method,num_documents=result['num_documents'],
                num_questions=result['num_questions'],num_chunks=result['num_chunks'],avg_chunk_words=result['avg_chunk_words'],
                **{field:result[field] for field in ('recall_at_3','recall_at_5','mrr_at_5','evidence_preservation_rate','mean_context_words')},
                **dict(Counter(r['failure_reason'] or 'hit' for r in result['details']))))
            for budget in (512,1024):
                hits,rr,lengths,counts=0,[],[],[]
                for question,detail in zip(questions,result['details'],strict=True):
                    chosen=[]
                    for c in detail['retrieved']:
                        if token_length('\n\n'.join(x['text'] for x in chosen+[c]))>budget:
                            break
                        chosen.append(c)
                    first=next((i for i,c in enumerate(chosen,1) if contains_evidence(c,question)),None)
                    hits+=first is not None
                    rr.append(1/first if first else 0)
                    lengths.append(token_length('\n\n'.join(c['text'] for c in chosen)) if chosen else 0)
                    counts.append(len(chosen))
                budget_rows.append(dict(view=view,method=method,budget=budget,questions=len(questions),hits=hits,
                    evidence_recall=hits/len(questions),mrr=statistics.mean(rr),mean_context_tokens=statistics.mean(lengths),
                    mean_selected_chunks=statistics.mean(counts),empty_contexts=sum(n==0 for n in counts)))
        print(f'Verified {view} retrieval; paired bootstrap and token diagnostics saved.',flush=True)
    assert frozen_hashes()==before
    write_json(OUT/'diagnostics.json',dict(tokenizer_stats=token_rows,budget_results=budget_rows,
        original_metrics_verified=metric_rows,production_sources_unchanged_during_run=True,
        source_sha256=before,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        budget_policy='Whole-chunk prefix of saved top-5; stop at first overflow; no skip/backfill/truncation. '
                      'Tokenize context with MiniLM tokenizer, no special/query/prompt tokens; not full-ranking fixed-budget.',
        tokenizer_policy='Actual model tokenizer; chunk lengths include special tokens, without truncation. '
                         'Overlength does not identify evidence lost to truncation.'))
    write_csv(OUT/'tokenizer_summary.csv',token_rows)
    write_csv(OUT/'budget_summary.csv',budget_rows)
    fields=list(dict.fromkeys(k for r in metric_rows for k in r))
    write_csv(OUT/'metrics_verified.csv',[{key:r.get(key,0) for key in fields} for r in metric_rows])
    print('Paper experiments complete; production weights remained frozen.',flush=True)


if __name__=='__main__':
    main()
