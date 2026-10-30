"""Read-only post-hoc diagnostics on saved rankings; never reruns chunking.

Run: .\venv\Scripts\python.exe -X utf8 scripts/audit_report_evidence.py
Outputs: evaluation/results/report_audit/*
Budget policy: preserve saved top-5 order, append whole chunks while the
tokenization of '\n\n'.join(context) fits 512/1024 tokens; stop at first
overflow. No clipping/skipping/backfilling; not unrestricted budget retrieval.
"""
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.evaluate_retrieval import contains_evidence, evaluate_retrieval

OUT = ROOT / 'evaluation/results/report_audit'
MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
SNAPSHOT = Path.home()/'.cache/huggingface/hub'/('models--'+MODEL.replace('/','--'))/'snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in
            [*sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'baselines').glob('*.py'))]}

def csv_out(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    before=source_hashes()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(SNAPSHOT),local_files_only=True)
    max_seq=read(SNAPSHOT/'sentence_bert_config.json')['max_seq_length']
    assert max_seq==128
    def length(text, special=True):
        return len(tokenizer(text,add_special_tokens=special,truncation=False)['input_ids'])
    OUT.mkdir(parents=True,exist_ok=True)
    inputs={}; stats=[]; lengths=[]; budgets=[]; details=[]; checks=[]
    for view in ['paragraph','article']:
        base=ROOT/f'evaluation/results/xquad_vi_{view}_test'
        report_path=base/'rag_comparison.json'
        question_path=ROOT/f'data/benchmark/xquad_vi/{view}/test/questions.json'
        doc_path=question_path.with_name('documents.json')
        for p in [report_path,question_path,doc_path,base/'rag_comparison.csv']:
            inputs[str(p.relative_to(ROOT))]=sha(p)
        report=read(report_path); questions=read(question_path); docs=read(doc_path)
        csv_rows=list(csv.DictReader((base/'rag_comparison.csv').open(encoding='utf-8')))
        for result,csv_row in zip(report['results'],csv_rows):
            method=result['method']; chunks=result['chunks']; records=result['details']
            assert len(records)==len(questions)==957
            assert [d['question_id'] for d in records]==[q['id'] for q in questions]
            measured=evaluate_retrieval(questions,[d['retrieved'] for d in records],chunks)
            for key in ['recall_at_3','recall_at_5','mrr_at_5','evidence_preservation_rate','mean_context_words']:
                assert abs(measured[key]-result[key])<1e-10
                assert abs(measured[key]-float(csv_row[key]))<1e-10
            assert len(chunks)==result['num_chunks']==int(csv_row['num_chunks'])
            assert len(docs)==result['num_documents']==int(csv_row['num_documents'])
            assert len(questions)==result['num_questions']==int(csv_row['num_questions'])
            mean_words=statistics.mean(len(c['text'].split()) for c in chunks)
            assert abs(mean_words-result['avg_chunk_words'])<1e-10
            checks.append(dict(view=view,method=method,documents=len(docs),questions=len(questions),chunks=len(chunks),
                               **{k:measured[k] for k in ['recall_at_3','recall_at_5','mrr_at_5','evidence_preservation_rate','mean_context_words']},
                               avg_chunk_words=mean_words))
            lens=[]
            for c in chunks:
                n=length(c['text']); lens.append(n)
                lengths.append(dict(view=view,method=method,document_id=c['document_id'],chunk_index=c['chunk_index'],tokens_with_special=n,over_128=n>128))
            stats.append(dict(view=view,method=method,chunks=len(lens),mean=statistics.mean(lens),median=statistics.median(lens),max=max(lens),
                              over_128=sum(n>128 for n in lens),fraction_over_128=sum(n>128 for n in lens)/len(lens)))
            for budget in [512,1024]:
                local=[]
                for q,record in zip(questions,records):
                    selected=[]
                    for c in record['retrieved']:
                        if length('\n\n'.join(x['text'] for x in selected+[c]),False)>budget:
                            break
                        selected.append(c)
                    rank=next((i for i,c in enumerate(selected,1) if contains_evidence(c,q)),None)
                    item=dict(view=view,method=method,budget=budget,question_id=q['id'],selected_chunks=len(selected),
                              context_tokens=length('\n\n'.join(c['text'] for c in selected),False) if selected else 0,
                              hit=rank is not None,reciprocal_rank=1/rank if rank else 0)
                    assert item['context_tokens']<=budget
                    assert not item['hit'] or record['hit_at_5']
                    local.append(item)
                details.extend(local)
                budgets.append(dict(view=view,method=method,budget=budget,questions=len(local),hits=sum(x['hit'] for x in local),
                                    evidence_recall=sum(x['hit'] for x in local)/len(local),mrr=sum(x['reciprocal_rank'] for x in local)/len(local),
                                    mean_context_tokens=statistics.mean(x['context_tokens'] for x in local),
                                    mean_selected_chunks=statistics.mean(x['selected_chunks'] for x in local),empty_contexts=sum(x['selected_chunks']==0 for x in local)))
    # Boundary diagnostic uses already saved pilot chunks, not new labels or runs.
    pilot_path=ROOT/'evaluation/results/rag_comparison.json'; ann_path=ROOT/'data/benchmark/annotations.json'; docs_path=ROOT/'data/test_case.json'
    for p in [pilot_path,ann_path,docs_path]: inputs[str(p.relative_to(ROOT))]=sha(p)
    pilot=read(pilot_path); annotations=read(ann_path)['dependency_boundaries']; pilot_docs={d['id']:d for d in read(docs_path)}
    diagnostic=[]
    for method in pilot['results']:
        rows=[]
        for doc_id,boundaries in annotations.items():
            original=re.split(r'(?<=[.?!])\s+',pilot_docs[doc_id]['text'].strip())
            chunks=[c['text'] for c in method['chunks'] if c['document_id']==doc_id]
            assert chunks
            cuts=[]; start=0
            for text in chunks[:-1]:
                last=re.split(r'(?<=[.?!])\s+',text.strip())[-1]
                found=next(i for i in range(start,len(original)) if original[i]==last)
                cuts.append(found); start=found+1
            rows.append(dict(document_id=doc_id,annotated_boundaries=boundaries,predicted_cuts=cuts,violations=sorted(set(cuts)&set(boundaries))))
        total=sum(len(r['annotated_boundaries']) for r in rows); violations=sum(len(r['violations']) for r in rows)
        diagnostic.append(dict(method=method['method'],annotated_documents=len(rows),annotated_dependency_boundaries=total,violations=violations,
                               rate=violations/total if total else None,details=rows))
    after=source_hashes(); assert before==after
    config_files=['tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentence_bert_config.json']
    output=dict(provenance=dict(command=r'.\venv\Scripts\python.exe -X utf8 scripts/audit_report_evidence.py',
        script_sha256=sha(Path(__file__)),python=platform.python_version(),packages={x:importlib.metadata.version(x) for x in ['transformers','tokenizers']},
        tokenizer_snapshot=str(SNAPSHOT),tokenizer_sha256={x:sha(SNAPSHOT/x) for x in config_files},max_seq_length=max_seq,
        inputs_sha256=inputs,source_sha256_before=before,source_sha256_after=after,chunking_unchanged=before==after,seed=None,
        randomization='None; deterministic post-hoc analysis; no parameter selection, chunking rerun or generation.',
        budget_policy='512/1024 content tokens without special tokens, query or prompt. Tokenize concatenated chunks with two newlines. Whole-chunk prefix of saved top-5, stop at first overflow, no skip/truncation/backfill. Not full-ranking budget retrieval.',
        tokenizer_policy='Chunk lengths include special tokens, without truncation. Snapshot found locally; original retrieval artifact does not pin tokenizer revision.',
        dependency_policy='3 annotated boundaries in 4 pilot documents including one negative-control document; cut positions before overlap interpretation; not a generalization benchmark.',
        evidence_truncation='Not measured: no verified source-span to tokenizer-offset mapping for all evidence occurrences.'),
        original_metrics_verified=checks,tokenizer_stats=stats,budget_results=budgets,dependency_diagnostic=diagnostic)
    (OUT/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    csv_out('tokenizer_summary.csv',stats); csv_out('chunk_token_lengths.csv',lengths); csv_out('budget_summary.csv',budgets); csv_out('budget_questions.csv',details)
    csv_out('original_metrics_verified.csv',checks)
    print(json.dumps(dict(tokenizer=stats,budgets=budgets,dependency=[{k:v for k,v in d.items() if k!='details'} for d in diagnostic]),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
