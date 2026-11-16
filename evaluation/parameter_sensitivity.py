"""Dev-only sensitivity study; production chunkers and historical results stay frozen.

Run: .\\venv\\Scripts\\python.exe -X utf8 evaluation/parameter_sensitivity.py
No test questions, LLM calls, or automatic changes to production defaults.
"""

import argparse
import ast
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from types import ModuleType
import csv
import gzip

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))


@dataclass(frozen=True)
class Settings:
    minimum: int = 120
    target: int = 180
    maximum: int = 240
    semantic_weight: float = .4
    size_weight: float = .4
    risk_weight: float = .6
    drop: float = .30
    overlap_low: float = .30
    overlap_high: float = .55
    overlap_cap: int = 2
    fixed_overlap: int | None = None
    size_rounding: int | None = 2

    def validate(self):
        if not 0 < self.minimum < self.target < self.maximum:
            raise ValueError('Require 0 < MIN < TARGET < MAX')
        if any(not math.isfinite(w) or w < 0 for w in
               (self.semantic_weight, self.size_weight, self.risk_weight)):
            raise ValueError('Weights must be finite and nonnegative')
        if self.semantic_weight + self.size_weight <= 0:
            raise ValueError('At least one positive feature weight is required')
        if not -1 <= self.overlap_low < self.overlap_high <= 1:
            raise ValueError('Invalid overlap thresholds')


# Frozen baseline of the original study, not the current production default.
REFERENCE = Settings()


def production_settings():
    scorer = private_module('boundary_scorer')
    return replace(REFERENCE, semantic_weight=scorer.SEMANTIC_WEIGHT,
                   size_weight=scorer.SIZE_WEIGHT, risk_weight=scorer.DEPENDENCY_WEIGHT)


def experiment_grid():
    """Fix all grids and the joint search BEFORE seeing dev retrieval scores."""
    settings, families = {}, {}

    def add(family, config):
        config.validate()
        identifier = 'c' + hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:12]
        settings[identifier] = config
        families.setdefault(identifier, set()).add(family)

    add('default', REFERENCE)
    # Fix semantic+size=0.8 to remove redundant positive score rescaling.
    for semantic in (0., .1, .2, .3, .4, .5, .6, .7, .8):
        for risk in (0., .2, .4, .6, .8, 1.2):
            add('weights', replace(REFERENCE, semantic_weight=semantic,
                                   size_weight=round(.8 - semantic, 1), risk_weight=risk))
    for minimum in (60, 90, 120, 150):
        add('minimum', replace(REFERENCE, minimum=minimum))
    for target in (140, 160, 180, 200, 220):
        add('target', replace(REFERENCE, target=target))
    for maximum in (200, 240, 280, 320, 360):
        add('maximum', replace(REFERENCE, maximum=maximum))
    for sizes in ((60, 90, 120), (80, 120, 160), (100, 150, 200),
                  (120, 180, 240), (160, 240, 320), (200, 300, 400)):
        add('size_scale', replace(REFERENCE, minimum=sizes[0], target=sizes[1], maximum=sizes[2]))
    # Drop is bounded above by 2; 3 disables optional cuts unconditionally.
    for drop in (0., .1, .2, .25, .3, .35, .4, .5, .6, 3.):
        add('semantic_drop', replace(REFERENCE, drop=drop))
    for low in (0., .15, .30, .45):
        for high in (.40, .55, .70, .85):
            if low < high:
                add('overlap_thresholds', replace(REFERENCE, overlap_low=low, overlap_high=high))
    for cap in (1, 2, 3):
        add('overlap_cap', replace(REFERENCE, overlap_cap=cap))
    for count in (0, 1, 2, 3):
        add('fixed_overlap', replace(REFERENCE, fixed_overlap=count))
    for digits in (None, 1, 2, 3):
        add('size_rounding', replace(REFERENCE, size_rounding=digits))
    for name, config in {
        'without_size': replace(REFERENCE, size_weight=0.),
        'without_semantic_score': replace(REFERENCE, semantic_weight=0.),
        'without_risk': replace(REFERENCE, risk_weight=0.),
        'without_overlap': replace(REFERENCE, fixed_overlap=0),
        'without_optional_cut': replace(REFERENCE, drop=3.),
        'size_only_score': replace(REFERENCE, semantic_weight=0., risk_weight=0.),
        'semantic_only_score': replace(REFERENCE, size_weight=0., risk_weight=0.),
    }.items():
        add('ablation_' + name, config)
    # Small prespecified joint grid detects interactions, not exhaustive optimization.
    for sizes in ((80, 120, 160), (120, 180, 240), (160, 240, 320)):
        for drop in (.15, .30, .45):
            for low, high in ((.15, .40), (.30, .55), (.45, .70)):
                for semantic, size in ((.2, .6), (.4, .4), (.6, .2)):
                    add('joint', replace(REFERENCE, minimum=sizes[0], target=sizes[1], maximum=sizes[2],
                                         drop=drop, overlap_low=low, overlap_high=high,
                                         semantic_weight=semantic, size_weight=size))
    return settings, {key: sorted(value) for key, value in families.items()}


def private_module(name, weights=None):
    """Reuse the production implementation with private coefficient globals."""
    path = ROOT / 'src' / f'{name}.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    if weights is not None:
        expected = ast.parse('SIZE_WEIGHT * size_score + SEMANTIC_WEIGHT * semantic_score - DEPENDENCY_WEIGHT * dependency_score', mode='eval').body
        found = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'final_score' for t in node.targets):
                if ast.dump(node.value) != ast.dump(expected):
                    raise RuntimeError('Production scoring expression changed; review adapter first')
                found += 1
        if found != 1:
            raise RuntimeError('Expected exactly one scoring expression')
    module = ModuleType(f'_study_{name}')
    module.__file__ = str(path)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    if weights is not None:
        module.SIZE_WEIGHT = weights.size_weight
        module.SEMANTIC_WEIGHT = weights.semantic_weight
        module.DEPENDENCY_WEIGHT = weights.risk_weight
    return module


def configured_chunker(config):
    config.validate()
    scorer = private_module('boundary_scorer', config)
    overlap = private_module('dynamic_overlap')
    chunker = private_module('semantic_chunker')
    scorer.MIN_TOKENS = chunker.MIN_TOKENS = config.minimum
    scorer.TARGET_TOKENS = config.target
    scorer.MAX_TOKENS = chunker.MAX_TOKENS = config.maximum
    chunker.SEMANTIC_DROP_THRESHOLD = config.drop
    if config.size_rounding != 2:
        original_round = round
        scorer.round = lambda value, _: value if config.size_rounding is None else original_round(value, config.size_rounding)
    overlap.LOW_SIMILARITY_THRESHOLD = config.overlap_low
    overlap.HIGH_SIMILARITY_THRESHOLD = config.overlap_high
    original_count = overlap.determine_overlap_count

    def overlap_count(left, right):
        if config.fixed_overlap is not None:
            return config.fixed_overlap
        count = original_count(left, right)
        return config.overlap_cap if count == 2 else count

    overlap.determine_overlap_count = overlap_count
    chunker.get_overlap_sentences = overlap.get_overlap_sentences
    stats = Counter()
    risky_events = []
    original_score = scorer.score_boundaries

    def audited_scores(boundaries):
        scores = original_score(boundaries)
        stats['scoring_calls'] += 1
        eligible = [i for i, b in enumerate(boundaries) if config.minimum <= b.token_count <= config.maximum]
        risky = [i for i in eligible if scorer.dependency_risk(boundaries[i].left_sentence, boundaries[i].right_sentence)]
        stats['eligible_candidates'] += len(eligible)
        stats['risky_candidates'] += len(risky)
        stats['mixed_risk_calls'] += bool(risky and len(risky) < len(eligible))
        if eligible:
            selected = scores.index(max(scores))
            stats['selected_risky'] += selected in risky
            if risky:
                risky_events.append(dict(selected_position=boundaries[selected].position,
                                         selected_risky=selected in risky,
                                         risky_positions=[boundaries[i].position for i in risky]))
        return scores

    chunker.score_boundaries = audited_scores
    return chunker.semantic_chunk, stats, risky_events


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes():
    paths = [*sorted((ROOT / 'src').rglob('*.py')), *sorted((ROOT / 'baselines').glob('*.py'))]
    return {str(path.relative_to(ROOT)): hash_file(path) for path in paths}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_json_gzip(path, data):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(',', ':'))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'evaluation/results/parameter_study')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    # Deliberately use only locally cached weights. No credentials or downloads.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    configs, families = experiment_grid()
    before = frozen_hashes()
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        data='XQuAD.vi dev only: 233 identical question IDs, 10 source titles, two document views',
        primary='Equal-weight mean of paragraph/article dev Evidence Recall@5',
        secondary='Equal-weight mean dev MRR@5; retain all tied configurations',
        interpretation='Sensitivity and small joint grid; not exhaustive or unique global optimization. '
                       'Two views reuse the same questions and are not independent replications. '
                       'New exploratory study after historical test results were already seen; '
                       'does not retrospectively establish how original defaults were selected.',
        weights='semantic+size fixed at 0.8; nonnegative penalty. Only relative score weights matter.',
        unchanged='Production src/baselines, historical test results, labels, model length, rule matcher, optional-cut branch',
        grid={key: dict(parameters=asdict(value), families=families[key]) for key, value in configs.items()},
        retrieval_k=[1, 3, 5, 10], llm=None, threads=args.threads,
    )
    write_json(args.output_dir / 'protocol.json', protocol)
    print(f'Frozen protocol: {len(configs)} configurations, dev only.', flush=True)
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    np.random.seed(2026)
    print('Loading cached MiniLM...', flush=True)
    import semantic_similarity
    import evaluation.evaluate_retrieval as retrieval_evaluation
    from semantic_chunker import semantic_chunk
    from sentence_splitter import split_sentences
    from rag.retriever import CosineRetriever
    from evaluation.evaluate_retrieval import contains_evidence, evaluate_retrieval, validate_benchmark
    model = semantic_similarity.model
    model.to('cpu')
    model.eval()
    active_settings = production_settings()
    # This pure normalization cache changes no match decisions.
    retrieval_evaluation.normalize = lru_cache(maxsize=None)(retrieval_evaluation.normalize)
    # Same per-sentence model.encode calls; keep real embeddings for this process
    # rather than evicting at 512 during repeated configurations. Never fake vectors.
    semantic_similarity._encode_sentence = lru_cache(maxsize=None)(semantic_similarity._encode_sentence.__wrapped__)
    datasets, input_hashes = {}, {}
    for view in ('paragraph', 'article'):
        base = ROOT / f'data/benchmark/xquad_vi/{view}/dev'
        docs = json.loads((base / 'documents.json').read_text(encoding='utf-8'))
        questions = json.loads((base / 'questions.json').read_text(encoding='utf-8'))
        assert all(row['split'] == 'dev' for row in docs + questions)
        validate_benchmark(docs, questions)
        datasets[view] = (docs, questions)
        for filename in ('documents.json', 'questions.json'):
            input_hashes[str((base / filename).relative_to(ROOT))] = hash_file(base / filename)
    assert [q['id'] for q in datasets['paragraph'][1]] == [q['id'] for q in datasets['article'][1]]
    sentences = dict.fromkeys(s for docs, _ in datasets.values() for doc in docs for s in split_sentences(doc['text']))
    start = perf_counter()
    for sentence in sentences:
        semantic_similarity._encode_sentence(sentence)
    print(f'Encoded {len(sentences)} distinct real sentences in {perf_counter()-start:.1f}s.', flush=True)
    query_vectors = model.encode([q['question'] for q in datasets['paragraph'][1]],
                                 convert_to_numpy=True, show_progress_bar=False)
    corpora, diagnostics = {}, {}
    for index, (identifier, config) in enumerate(configs.items(), 1):
        for view, (docs, _) in datasets.items():
            chunker, stats, events = configured_chunker(config)
            chunks = []
            for doc in docs:
                output = chunker(doc['text'])
                if config == active_settings:
                    assert output == semantic_chunk(doc['text']), 'Default adapter changed production output'
                chunks.extend(dict(document_id=doc['id'], chunk_index=i, chunk_id=f"{doc['id']}:{i}", text=text)
                              for i, text in enumerate(output))
            corpora[(identifier, view)] = chunks
            diagnostics[(identifier, view)] = dict(counts=dict(stats), risky_events=events)
        if index % 20 == 0 or index == len(configs):
            print(f'Chunked {index}/{len(configs)} configurations.', flush=True)
    texts = list(dict.fromkeys(chunk['text'] for chunks in corpora.values() for chunk in chunks))
    print(f'Embedding {len(texts)} unique chunk texts for all configurations...', flush=True)
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    text_index = {text: i for i, text in enumerate(texts)}
    evaluation_cache, rows, details = {}, [], {}
    default_rankings = {}
    for (identifier, view), chunks in corpora.items():
        signature = hashlib.sha256(json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache_key = (view, signature)
        if cache_key not in evaluation_cache:
            chunk_vectors = np.asarray([vectors[text_index[c['text']]] for c in chunks])
            ranked = CosineRetriever(chunks, chunk_vectors).retrieve_vectors(query_vectors, 10)
            metrics = evaluate_retrieval(datasets[view][1], [r[:5] for r in ranked], chunks)
            evaluation_cache[cache_key] = (metrics, ranked)
        metrics, ranked = evaluation_cache[cache_key]
        config = configs[identifier]
        if config == REFERENCE:
            default_rankings[view] = ranked
        details[f'{identifier}/{view}'] = dict(corpus_sha256=signature,
            questions=[dict(question_id=d['question_id'], hit_at_3=d['hit_at_3'], hit_at_5=d['hit_at_5'],
                            reciprocal_rank_at_5=d['reciprocal_rank_at_5'], failure_reason=d['failure_reason'],
                            context_words=d['context_words'],
                            retrieved_ids=[c['chunk_id'] for c in d['retrieved']]) for d in metrics['details']],
            **diagnostics[(identifier, view)])
        rows.append(dict(config_id=identifier, view=view, families=';'.join(families[identifier]),
            **asdict(config), num_documents=len(datasets[view][0]), num_questions=len(datasets[view][1]),
            num_chunks=len(chunks), avg_chunk_words=sum(len(c['text'].split()) for c in chunks)/len(chunks),
            **{key: value for key, value in metrics.items() if key != 'details'},
            **diagnostics[(identifier, view)]['counts'], corpus_sha256=signature))
    # Fixed column set even if an audit counter never occurred.
    count_fields = ('scoring_calls', 'eligible_candidates', 'risky_candidates', 'mixed_risk_calls', 'selected_risky')
    for row in rows:
        for field in count_fields:
            row.setdefault(field, 0)
    aggregate = []
    for identifier, config in configs.items():
        pair = [r for r in rows if r['config_id'] == identifier]
        aggregate.append(dict(config_id=identifier, families=';'.join(families[identifier]), **asdict(config),
            mean_recall_at_5=sum(r['recall_at_5'] for r in pair)/2,
            mean_mrr_at_5=sum(r['mrr_at_5'] for r in pair)/2,
            mean_context_words=sum(r['mean_context_words'] for r in pair)/2))
    aggregate.sort(key=lambda r: (-r['mean_recall_at_5'], -r['mean_mrr_at_5'], r['config_id']))
    best = aggregate[0]
    winners = [r['config_id'] for r in aggregate if
               abs(r['mean_recall_at_5']-best['mean_recall_at_5']) < 1e-12 and
               abs(r['mean_mrr_at_5']-best['mean_mrr_at_5']) < 1e-12]
    default_id = next(key for key, value in configs.items() if value == REFERENCE)
    family_summary = {}
    for family in sorted({f for fs in families.values() for f in fs}):
        candidates = [r for r in aggregate if family in families[r['config_id']] or r['config_id'] == default_id]
        top = candidates[0]
        family_summary[family] = dict(configurations=len(candidates), best_config_id=top['config_id'],
            best_mean_recall_at_5=top['mean_recall_at_5'], best_mean_mrr_at_5=top['mean_mrr_at_5'],
            default_tied_on_primary=abs(next(r for r in aggregate if r['config_id']==default_id)['mean_recall_at_5']-top['mean_recall_at_5'])<1e-12)
    k_rows = []
    for view, ranked in default_rankings.items():
        questions = datasets[view][1]
        for k in (1, 3, 5, 10):
            first = [next((i for i, c in enumerate(r[:k], 1) if contains_evidence(c, q)), None)
                     for q, r in zip(questions, ranked)]
            k_rows.append(dict(view=view, method='Ours default', k=k,
                recall=sum(r is not None for r in first)/len(first),
                mrr=sum(1/r if r else 0 for r in first)/len(first),
                mean_context_words=sum(sum(len(c['text'].split()) for c in r[:k]) for r in ranked)/len(ranked)))
    assert frozen_hashes() == before
    from huggingface_hub import try_to_load_from_cache
    cached_config = Path(try_to_load_from_cache(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'config.json'))
    snapshot = cached_config.parent
    summary = dict(default_config_id=default_id, default=next(r for r in aggregate if r['config_id']==default_id),
        tested_configurations=len(configs), distinct_corpora=len(evaluation_cache),
        best_config_ids=winners, best=best, by_family=family_summary,
        note='Dev winners are selected estimates, not independent confirmation or proof of a global optimum.',
        production_parameters=asdict(active_settings),
        production_adapter_matches_original_on_all_60_dev_documents=True,
        provenance=dict(python=platform.python_version(), packages={p:importlib.metadata.version(p) for p in
            ('torch','numpy','sentence-transformers','transformers','scikit-learn')},
            device=str(model.device), max_seq_length=model.max_seq_length, seed=2026,
            script_sha256=hash_file(Path(__file__)), inputs_sha256=input_hashes,
            model_snapshot=str(snapshot), model_files_sha256={str(p.relative_to(snapshot)):hash_file(p)
                for p in sorted(snapshot.rglob('*')) if p.is_file() and p.suffix in ('.json','.safetensors')},
            source_sha256_before=before, source_sha256_after=frozen_hashes(),
            embedding_cache='Real single-sentence encode, unbounded process-local cache. Unique chunks batch encoded once.',
            no_test_data_or_llm=True))
    write_csv(args.output_dir/'sensitivity.csv', rows)
    write_csv(args.output_dir/'aggregate.csv', aggregate)
    write_csv(args.output_dir/'retrieval_k.csv', k_rows)
    write_json(args.output_dir/'summary.json', summary)
    write_json_gzip(args.output_dir/'details.json.gz', details)
    # Store distinct corpus content, embeddings, queries, and configuration mapping
    # so every ranking/metric can be recomputed without calling the model again.
    saved_corpora = {}
    mapping = {}
    for key, chunks in corpora.items():
        signature = details[f'{key[0]}/{key[1]}']['corpus_sha256']
        saved_corpora.setdefault(signature, chunks)
        mapping[f'{key[0]}/{key[1]}'] = signature
    write_json_gzip(args.output_dir/'corpora.json.gz', dict(texts=texts, corpora=saved_corpora, mapping=mapping))
    np.savez_compressed(args.output_dir/'embeddings.npz', chunk_vectors=vectors, query_vectors=query_vectors)
    print(json.dumps(dict(default=summary['default'], best=best, best_config_ids=winners), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
