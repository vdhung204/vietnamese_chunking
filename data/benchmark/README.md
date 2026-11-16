# Benchmark datasets

The main evaluation uses a local retrieval adaptation of [Vietnamese XQuAD](xquad_vi/README.md). The smaller pilot remains available for development diagnostics. Run commands from the repository root; see the [experiment guide](../../docs/EXPERIMENTS.md).

| Dataset | Purpose | Size | Location |
|---|---|---|---|
| Vietnamese XQuAD dev | Parameter sensitivity study | 233 questions from 10 source titles | `xquad_vi/{paragraph,article}/dev/` |
| Vietnamese XQuAD test | Reported retrieval results | 957 questions from 38 source titles | `xquad_vi/{paragraph,article}/test/` |
| Original pilot | Development and partial boundary checks | 13 documents and 16 retrieval questions | `data/test_case.json`, `questions.json`, `annotations.json` |

XQuAD contains 240 source contexts and 1,190 questions from 48 source titles in total. The `paragraph` and `article` views share the same content and questions; they are not independent datasets. Development and test sets are disjoint by source title. The dataset documentation covers source attribution, licensing, the split, and automatically derived evidence labels.

## Current results

The manuscript uses the `0.5/0.3/0.6` configuration and artifacts in [scientific_paper_current](../../evaluation/results/scientific_paper_current). See [verified metrics](../../evaluation/results/scientific_paper_current/metrics_verified.csv) and the [run protocol](../../evaluation/results/scientific_paper_current/protocol.json).

The [30 October results](xquad_vi/RESULTS_2026-10-30.md) belong to the earlier `0.4/0.4/0.6` configuration and are retained as historical evidence.

## Pilot annotations

The pilot uses partial `must_cut_after` / `must_not_cut_after` constraints. Metrics at these positions are not a fully annotated Boundary F1 benchmark. Dependency annotations cover only explicitly reviewed pairs. The 16 retrieval questions are assistant-authored development examples, rather than a human-reviewed held-out dataset.

`evaluation/compare_rag.py` defaults to this pilot when dataset paths are omitted. Use explicit `--documents`, `--questions`, and `--output-dir` arguments for XQuAD. Detailed pilot schemas and the historical LlamaIndex evaluation are preserved in the [archived benchmark guide](../../docs/archive/PILOT_BENCHMARK.md).
