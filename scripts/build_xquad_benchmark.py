"""Reproducible Vietnamese XQuAD adaptation; no chunker predictions are used.

The source is CC BY-SA 4.0, from Google DeepMind's XQuAD release. The derived
data retain that license. See data/benchmark/xquad_vi/README.md for limitations.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from sentence_splitter import split_sentences
from evaluation.evaluate_retrieval import normalize, validate_benchmark

REVISION = '7d30520c717524000f0d9d2f9c10a069acd9d285'
BASE_URL = f'https://raw.githubusercontent.com/google-deepmind/xquad/{REVISION}/'
SOURCE_HASH = 'f619a1eb11fb42d3ab0834259e488a65f585447ef6154437bfb7199d85161a04'
SPLIT_SEED = 'utc-rag-xquad-vi-v1'
DOMAINS = {
    'science_technology': ['Nikola_Tesla', 'Computational_complexity_theory', 'Steam_engine',
                           'Oxygen', 'Apollo_program', 'Packet_switching', 'Geology', 'Prime_number', 'Force'],
    'nature_health': ['Amazon_rainforest', 'Ctenophora', 'Black_Death', 'Pharmacy',
                      'Immune_system', 'Intergovernmental_Panel_on_Climate_Change', 'Chloroplast'],
    'history_religion': ['Normans', 'Martin_Luther', 'Huguenot', 'Genghis_Khan',
                         'Yuan_dynasty', 'Islamism', 'Imperialism', 'United_Methodist_Church', 'French_and_Indian_War'],
    'geography': ['Warsaw', 'Southern_California', 'Victoria_(Australia)', 'Fresno,_California',
                  'Newcastle_upon_Tyne', 'Jacksonville,_Florida', 'Kenya', 'Rhine'],
    'education': ['Teacher', 'Private_school', 'Harvard_University', 'University_of_Chicago'],
    'society_economy_law': ['1973_oil_crisis', 'European_Union_law', 'Civil_disobedience',
                           'Construction', 'Economic_inequality', 'Scottish_Parliament'],
    'culture_sport_media': ['Super_Bowl_50', 'Sky_(United_Kingdom)', 'Victoria_and_Albert_Museum',
                            'American_Broadcasting_Company', 'Doctor_Who'],
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sentence_evidence(context, answer):
    """Smallest envelope of project-split sentences covering the source answer.

    This is an automatic proxy for evidence, not a human supporting-fact label.
    Character offsets refer to the unchanged source context.
    """
    start, end = answer['answer_start'], answer['answer_start'] + len(answer['text'])
    if not answer['text'].strip() or context[start:end] != answer['text']:
        raise ValueError('Source answer offsets do not align')
    spans, cursor = [], 0
    for sentence in split_sentences(context):
        position = context.find(sentence, cursor)
        if position < 0:
            raise ValueError('Cannot align sentence with source')
        spans.append((position, position + len(sentence)))
        cursor = position + len(sentence)
    covered = [(a, b) for a, b in spans if a < end and b > start]
    if not covered:
        raise ValueError('No sentence covers source answer')
    lo, hi = covered[0][0], covered[-1][1]
    return context[lo:hi], lo, hi, len(covered)


def title_splits(titles):
    """Freeze ~20% per broad domain as dev, remaining titles as test."""
    if set(titles) != {t for group in DOMAINS.values() for t in group}:
        raise ValueError('Domain map must cover source titles exactly')
    assigned = {}
    for domain, group in DOMAINS.items():
        ranked = sorted(group, key=lambda title: digest(f'{SPLIT_SEED}:{title}'.encode()))
        dev_count = max(1, round(.2 * len(ranked)))
        for i, title in enumerate(ranked):
            assigned[title] = (domain, 'dev' if i < dev_count else 'test')
    return assigned


def convert(source):
    assignment = title_splits([a['title'] for a in source['data']])
    paragraphs, articles, questions, long_questions = [], [], [], []
    duplicate_groups = defaultdict(list)
    for ai, article in enumerate(source['data']):
        title = article['title']
        domain, split = assignment[title]
        article_id = f'xquad_vi_a{ai:03d}'
        contents, origins, offset = [], [], 0
        for pi, paragraph in enumerate(article['paragraphs']):
            context = paragraph['context']
            doc_id = f'{article_id}_p{pi:03d}'
            metadata = dict(title=title, article_id=article_id, domain=domain, split=split,
                            source_dataset='XQuAD.vi', source_revision=REVISION,
                            source_url=BASE_URL + 'xquad.vi.json', source_paragraph_index=pi)
            paragraphs.append(dict(id=doc_id, text=context, **metadata))
            origins.append(dict(document_id=doc_id, start=offset, end=offset + len(context)))
            for qa in paragraph['qas']:
                for answer in qa['answers']:
                    start = answer['answer_start']
                    if context[start:start + len(answer['text'])] != answer['text']:
                        raise ValueError(f"Invalid original answer: {qa['id']}")
                first = qa['answers'][0]
                evidence, lo, hi, count = sentence_evidence(context, first)
                record = dict(
                    id='xquad_vi_' + qa['id'], document_id=doc_id, question=qa['question'],
                    gold_answer=first['text'], gold_answers=[a['text'] for a in qa['answers']],
                    gold_evidence=evidence, evidence_start=lo, evidence_end=hi,
                    evidence_sentence_count=count, source_answers=qa['answers'],
                    source_question_id=qa['id'], article_id=article_id, title=title,
                    domain=domain, split=split, source_dataset='XQuAD.vi',
                    evidence_annotation='automatic_sentence_envelope_of_first_source_answer',
                    review_status='source_QA_preserved_local_evidence_not_human_reviewed')
                questions.append(record)
                long_questions.append(dict(record, document_id=article_id,
                    source_document_id=doc_id, evidence_start=lo + offset, evidence_end=hi + offset))
                duplicate_groups[normalize(qa['question'])].append(record)
            contents.append(context)
            offset += len(context) + 2
        articles.append(dict(id=article_id, text='\n\n'.join(contents), title=title,
                             article_id=article_id, domain=domain, split=split,
                             construction='source_paragraphs_same_title_joined_in_release_order',
                             source_dataset='XQuAD.vi', source_revision=REVISION,
                             source_url=BASE_URL + 'xquad.vi.json', source_paragraphs=origins))
    duplicates = [dict(question=key, ids=[q['id'] for q in values],
                       answers=[q['gold_answer'] for q in values],
                       document_ids=[q['document_id'] for q in values],
                       answer_disagreement=len({normalize(q['gold_answer']) for q in values}) > 1)
                  for key, values in duplicate_groups.items() if len(values) > 1]
    return paragraphs, articles, questions, long_questions, duplicates


def describe(documents, questions):
    lengths = [len(d['text'].split()) for d in documents]
    return dict(documents=len(documents), questions=len(questions),
                independent_source_titles=len({d['article_id'] for d in documents}),
                whitespace_units=dict(min=min(lengths), median=statistics.median(lengths),
                                      max=max(lengths), total=sum(lengths)),
                length_bands=dict(Counter('short_lt120' if n < 120 else
                                         'medium_120_240' if n <= 240 else 'long_gt240' for n in lengths)),
                documents_by_domain=dict(Counter(d['domain'] for d in documents)),
                questions_by_domain=dict(Counter(q['domain'] for q in questions)))


def build(output_dir, offline=False):
    raw_dir = output_dir / 'source'
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in ('xquad.vi.json', 'CC-BY-SA4.0.txt', 'README.md'):
        path = raw_dir / filename
        if not path.exists():
            if offline:
                raise FileNotFoundError(path)
            with urlopen(BASE_URL + filename, timeout=60) as response:
                path.write_bytes(response.read())
    source_bytes = (raw_dir / 'xquad.vi.json').read_bytes()
    if digest(source_bytes) != SOURCE_HASH:
        raise ValueError('Source checksum differs from the pinned release')
    paragraphs, articles, questions, long_questions, duplicates = convert(json.loads(source_bytes))
    summaries = {}
    for view, docs, qs in (('paragraph', paragraphs, questions), ('article', articles, long_questions)):
        for split in ('full', 'dev', 'test'):
            selected_docs = [d for d in docs if split == 'full' or d['split'] == split]
            selected_qs = [q for q in qs if split == 'full' or q['split'] == split]
            validate_benchmark(selected_docs, selected_qs)
            by_id = {d['id']: d['text'] for d in selected_docs}
            for q in selected_qs:
                assert by_id[q['document_id']][q['evidence_start']:q['evidence_end']] == q['gold_evidence']
                assert q['gold_answer'] in q['gold_evidence']
            write_json(output_dir / view / split / 'documents.json', selected_docs)
            write_json(output_dir / view / split / 'questions.json', selected_qs)
            summaries[f'{view}/{split}'] = describe(selected_docs, selected_qs)
    write_json(output_dir / 'duplicate_question_audit.json', duplicates)
    with (output_dir / 'review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['id', 'title', 'domain', 'split', 'question', 'gold_answer', 'gold_evidence',
                  'duplicate_question', 'source_label_suspect', 'reviewed', 'review_note']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        duplicate_ids = {qid for group in duplicates for qid in group['ids']}
        # These issues were noticed in the source BEFORE any chunker evaluation.
        # Keep original labels, but make their status explicit for human review.
        suspects = {
            '56beb4343aeaaa14008c925e': 'Question asks Josh Norman interceptions; answer points to Pro Bowl selections.',
            '56d6f3500d65d21400198293': 'Question asks Thomas Davis fumbles; answer points to Pro Bowl selections.',
            '56d9992fdc89441400fdb5a0': 'Question asks Josh Norman interceptions; answer points to Mario Addison recoveries.',
        }
        for q in questions:
            note = suspects.get(q['source_question_id'], '')
            writer.writerow(dict(q, duplicate_question=q['id'] in duplicate_ids,
                                 source_label_suspect=bool(note), reviewed='', review_note=note))
    manifest = dict(
        source=dict(dataset='XQuAD Vietnamese', repository='https://github.com/google-deepmind/xquad',
                    revision=REVISION, sha256=SOURCE_HASH, license='CC-BY-SA-4.0',
                    attribution='Artetxe, Ruder and Yogatama (2020), On the Cross-lingual Transferability of Monolingual Representations',
                    paper='https://aclanthology.org/2020.acl-main.421/'),
        adaptation_license='CC-BY-SA-4.0', split_seed=SPLIT_SEED,
        split_policy='Disjoint source titles, hash order within each manually assigned broad domain; ~20% dev',
        evidence_policy='Whole sentence envelope around first original answer, using unchanged project sentence splitter',
        article_policy='Concatenate ONLY supplied paragraphs of the same title in source release order; not full original articles',
        qa_authorship='Original published QA and translations retained; no assistant-generated questions or answers',
        label_review='Mechanical validation only; evidence derived automatically; see review.csv for known source issues',
        original_questions_retained=len(questions), unique_normalized_questions=len({normalize(q['question']) for q in questions}),
        repeated_question_groups=len(duplicates),
        source_context_duplicates=len(paragraphs) - len({normalize(d['text']) for d in paragraphs}),
        summaries=summaries)
    manifest['artifact_sha256'] = {
        str(p.relative_to(output_dir)): digest(p.read_bytes())
        for p in sorted(output_dir.rglob('*')) if p.is_file() and p.name != 'manifest.json' and p.suffix in ('.json', '.csv')}
    write_json(output_dir / 'manifest.json', manifest)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/benchmark/xquad_vi')
    parser.add_argument('--offline', action='store_true')
    arguments = parser.parse_args()
    build(arguments.output_dir, arguments.offline)
