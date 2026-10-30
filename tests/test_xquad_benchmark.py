import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_xquad_benchmark import SOURCE_HASH, convert, sentence_evidence
from evaluation.evaluate_retrieval import validate_benchmark

DATA = ROOT / 'data/benchmark/xquad_vi'


class XQuADTests(unittest.TestCase):
    def test_release_integrity_and_source_qa_preservation(self):
        raw = (DATA / 'source/xquad.vi.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SOURCE_HASH)
        source = json.loads(raw)
        paragraphs, articles, questions, long_questions, duplicates = convert(source)
        self.assertEqual((len(paragraphs), len(articles), len(questions)), (240, 48, 1190))
        original = {q['id']: q for a in source['data'] for p in a['paragraphs'] for q in p['qas']}
        self.assertEqual(len({q['id'] for q in questions}), 1190)
        for q in questions:
            self.assertEqual(q['question'], original[q['source_question_id']]['question'])
            self.assertEqual(q['source_answers'], original[q['source_question_id']]['answers'])
        for view, docs, qs in (('paragraph', paragraphs, questions), ('article', articles, long_questions)):
            validate_benchmark(docs, qs)
            texts = {d['id']: d['text'] for d in docs}
            for q in qs:
                self.assertEqual(texts[q['document_id']][q['evidence_start']:q['evidence_end']], q['gold_evidence'])
                self.assertIn(q['gold_answer'], q['gold_evidence'])
            self.assertEqual(docs, json.loads((DATA / view / 'full/documents.json').read_text(encoding='utf-8')))
            self.assertEqual(qs, json.loads((DATA / view / 'full/questions.json').read_text(encoding='utf-8')))
        self.assertEqual(len(duplicates), 9)

    def test_no_article_context_or_question_leakage_between_splits(self):
        for view in ('paragraph', 'article'):
            loaded = {}
            for split in ('dev', 'test'):
                docs = json.loads((DATA / view / split / 'documents.json').read_text(encoding='utf-8'))
                qs = json.loads((DATA / view / split / 'questions.json').read_text(encoding='utf-8'))
                validate_benchmark(docs, qs)
                loaded[split] = ({d['article_id'] for d in docs}, {d['text'] for d in docs}, {q['id'] for q in qs})
            for dev, test in zip(loaded['dev'], loaded['test']):
                self.assertFalse(dev & test)
            self.assertEqual(len(loaded['dev'][2] | loaded['test'][2]), 1190)

    def test_sentence_envelope_handles_whitespace_and_cross_sentence_span(self):
        text = 'Alpha.\n\nBeta ends. Gamma.'
        evidence, start, end, count = sentence_evidence(text, dict(text='Alpha.\n\nBeta', answer_start=0))
        self.assertEqual(evidence, 'Alpha.\n\nBeta ends.')
        self.assertEqual(text[start:end], evidence)
        self.assertEqual(count, 2)
        with self.assertRaises(ValueError):
            sentence_evidence(text, dict(text='missing', answer_start=0))


if __name__ == '__main__':
    unittest.main()
