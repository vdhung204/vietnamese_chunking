import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.compare_chunkers import (
    calculate_boundary_metrics, calculate_dependency_violation_rate,
    retrieval_metrics, validate_documents, validate_questions,
)


class ComparisonTests(unittest.TestCase):
    def test_partial_labels_and_micro_counts(self):
        result = calculate_boundary_metrics([
            dict(predicted_cuts=[0, 1, 3], gold_boundaries=[0, 2], negative_boundaries=[1])])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 1))
        self.assertEqual(result['f1'], 0.5)
        self.assertEqual(result['ignored_predictions'], 1)

    def test_dependency_denominator_is_annotations_not_predicted_cuts(self):
        result = calculate_dependency_violation_rate([
            dict(predicted_cuts=[1, 3, 5], dependency_boundaries=[1, 2]),
            dict(predicted_cuts=[0], dependency_boundaries=None)])
        self.assertEqual(result['total_dependencies'], 2)
        self.assertEqual(result['violations'], 1)
        self.assertEqual(result['rate'], 0.5)

    def test_missing_data_is_not_perfect_score(self):
        self.assertIsNone(calculate_boundary_metrics([])['f1'])
        self.assertIsNone(calculate_dependency_violation_rate([])['rate'])
        self.assertIsNone(retrieval_metrics([], [], None)['recall_at_3'])

    def test_complete_annotations_include_all_negatives(self):
        docs = [dict(id='d', text='A. B. C.', gold_boundaries=[1])]
        validate_documents(docs, {})
        self.assertEqual(docs[0]['negative_boundaries'], [0])

    def test_reject_conflicting_annotation_and_wrong_evidence(self):
        with self.assertRaises(ValueError):
            validate_documents([dict(id='d', text='A. B.', gold_boundaries=[0],
                                     dependency_boundaries=[0])], {})
        with self.assertRaises(ValueError):
            validate_questions([dict(document_id='d', question='Q?', gold_evidence='C.')],
                               [dict(id='d', text='A. B.')])

    def test_retrieval_requires_full_evidence_in_correct_document(self):
        import numpy as np

        class Encoder:
            def encode(self, texts, **kwargs):
                vectors = {'Q?': [1., 0.], 'A. B.': [1., 0.],
                           'A.': [.9, .1], 'B.': [.8, .2], 'Other.': [.7, .3],
                           'A. B. C.': [0., 1.]}
                return np.array([vectors[t] for t in texts])

        chunks = [dict(document_id='wrong', text='A. B.'),
                  *[dict(document_id='d', text=t) for t in ['A.', 'B.', 'Other.', 'A. B. C.']]]
        result = retrieval_metrics(chunks, [dict(document_id='d', question='Q?',
                                                gold_evidence='A. B.')], Encoder())
        self.assertEqual(result['recall_at_3'], 0.0)
        self.assertEqual(result['recall_at_5'], 1.0)
        self.assertTrue(result['details'][0]['evidence_preserved'])
        self.assertIsNone(result['details'][0]['failure_reason'])

    def test_rag_prompt_does_not_receive_gold_labels(self):
        from evaluation.rag_retrieval import build_rag_prompt
        prompt = build_rag_prompt('Question?', [dict(document_id='d', chunk_index=2,
                                                   text='Retrieved text.')])
        self.assertIn('[d:2]', prompt)
        self.assertIn('Retrieved text.', prompt)
        self.assertIn('Question?', prompt)
        self.assertNotIn('gold_evidence', prompt)

    def test_split_evidence_is_diagnosed_separately_from_ranking(self):
        import numpy as np

        class Encoder:
            def encode(self, texts, **kwargs):
                return np.array([[1., 0.] for _ in texts])

        questions = [dict(document_id='d', question='Q?', gold_evidence='A. B.')]
        chunks = [dict(document_id='d', text=t) for t in ['A.', 'B.']]
        result = retrieval_metrics(chunks, questions, Encoder())
        self.assertEqual(result['recall_at_5'], 0.0)
        self.assertEqual(result['details'][0]['failure_reason'], 'evidence_split')

    def test_benchmark_questions_match_source_documents(self):
        import json
        root = Path(__file__).resolve().parents[1]
        questions = json.loads((root / 'data/benchmark/questions.json').read_text(encoding='utf-8'))
        documents = json.loads((root / 'data/test_case.json').read_text(encoding='utf-8'))
        self.assertGreater(len(questions), 0)
        validate_questions(questions, documents)


if __name__ == '__main__':
    unittest.main()
