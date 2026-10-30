import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.compare_rag import compare_rag, write_results
from evaluation.evaluate_retrieval import contains_evidence, evaluate_retrieval, validate_benchmark
from rag.embedder import SharedEmbedder
from rag.retriever import CosineRetriever
from rag.rag_pipeline import RAGPipeline
from rag.llm_client import LLMClient, GenerationConfig


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0.] if 'A' in text else [0., 1.] for text in texts])


class RecordingLLM(LLMClient):
    def __init__(self):
        super().__init__(GenerationConfig('test-model'))
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return 'test answer'


class RAGTests(unittest.TestCase):
    def test_global_cosine_not_dot_product_and_stable_ties(self):
        chunks = [dict(document_id=str(i), text=str(i)) for i in range(3)]
        retriever = CosineRetriever(chunks, [[100., 100.], [1., 0.], [2., 0.]])
        ranked = retriever.retrieve_vectors([[1., 0.]], 5)[0]
        self.assertEqual([c['document_id'] for c in ranked], ['1', '2', '0'])
        self.assertEqual([c['rank'] for c in ranked], [1, 2, 3])
        self.assertAlmostEqual(ranked[0]['score'], 1.)
        with self.assertRaises(ValueError):
            retriever.retrieve_vectors([[1., 0.]], 0)

    def test_zero_vectors_are_finite(self):
        retriever = CosineRetriever([dict(text='x')], [[0., 0.]])
        self.assertEqual(retriever.retrieve_vectors([[0., 0.]])[0][0]['score'], 0.)

    def test_normalization_and_source_requirement(self):
        q = dict(document_id='d', gold_evidence='CÂU một.\n Câu hai.')
        self.assertTrue(contains_evidence(dict(document_id='d', text='câu một. câu hai.'), q))
        self.assertFalse(contains_evidence(dict(document_id='other', text=q['gold_evidence']), q))

    def test_recall_mrr_and_split_evidence(self):
        q = dict(id='q', document_id='d', question='Q', gold_evidence='A B')
        miss = dict(document_id='d', text='unrelated')
        hit = dict(document_id='d', text='A B')
        rankings = [[miss, miss, miss, hit, miss]]
        result = evaluate_retrieval([q], rankings, [hit])
        self.assertEqual((result['recall_at_3'], result['recall_at_5'], result['mrr_at_5']), (0, 1, .25))
        split = [dict(document_id='d', text=t) for t in ('A', 'B')]
        result = evaluate_retrieval([q], [split], split)
        self.assertEqual(result['recall_at_5'], 0)
        self.assertEqual(result['details'][0]['failure_reason'], 'evidence_split')

    def test_pipeline_and_comparison_share_generation_and_export(self):
        embedder = SharedEmbedder(FakeModel())
        docs = [dict(id='d1', text='A B'), dict(id='d2', text='C D')]
        questions = [dict(id='q', document_id='d1', question='A?', gold_evidence='A B', gold_answer='SECRET')]
        llm = RecordingLLM()
        chunker = lambda text: [text]
        report = compare_rag(docs, questions, embedder, [('one', chunker), ('two', chunker)], llm=llm)
        self.assertEqual(llm.prompts[0], llm.prompts[1])
        self.assertNotIn('SECRET', llm.prompts[0])
        self.assertEqual(report['results'][0]['recall_at_5'], 1)
        self.assertEqual(report['results'][0]['details'][0]['answer'], 'test answer')
        with tempfile.TemporaryDirectory() as directory:
            write_results(report, Path(directory))
            saved = json.loads((Path(directory) / 'rag_comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['results'][0]['num_chunks'], 2)
            self.assertTrue((Path(directory) / 'rag_comparison.csv').exists())
        pipeline = RAGPipeline(chunker, embedder, llm).index_documents(docs)
        self.assertEqual(pipeline.answer('A?')['answer'], 'test answer')
        self.assertEqual(pipeline.retrieve('A?')[0]['chunk_id'], 'd1:0')

    def test_input_validation_and_missing_llm(self):
        pipeline = RAGPipeline(lambda t: [t], SharedEmbedder(FakeModel()))
        with self.assertRaises(RuntimeError):
            pipeline.retrieve('A')
        with self.assertRaises(RuntimeError):
            pipeline.answer('A')
        with self.assertRaises(ValueError):
            pipeline.index_documents([])
        with self.assertRaises(ValueError):
            validate_benchmark([dict(id='d', text='A')],
                               [dict(id='q', document_id='d', question='Q', gold_evidence='missing')])


if __name__ == '__main__':
    unittest.main()
