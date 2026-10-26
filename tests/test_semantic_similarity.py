import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "src" / "semantic_similarity.py"


class SemanticSimilarityTests(unittest.TestCase):
    def setUp(self):
        # Isolate the model dependency: these tests never download model weights.
        self.vectors = {
            "a": [1.0, 0.0],
            "b": [1.0, 0.0],
            "c": [0.0, 1.0],
            "d": [0.0, 1.0],
        }
        self.model = Mock()
        self.model.encode.side_effect = lambda sentence, **kwargs: self.vectors[sentence]
        package = types.ModuleType("sentence_transformers")
        package.SentenceTransformer = Mock(return_value=self.model)
        util = types.ModuleType("sentence_transformers.util")
        util.cos_sim = self.cosine
        spec = importlib.util.spec_from_file_location("similarity_under_test", SOURCE)
        self.similarity = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "sentence_transformers": package,
            "sentence_transformers.util": util,
        }):
            spec.loader.exec_module(self.similarity)

    @staticmethod
    def cosine(left, right):
        dot = sum(a * b for a, b in zip(left, right))
        norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
        return Mock(item=Mock(return_value=dot / norm))

    def test_similarity_values_and_symmetry(self):
        similarity = self.similarity.calculate_semantic_similarity
        self.assertEqual(similarity("a", "b"), 1.0)
        self.assertEqual(similarity("a", "c"), 0.0)
        self.assertEqual(similarity("c", "a"), 0.0)

    def test_shared_sentences_encoded_once_across_drop_and_overlap(self):
        self.assertEqual(
            self.similarity.calculate_semantic_similarity_drop("a", "b", "c", "d"),
            1.0,
        )
        self.similarity.calculate_semantic_similarity("b", "c")
        self.similarity.calculate_semantic_similarity_drop("a", "b", "c", "d")
        self.assertEqual(self.model.encode.call_count, 4)
        self.assertEqual(
            {call.args[0] for call in self.model.encode.call_args_list},
            {"a", "b", "c", "d"},
        )
        for call in self.model.encode.call_args_list:
            self.assertEqual(call.kwargs, {"convert_to_tensor": True})

    def test_empty_inputs_do_not_call_model(self):
        for left, right in [(None, "a"), ("a", None), ("", "a"), ("a", "")]:
            with self.subTest(left=left, right=right):
                self.assertEqual(
                    self.similarity.calculate_semantic_similarity(left, right), 0.0
                )
        self.model.encode.assert_not_called()

    def test_failed_encoding_can_be_retried(self):
        self.model.encode.side_effect = [RuntimeError("temporary failure"), [1.0, 0.0]]
        with self.assertRaises(RuntimeError):
            self.similarity.calculate_semantic_similarity("a", "a")
        self.assertEqual(self.similarity.calculate_semantic_similarity("a", "a"), 1.0)
        self.assertEqual(self.model.encode.call_count, 2)

    def test_cache_is_bounded_and_evicted_sentence_can_be_reencoded(self):
        self.model.encode.side_effect = None
        self.model.encode.return_value = [1.0, 0.0]
        for i in range(513):
            self.similarity.calculate_semantic_similarity(str(i), str(i))
        self.assertEqual(self.model.encode.call_count, 513)
        self.assertLessEqual(self.similarity._encode_sentence.cache_info().currsize, 512)
        self.assertEqual(self.similarity.calculate_semantic_similarity("0", "0"), 1.0)
        self.assertEqual(self.model.encode.call_count, 514)


if __name__ == "__main__":
    unittest.main()
