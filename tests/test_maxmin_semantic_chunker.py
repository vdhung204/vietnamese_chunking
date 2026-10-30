import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.maxmin_semantic_chunker import maxmin_semantic_chunk, process_sentences


class MaxMinTests(unittest.TestCase):
    def test_empty_and_single_sentence(self):
        self.assertEqual(maxmin_semantic_chunk(" \n "), [])
        self.assertEqual(process_sentences([], np.empty((0, 2))), [])
        self.assertEqual(process_sentences(["A."], np.array([[1., 0.]])), [["A."]])

    def test_initial_constant_and_fixed_floor(self):
        vectors = np.array([[1., 0.], [.5, np.sqrt(.75)]])
        self.assertEqual(process_sentences(["A", "B"], vectors), [["A", "B"]])
        self.assertEqual(process_sentences(["A", "B"], vectors, init_constant=1),
                         [["A"], ["B"]])

    def test_equality_starts_new_cluster(self):
        self.assertEqual(process_sentences(["A", "B"], np.eye(2), fixed_threshold=0),
                         [["A"], ["B"]])

    def test_max_similarity_can_match_an_earlier_sentence(self):
        vectors = np.array([[1., 0.], [.5, np.sqrt(.75)], [1., 0.]])
        self.assertEqual(process_sentences(["A", "B", "C"], vectors),
                         [["A", "B", "C"]])

    def test_adaptive_threshold_uses_old_minimum(self):
        # A/B similarity is 1. C's max is .65: above the fixed floor .6,
        # below .9 * sigmoid(1) ~= .658. Updating min first would wrongly join.
        vectors = np.array([[1., 0.], [1., 0.], [.65, np.sqrt(1 - .65**2)]])
        self.assertEqual(process_sentences(["A", "B", "C"], vectors),
                         [["A", "B"], ["C"]])
        self.assertEqual(process_sentences(["A", "B", "C"], vectors, c=.8),
                         [["A", "B", "C"]])

    def test_new_cluster_resets_state_and_keeps_occurrences(self):
        sentences = ["same", "same", "other", "other"]
        vectors = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        clusters = process_sentences(sentences, vectors)
        self.assertEqual(clusters, [["same", "same"], ["other", "other"]])
        self.assertEqual([s for cluster in clusters for s in cluster], sentences)

    def test_adapter_uses_shared_model_and_splitter(self):
        model = Mock()
        model.encode.return_value = np.eye(3)
        with patch.dict(sys.modules, {"semantic_similarity": SimpleNamespace(model=model)}):
            chunks = maxmin_semantic_chunk("Câu một. Câu hai? Câu ba!")
        self.assertEqual(chunks, ["Câu một.", "Câu hai?", "Câu ba!"])
        model.encode.assert_called_once_with(
            chunks, convert_to_numpy=True, show_progress_bar=False)

    def test_embedding_alignment_required(self):
        with self.assertRaises(ValueError):
            process_sentences(["A", "B"], np.ones((1, 2)))


if __name__ == "__main__":
    unittest.main()
