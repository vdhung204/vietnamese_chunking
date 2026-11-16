"""Verify experimental adapters against production without downloading a model."""

from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation.parameter_sensitivity import REFERENCE, configured_chunker, experiment_grid, private_module, production_settings


class ParameterStudyTests(unittest.TestCase):
    def setUp(self):
        fake = ModuleType('semantic_similarity')
        fake.calculate_semantic_similarity = lambda a, b: .65
        fake.calculate_semantic_similarity_drop = lambda a, b, c, d: .4 if 'B.' in b else .1
        self.modules = patch.dict(sys.modules, {'semantic_similarity': fake})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_default_adapter_reproduces_production_across_length_branches(self):
        production = private_module('semantic_chunker').semantic_chunk
        active = production_settings()
        self.assertEqual((active.semantic_weight, active.size_weight, active.risk_weight), (.5, .3, .6))
        configured, stats, _ = configured_chunker(active)
        sentences = [' '.join([letter] * 39) + f' {letter}.' for letter in 'ABCDEFGH']
        for length in (1, 4, 6, 8):
            text = ' '.join(sentences[:length])
            self.assertEqual(configured(text), production(text))
        self.assertGreater(stats['scoring_calls'], 0)

    def test_private_score_literals_change_ranking_and_positive_scale_preserves_it(self):
        from boundary import create_candidate_boundaries
        from sentence_splitter import split_sentences
        text = ' '.join(' '.join([letter] * 59) + f' {letter}.' for letter in 'ABCDE')
        boundaries = create_candidate_boundaries(split_sentences(text))
        size = private_module('boundary_scorer', replace(REFERENCE, semantic_weight=0., size_weight=.8))
        semantic = private_module('boundary_scorer', replace(REFERENCE, semantic_weight=.8, size_weight=0.))
        self.assertNotEqual(size.score_boundaries(boundaries).index(max(size.score_boundaries(boundaries))),
                            semantic.score_boundaries(boundaries).index(max(semantic.score_boundaries(boundaries))))
        original = private_module('boundary_scorer', REFERENCE).score_boundaries(boundaries)
        scaled = private_module('boundary_scorer', replace(REFERENCE, semantic_weight=.8, size_weight=.8, risk_weight=1.2)).score_boundaries(boundaries)
        self.assertEqual(scaled, [2 * score for score in original])

    def test_fixed_overlap_and_cap_do_not_change_core_cut_locations(self):
        text = ' '.join(' '.join([letter] * 39) + f' {letter}.' for letter in 'ABCDEFGH')
        none, _, _ = configured_chunker(replace(REFERENCE, fixed_overlap=0))
        fixed, _, _ = configured_chunker(replace(REFERENCE, fixed_overlap=1))
        capped, _, _ = configured_chunker(replace(REFERENCE, overlap_cap=1))
        plain = none(text)
        self.assertGreater(len(plain), 1)
        self.assertEqual(fixed(text), capped(text))
        self.assertTrue(all(output.endswith(core) for output, core in zip(fixed(text), plain)))

    def test_grid_contains_original_and_all_declared_families(self):
        configs, families = experiment_grid()
        self.assertEqual(len(configs), 180)
        self.assertEqual(sum(config == REFERENCE for config in configs.values()), 1)
        for config in configs.values():
            config.validate()
        names = {name for values in families.values() for name in values}
        self.assertTrue({'weights', 'minimum', 'target', 'maximum', 'size_scale', 'semantic_drop',
                         'overlap_thresholds', 'overlap_cap', 'fixed_overlap', 'joint', 'size_rounding'} <= names)


if __name__ == '__main__':
    unittest.main()
