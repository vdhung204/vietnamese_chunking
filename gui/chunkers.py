"""Configure private runtime instances of the frozen Ours modules.

No source rewrite, global monkeypatch, or duplicated chunking implementation.
Each adapter owns its module globals, so Streamlit sessions cannot overwrite
one another's thresholds or the benchmark's imported modules.
"""

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


@dataclass(frozen=True)
class OursSettings:
    min_size: int = 120
    target_size: int = 180
    max_size: int = 240
    semantic_drop: float = .30
    overlap_low: float = .30
    overlap_high: float = .55

    def validate(self):
        sizes = (self.min_size, self.target_size, self.max_size)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in sizes):
            raise ValueError('Độ dài phải là số nguyên.')
        if not 0 < self.min_size < self.target_size < self.max_size:
            raise ValueError('Cần thỏa mãn: 0 < MIN < TARGET < MAX.')
        if not math.isfinite(self.semantic_drop) or not 0 <= self.semantic_drop <= 2:
            raise ValueError('Semantic drop phải nằm trong khoảng 0–2.')
        if not -1 <= self.overlap_low < self.overlap_high <= 1:
            raise ValueError('Cần thỏa mãn: -1 ≤ overlap thấp < overlap cao ≤ 1.')


def _private_module(name):
    spec = importlib.util.spec_from_file_location(f'_playground_{name}', ROOT / 'src' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured_ours(settings: OursSettings):
    settings.validate()
    scorer = _private_module('boundary_scorer')
    overlap = _private_module('dynamic_overlap')
    chunker = _private_module('semantic_chunker')
    scorer.MIN_TOKENS = chunker.MIN_TOKENS = settings.min_size
    scorer.MAX_TOKENS = chunker.MAX_TOKENS = settings.max_size
    scorer.TARGET_TOKENS = settings.target_size
    chunker.SEMANTIC_DROP_THRESHOLD = settings.semantic_drop
    overlap.LOW_SIMILARITY_THRESHOLD = settings.overlap_low
    overlap.HIGH_SIMILARITY_THRESHOLD = settings.overlap_high
    # Bind imported function aliases to their private, configured module globals.
    chunker.score_boundaries = scorer.score_boundaries
    chunker.get_overlap_sentences = overlap.get_overlap_sentences
    return chunker.semantic_chunk
