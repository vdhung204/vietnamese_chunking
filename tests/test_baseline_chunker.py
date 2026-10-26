import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src" / "baseline_chunker.py"
SPEC = importlib.util.spec_from_file_location("baseline_under_test", SOURCE)
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)
fixed_chunk = BASELINE.fixed_chunk


class FixedChunkTests(unittest.TestCase):
    def test_overlap_preserves_order_and_last_partial_chunk(self):
        self.assertEqual(
            fixed_chunk("a b c d e f g h", chunk_size=4, overlap=1),
            ["a b c d", "d e f g", "g h"],
        )

    def test_zero_overlap_covers_input_once(self):
        text = "một hai ba bốn năm"
        chunks = fixed_chunk(text, chunk_size=2, overlap=0)
        self.assertEqual(chunks, ["một hai", "ba bốn", "năm"])
        self.assertEqual(" ".join(chunks), text)

    def test_empty_and_whitespace_input(self):
        for text in ["", " \n\t "]:
            with self.subTest(text=text):
                self.assertEqual(fixed_chunk(text), [])

    def test_exact_size_has_no_extra_chunk(self):
        self.assertEqual(fixed_chunk("a b c", 3, 2), ["a b c"])

    def test_nonpositive_chunk_size_rejected(self):
        for size in [0, -1]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                fixed_chunk("a b c", size, 0)

    def test_overlap_that_prevents_progress_rejected(self):
        for overlap in [-1, 3, 4]:
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                fixed_chunk("a b c d", 3, overlap)

    def test_noninteger_sizes_rejected(self):
        for value in [True, False, 2.5, "3", None]:
            with self.subTest(chunk_size=value), self.assertRaises(TypeError):
                fixed_chunk("a b c", value, 0)
            with self.subTest(overlap=value), self.assertRaises(TypeError):
                fixed_chunk("a b c", 3, value)

    def test_nonstring_text_rejected(self):
        for text in [None, 42, ["a", "b"]]:
            with self.subTest(text=text), self.assertRaises(TypeError):
                fixed_chunk(text)


if __name__ == "__main__":
    unittest.main()
