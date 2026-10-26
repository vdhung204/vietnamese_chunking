import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from semantic_chunker import semantic_chunk
from sentence_splitter import split_sentences


def load_test_cases(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    return test_cases

def extract_cut_positions(
        text: str,
        chunks: list[str]
) -> list[int]:
    if not chunks:
        return []

    original_sentences = split_sentences(text)

    cut_positions = []
    search_start = 0

    for chunk in chunks[:-1]:
        chunk_sentences = split_sentences(chunk)

        if not chunk_sentences:
            continue

        last_sentence = chunk_sentences[-1]

        for i in range(search_start, len(original_sentences)):
            if original_sentences[i] == last_sentence:
                cut_positions.append(i)
                search_start = i + 1
                break

    return cut_positions

def evaluate_case(case: dict) -> dict:
    text = case["text"]
    expected = case["expected"]

    chunks = semantic_chunk(text)

    predicted_cuts = extract_cut_positions(
        text,
        chunks
    )

    predicted_overlaps = extract_overlap_counts(chunks)

    must_cut = expected.get("must_cut_after", [])
    must_not_cut = expected.get("must_not_cut_after", [])

    min_cuts = expected.get("min_cuts", 0)
    expected_overlap = expected.get("expected_overlap")

    # 1. Kiểm tra các boundary bắt buộc phải cắt
    must_cut_pass = all(
        pos in predicted_cuts
        for pos in must_cut
    )

    # 2. Kiểm tra các boundary không được cắt
    must_not_cut_pass = all(
        pos not in predicted_cuts
        for pos in must_not_cut
    )

    # 3. Kiểm tra số lượng cut tối thiểu
    min_cuts_pass = (
        len(predicted_cuts) >= min_cuts
    )

    # 4. Kiểm tra overlap
    # expected_overlap = null
    # -> không đánh giá overlap cho case này
    if expected_overlap is None:
        overlap_pass = True

    else:
        overlap_pass = (
            len(predicted_overlaps) > 0
            and all(
                overlap == expected_overlap
                for overlap in predicted_overlaps
            )
        )

    passed = (
        must_cut_pass
        and must_not_cut_pass
        and min_cuts_pass
        and overlap_pass
    )

    return {
        "id": case["id"],
        "name": case["name"],

        "predicted_cuts": predicted_cuts,
        "predicted_overlaps": predicted_overlaps,

        "expected_must_cut": must_cut,
        "expected_must_not_cut": must_not_cut,
        "expected_overlap": expected_overlap,

        "min_cuts": min_cuts,

        "must_cut_pass": must_cut_pass,
        "must_not_cut_pass": must_not_cut_pass,
        "min_cuts_pass": min_cuts_pass,
        "overlap_pass": overlap_pass,

        "passed": passed
    }

def calculate_metrics(results: list[dict]) -> dict:
    total_required_cuts = 0
    correct_required_cuts = 0

    total_forbidden_cuts = 0
    correct_forbidden_cuts = 0

    total_predicted_cuts = 0
    correct_predicted_cuts = 0

    total_overlap_cases = 0
    correct_overlap_cases = 0

    for result in results:
        predicted = result["predicted_cuts"]
        must_cut = result["expected_must_cut"]
        must_not_cut = result["expected_must_not_cut"]

        # =========================
        # 1. Must-cut Recall
        # =========================
        total_required_cuts += len(must_cut)

        for pos in must_cut:
            if pos in predicted:
                correct_required_cuts += 1

        # =========================
        # 2. Must-not-cut Accuracy
        # =========================
        total_forbidden_cuts += len(must_not_cut)

        for pos in must_not_cut:
            if pos not in predicted:
                correct_forbidden_cuts += 1

        # =========================
        # 3. Boundary Precision
        # =========================

        has_boundary_annotation = (
            len(must_cut) > 0
            or len(must_not_cut) > 0
        )

        if has_boundary_annotation:
            total_predicted_cuts += len(predicted)

            for pos in predicted:
                if pos in must_cut:
                    correct_predicted_cuts += 1

        # =========================
        # 4. Overlap Accuracy
        # =========================

        expected_overlap = result["expected_overlap"]

        if expected_overlap is not None:
            total_overlap_cases += 1

            if result["overlap_pass"]:
                correct_overlap_cases += 1

    # =========================
    # Calculate metrics
    # =========================

    must_cut_recall = (
        correct_required_cuts / total_required_cuts
        if total_required_cuts > 0
        else 1.0
    )

    must_not_cut_accuracy = (
        correct_forbidden_cuts / total_forbidden_cuts
        if total_forbidden_cuts > 0
        else 1.0
    )

    boundary_precision = (
        correct_predicted_cuts / total_predicted_cuts
        if total_predicted_cuts > 0
        else 1.0
    )

    overlap_accuracy = (
        correct_overlap_cases / total_overlap_cases
        if total_overlap_cases > 0
        else 1.0
    )

    return {
        "must_cut_recall": must_cut_recall,
        "must_not_cut_accuracy": must_not_cut_accuracy,
        "boundary_precision": boundary_precision,
        "overlap_accuracy": overlap_accuracy,

        "correct_required_cuts": correct_required_cuts,
        "total_required_cuts": total_required_cuts,

        "correct_forbidden_cuts": correct_forbidden_cuts,
        "total_forbidden_cuts": total_forbidden_cuts,

        "correct_predicted_cuts": correct_predicted_cuts,
        "total_predicted_cuts": total_predicted_cuts,

        "correct_overlap_cases": correct_overlap_cases,
        "total_overlap_cases": total_overlap_cases
    }

def extract_overlap_counts(chunks: list[str]) -> list[int]:
    if not chunks or len(chunks) < 2:
        return []

    overlap_counts = []

    for i in range(len(chunks) - 1):
        left_sentences = split_sentences(chunks[i])
        right_sentences = split_sentences(chunks[i + 1])

        overlap_count = 0

        max_overlap = min(
            2,
            len(left_sentences),
            len(right_sentences)
        )

        for count in range(max_overlap, 0, -1):
            if left_sentences[-count:] == right_sentences[:count]:
                overlap_count = count
                break

        overlap_counts.append(overlap_count)

    return overlap_counts


if __name__ == "__main__":
    TEST_CASE_PATH = ROOT_DIR / "data" / "test_case.json"

    E2E_CASES = {
        "case_004",
        "case_005",
        "case_006",
        "case_007",
        "case_008",
        "case_009",
        "case_010",
        "case_011",
        "case_012",
        "case_013"
    }

    cases = load_test_cases(TEST_CASE_PATH)

    results = []

    print("===== CHUNKING EVALUATION =====\n")

    for case in cases:
        if case["id"] not in E2E_CASES:
            continue

        result = evaluate_case(case)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"

        print(f"{result['id']} - {result['name']}")
        print(f"Status: {status}")

        print(
            "Predicted cuts:",
            result["predicted_cuts"]
        )

        print(
            "Expected must cut:",
            result["expected_must_cut"]
        )

        print(
            "Expected must not cut:",
            result["expected_must_not_cut"]
        )

        print(
            "Min cuts:",
            result["min_cuts"]
        )

        print(
            "Min cuts pass:",
            result["min_cuts_pass"]
        )

        print(
            "Predicted overlaps:",
            result["predicted_overlaps"]
        )

        print(
            "Expected overlap:",
            result["expected_overlap"]
        )

        print(
            "Overlap pass:",
            result["overlap_pass"]
        )

        print(
            "Must cut pass:",
            result["must_cut_pass"]
        )

        print(
            "Must not cut pass:",
            result["must_not_cut_pass"]
        )

        print("-" * 60)

    # ==============================
    # CASE-LEVEL SUMMARY
    # ==============================

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["passed"]
    )

    failed = total - passed

    accuracy = (
        passed / total
        if total > 0
        else 0.0
    )

    print("\n===== SUMMARY =====")
    print("Total cases:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # ==============================
    # BOUNDARY METRICS
    # ==============================

    metrics = calculate_metrics(results)

    print("\n===== BOUNDARY METRICS =====")

    print(
        "Must-cut Recall:",
        f"{metrics['must_cut_recall'] * 100:.2f}%"
    )

    print(
        "Must-not-cut Accuracy:",
        f"{metrics['must_not_cut_accuracy'] * 100:.2f}%"
    )

    print(
        "Boundary Precision:",
        f"{metrics['boundary_precision'] * 100:.2f}%"
    )

    print(
        "Required cuts:",
        f"{metrics['correct_required_cuts']}/"
        f"{metrics['total_required_cuts']}"
    )

    print(
        "Forbidden cuts preserved:",
        f"{metrics['correct_forbidden_cuts']}/"
        f"{metrics['total_forbidden_cuts']}"
    )

    print(
        "Correct predicted cuts:",
        f"{metrics['correct_predicted_cuts']}/"
        f"{metrics['total_predicted_cuts']}"
    )
    print(
        "Overlap Accuracy:",
        f"{metrics['overlap_accuracy'] * 100:.2f}%"
    )

    print(
        "Correct overlap cases:",
        f"{metrics['correct_overlap_cases']}/"
        f"{metrics['total_overlap_cases']}"
    )

