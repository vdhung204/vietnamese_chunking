from boundary import CandidateBoundary, create_candidate_boundaries
from dynamic_overlap import get_overlap_sentences
from sentence_splitter import split_sentences
from boundary_scorer import MAX_TOKENS, MIN_TOKENS, score_boundaries, calculate_boundary_semantic_drop

SEMANTIC_DROP_THRESHOLD = 0.30

def select_best_boundary(candidates: list[CandidateBoundary]) -> CandidateBoundary | None:
    """
    Select the best boundary from a list of candidates based on their scores.

    Args:
        candidates (list): A list of candidate boundaries, each represented as a tuple
        containing the boundary and its score.
    Returns:
        CandidateBoundary: The best boundary and its score.
    """
    if not candidates:
        return None

    scores = score_boundaries(candidates)

    best_score = max(scores)
    best_index = scores.index(best_score)

    if best_score == float("-inf"):
        return None
    return candidates[best_index]


def split_at_boundary(sentences: list[str], boundary: CandidateBoundary) -> tuple[list[str], list[str]]:
    """
    Split a list of sentences at the specified boundary.

    Args:
        sentences (list): A list of sentences to be split.
        boundary (CandidateBoundary): The boundary at which to split the sentences.

    Returns:
        tuple: A tuple containing two lists of sentences, one before the boundary and one after.
    """
    if not sentences or not boundary:
        return [], []

    chunk_sentences = sentences[:boundary.position + 1]
    remaining_sentences = sentences[boundary.position + 1:]

    return chunk_sentences, remaining_sentences

def create_first_chunk(text: str) -> tuple[str, str]:
    sentences = split_sentences(text)

    if not sentences:
        return "", ""

    total_tokens = sum(
        len(sentence.split())
        for sentence in sentences
    )

    if total_tokens < MIN_TOKENS:
        return " ".join(sentences), ""

    boundaries = create_candidate_boundaries(sentences)

    # Case 2: optional cut
    if total_tokens <= MAX_TOKENS:
        best_semantic_boundary = select_strong_semantic_boundary(
            boundaries
        )

        # nếu không có boundary đủ context
        if best_semantic_boundary is None:
            return " ".join(sentences), ""

        semantic_drop = calculate_boundary_semantic_drop(
            best_semantic_boundary
        )

        if semantic_drop >= SEMANTIC_DROP_THRESHOLD:
            chunk_sentences, remaining_sentences = split_at_boundary(
                sentences,
                best_semantic_boundary
            )

            return (
                " ".join(chunk_sentences),
                " ".join(remaining_sentences)
            )

        return " ".join(sentences), ""
        
    best_boundary = select_best_boundary(boundaries)

    if best_boundary is None:
        return " ".join(sentences), ""

    chunk_sentences, remaining_sentences = split_at_boundary(
        sentences,
        best_boundary
    )

    return (
        " ".join(chunk_sentences),
        " ".join(remaining_sentences)
    )


def semantic_chunk(text: str) -> list[str]:
    chunks = []
    remaining_text = text
    pending_overlap = []

    while remaining_text:
        chunk, new_remaining_text = create_first_chunk(
            remaining_text
        )

        if not chunk:
            break

        chunk_sentences = split_sentences(chunk)

        remaining_sentences = split_sentences(
            new_remaining_text
        )

        # overlap từ chunk trước + chunk hiện tại
        output_sentences = pending_overlap + chunk_sentences

        output_chunk = " ".join(output_sentences)

        chunks.append(output_chunk)

        # chuẩn bị overlap cho chunk TIẾP THEO
        pending_overlap = get_overlap_sentences(
            chunk_sentences,
            remaining_sentences
        )

        # phần chưa xử lý thật sự
        remaining_text = new_remaining_text

    return chunks


def select_strong_semantic_boundary(
    boundaries: list[CandidateBoundary]
) -> CandidateBoundary | None:

    if not boundaries:
        return None

    best_boundary = None
    best_drop = float("-inf")

    for boundary in boundaries:

        # Không cho optional semantic cut quá sớm
        if boundary.token_count < MIN_TOKENS:
            continue

        semantic_drop = calculate_boundary_semantic_drop(
            boundary
        )

        if semantic_drop is None:
            continue

        if semantic_drop > best_drop:
            best_drop = semantic_drop
            best_boundary = boundary

    return best_boundary

