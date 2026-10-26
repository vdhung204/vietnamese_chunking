from boundary import CandidateBoundary
from dependency_rules import dependency_risk
from semantic_similarity import calculate_semantic_similarity_drop

MIN_TOKENS = 120
TARGET_TOKENS = 180
MAX_TOKENS= 240

def calculate_size_score(token_count):
    if token_count < MIN_TOKENS or token_count > MAX_TOKENS:
        return 0.0

    if token_count <= TARGET_TOKENS:
        score = (
            (token_count - MIN_TOKENS)
            / (TARGET_TOKENS - MIN_TOKENS)
        )
    else:
        score = (
            (MAX_TOKENS - token_count)
            / (MAX_TOKENS - TARGET_TOKENS)
        )

    return round(score, 2)




def calculate_boundary_semantic_drop(boundary: CandidateBoundary) -> float | None:
    if boundary.prev_sentence is None or boundary.next_sentence is None:
        return None
    else:
        semantic_drop = calculate_semantic_similarity_drop(
            boundary.prev_sentence,
            boundary.left_sentence,
            boundary.right_sentence,
            boundary.next_sentence
        )

    return semantic_drop

def normalize_semantic_drops(semantic_drops: list[CandidateBoundary]) -> list[float]:
    if not semantic_drops:
        return []
    
    drop_values = [calculate_boundary_semantic_drop(boundary) for boundary in semantic_drops]

    valid_drops = [drop for drop in drop_values 
                   if drop is not None]

    if not valid_drops:
        return [0.0 for _ in drop_values]
    
    min_drop = min(valid_drops)
    max_drop = max(valid_drops)

    if min_drop == max_drop:
        return [0.0 for _ in drop_values]

    normalized_drops = []

    for drop in drop_values:
        if drop is None:
            normalized_drops.append(0.0)
        else:
            normalized_drop = (drop - min_drop) / (max_drop - min_drop)
            normalized_drops.append(normalized_drop)

    return normalized_drops

def score_boundaries(boundaries: list[CandidateBoundary]) -> list[float]:

    valid_indices = [
        i 
        for i, boundary in enumerate(boundaries)
        if MIN_TOKENS <= boundary.token_count <= MAX_TOKENS
    ]
    if not valid_indices:
        return [float('-inf') for _ in boundaries]

    
    valid_boundaries = [
        boundaries[i]
        for i in valid_indices
    ]

    semantic_scores = normalize_semantic_drops(valid_boundaries)
    scores = [
        float("-inf")
        for _ in boundaries
    ]
    for semantic_index, boundary_index in enumerate(valid_indices):

        boundary = boundaries[boundary_index]

        size_score = calculate_size_score(
            boundary.token_count
        )

        dependency_score = dependency_risk(
            boundary.left_sentence,
            boundary.right_sentence
        )

        semantic_score = semantic_scores[semantic_index]

        final_score = (
            0.4 * size_score
            + 0.4 * semantic_score
            - 0.6 * dependency_score
        )

        scores[boundary_index] = final_score

    return scores

    