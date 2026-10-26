
from semantic_similarity import calculate_semantic_similarity


LOW_SIMILARITY_THRESHOLD = 0.30
HIGH_SIMILARITY_THRESHOLD = 0.55

def determine_overlap_count(
    left_sentence: str,
    right_sentence: str
) -> int:
    similarity_score = calculate_semantic_similarity(left_sentence, right_sentence)

    if similarity_score < LOW_SIMILARITY_THRESHOLD:
        return 0
    elif similarity_score < HIGH_SIMILARITY_THRESHOLD:
        return 1
    else:
        return 2



def get_overlap_sentences(
    chunk_sentences: list[str],
    remaining_sentences: list[str]
) -> list[str]:
    
    if not chunk_sentences or not remaining_sentences:
        return []

    left_sentence = chunk_sentences[-1]
    right_sentence = remaining_sentences[0]

    overlap_count = determine_overlap_count(
        left_sentence,
        right_sentence
    )

    if overlap_count == 0:
        return []

    else: 
        overlap_sentences = chunk_sentences[-overlap_count:]
    
    return overlap_sentences


