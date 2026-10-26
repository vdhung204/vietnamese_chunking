
from dataclasses import dataclass


@dataclass
class CandidateBoundary:

    position: int
    token_count: int
    left_sentence: str
    right_sentence: str
    prev_sentence: str | None
    next_sentence: str | None

def create_candidate_boundaries(sentences: list[str]) -> list[CandidateBoundary]:
    """create candidate boundary

    Parameters
    ----------
    sentences : list[str]
        list of sentences to be split into two parts

    Returns
    -------
    list[CandidateBoundary]
        list of candidate boundary objects
    """
    if sentences is None or len(sentences) < 2:
        return []

    cumulative_tokens = 0
    boundaries = []

    for i in range(len(sentences) - 1):
        cumulative_tokens += len(sentences[i].split())
        prev_sentence = sentences[i - 1] if i > 0 else None
        left_sentence = sentences[i]
        right_sentence = sentences[i + 1]
        next_sentence = sentences[i + 2] if i < len(sentences) - 2 else None
        boundaries.append(
            CandidateBoundary(
                position=i,
                token_count=cumulative_tokens,
                prev_sentence=prev_sentence,
                left_sentence=left_sentence,
                right_sentence=right_sentence,
                next_sentence=next_sentence
            )
        )
    return boundaries

