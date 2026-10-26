from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim



model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

@lru_cache(maxsize=512)
def _encode_sentence(sentence: str):
    """Reuse sentence embeddings with a bounded cache for the fixed model."""
    return model.encode(sentence, convert_to_tensor=True)


def calculate_semantic_similarity(left_sentence, right_sentence):
    if not left_sentence or not right_sentence:
       return 0.0
    
    left_embedding = _encode_sentence(left_sentence)
    right_embedding = _encode_sentence(right_sentence)

    similarity_score = cos_sim(left_embedding, right_embedding).item()

    return similarity_score

# def calculate_semantic_similarity_drop(
#     prev_sentence: str,
#     left_sentence: str,
#     right_sentence: str,
#     next_sentence:str
# ) -> float:
#     sim_before = calculate_semantic_similarity(prev_sentence, left_sentence)
#     sim_cross = calculate_semantic_similarity(left_sentence, right_sentence)
#     sim_after = calculate_semantic_similarity(right_sentence, next_sentence)

#     semantic_drop = (sim_before + sim_after) / 2 - sim_cross

#     return round(semantic_drop, 2)
def calculate_semantic_similarity_drop(s1, s2, s3, s4):
    sim_before = calculate_semantic_similarity(s1, s2)
    sim_cross = calculate_semantic_similarity(s2, s3)
    sim_after = calculate_semantic_similarity(s3, s4)


    semantic_drop = (
        (sim_before + sim_after) / 2
        - sim_cross
    )

    return semantic_drop

