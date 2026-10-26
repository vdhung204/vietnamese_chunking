

DEPENDENCY_PAIRS = [
    ("nếu", "thì"),
    ("vì", "nên"),
    ("do", "nên"),
    ("tuy", "nhưng"),
    ("mặc dù", "nhưng"),
    ("không những", "mà còn"),
    ("không phải", "mà"),
    ("hễ", "thì")
]
EXCEPTION_PHRASES = [
    "vì vậy",
    "do đó",
]

def dependency_risk(left_text:str, right_text:str) -> float:
    left = left_text.lower()
    right =right_text.lower()

    for first, second in DEPENDENCY_PAIRS:

        if first == "vì" and "vì vậy" in left:
            continue

        if first == "do" and "do đó" in left:
            continue

        if first in left and second in right:
            return 1.0

    return 0.0


