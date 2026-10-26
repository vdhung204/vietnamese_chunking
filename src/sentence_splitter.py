import re

def split_sentences(text: str) -> list[str]:
    """split sentences   

    Parameters
    ----------
    text : str
        raw text to be split into sentences

    Returns
    -------
    list[str]
        list of sentences
    """
    if text is None :
        return []
    string = text.strip()
    if not string:
        return []

    return re.split(r"(?<=[.?!])\s+", string)


