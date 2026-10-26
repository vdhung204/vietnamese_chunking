

def fixed_chunk(text, chunk_size = 200, overlap = 30):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise TypeError("overlap must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    tokens = text.split()

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size

        chunk = tokens[start:end]

        chunks.append(" ".join(chunk))

        if end >=  len(tokens):
            break

        start = end -overlap

    return chunks
