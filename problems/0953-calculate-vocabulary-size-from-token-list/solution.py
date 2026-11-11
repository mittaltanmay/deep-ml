def vocab_size(tokens, special_tokens=None):
    """Return the vocabulary size from a list of preprocessed tokens.

    Args:
        tokens: list[str] of preprocessed tokens
        special_tokens: optional list[str] of special tokens to include
    Returns:
        int: number of unique tokens in the combined vocabulary
    """
    unique=set(tokens)
    count=(int)(len(unique))+(0 if special_tokens is None else (int)(len(special_tokens)))
    if special_tokens is not None:
        for i in range(len(tokens)):
            if tokens[i] in special_tokens:
                count-=1
    return count