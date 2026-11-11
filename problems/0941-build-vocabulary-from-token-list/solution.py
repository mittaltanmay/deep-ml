def build_vocab(tokens):
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """
    tokens=sorted(set(tokens))
    ct=0
    token_id={}
    for i in range(len(tokens)):
        token_id[tokens[i]]=ct
        ct+=1
    return token_id
