import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """
    n=len(token_ids)
    v=W.shape[0]
    H=np.zeros((n,v),dtype=np.float32)
    for ind,ids in enumerate(token_ids):
        H[ind][ids]=1.0
    E=H @ W
    return E
