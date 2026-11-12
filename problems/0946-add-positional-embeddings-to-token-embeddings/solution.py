import numpy as np

def add_positional_embeddings(token_embeddings: np.ndarray, pos_embedding_matrix: np.ndarray) -> np.ndarray:
    """Add absolute positional embeddings to token embeddings."""
    seq_len=token_embeddings.shape[1]
    postion_embeddings=pos_embedding_matrix[0:seq_len,:]
    for i in range(token_embeddings.shape[0]):
        token_embeddings[i]+=postion_embeddings
    return token_embeddings
        
