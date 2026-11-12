import numpy as np

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    X_np=np.array(X,dtype=float)
    context=X_np @ X_np.T
    contex_shifted = context - np.max(context,axis=-1, keepdims=True)
    exp_x = np.exp(contex_shifted)
    context_values=exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return (context_values @ X_np).tolist()
