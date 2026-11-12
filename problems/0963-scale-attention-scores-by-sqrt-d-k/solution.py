import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    def softmax(x, axis=-1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    attention=Q @ K.T
    d_k=Q.shape[1]
    attention_Scores=softmax(attention/(d_k)**0.5,-1)
    return (attention_Scores).tolist()
