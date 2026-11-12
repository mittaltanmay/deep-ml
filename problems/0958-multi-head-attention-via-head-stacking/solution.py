import numpy as np
def softmax(x, axis=-1):
  x_shifted = x - np.max(x, axis=axis, keepdims=True)
  exp_x = np.exp(x_shifted)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_attention_stack(x, heads):
    """
    Multi-head causal self-attention by stacking single-head attention modules.

    Args:
        x: array-like of shape (B, T, d_in)
        heads: list of H tuples (W_q, W_k, W_v); each weight is shape (d_in, d_head)

    Returns:
        numpy.ndarray of shape (B, T, H * d_head)
    """
    # Your code here
    X=np.array(x,dtype=float)
    output=[]
    mask=np.zeros((X.shape[1],X.shape[1]),dtype=float)
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if(i-j<0):
                mask[i][j]=float('-inf')
    for i in range(len(heads)):
        w_q,w_k,w_v=heads[i]
        W_q=np.array(w_q,dtype=float)
        W_k=np.array(w_k,dtype=float)
        W_v=np.array(w_v,dtype=float)
        Q=X @ W_q
        K=X @ W_k
        V=X @ W_v
        attention=Q @ K.transpose(0,2,1)
        attention/=W_q.shape[1]**0.5
        attention+=mask
        attention_weights=softmax(attention,-1)
        output.append(attention_weights @ V)
    return np.concatenate(output,axis=-1)


