import numpy as np

def attention_dropout(attn_weights, values, dropout_rate, mask):
    """Apply dropout mask to attention weights and compute context vectors."""
    n=len(attn_weights)
    m=len(attn_weights[0])
    for i in range(n):
        for j in range(m):
            if mask[i][j]==0:
                attn_weights[i][j]=0
            else:
                attn_weights[i][j]*=(1/(1-dropout_rate))
    weights=np.array(attn_weights,dtype=float)
    v=np.array(values,dtype=float)
    return (weights @ v).tolist()
