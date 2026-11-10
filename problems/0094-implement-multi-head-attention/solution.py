import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q=X @ W_q
    K=X @ W_k
    V=X @ W_v
    return Q,K,V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    attentions=Q @ K.T
    attentions/=(K.shape[1])**0.5
    attentions_score=torch.softmax(attentions,dim=-1)
    return attentions_score @ V

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    output=[]
    splits=Q.shape[1]//n_heads
    Q_splits=torch.split(Q,splits,dim=1)
    K_splits=torch.split(K,splits,dim=1)
    V_splits=torch.split(V,splits,dim=1)
    for Q_i,K_i,V_i in zip(Q_splits,K_splits,V_splits):
        output.append(self_attention(Q_i,K_i,V_i))
    return torch.cat(output,dim=1)
    