import torch
import math
def entropy_and_cross_entropy(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution (torch.Tensor)
        Q: Predicted probability distribution (torch.Tensor)
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    log_p=torch.log(P)
    log_q=torch.log(Q)
    for i in range(P.shape[0]):
        log_p[i]*=P[i]
        log_q[i]*=P[i]
    # for i in range(q.shape[1]):
    entropy=torch.sum(log_p).item()
    cross_entropy=torch.sum(log_q).item()
    if math.isnan(entropy):
        entropy=0.0
    return (-1*entropy,-1*cross_entropy)