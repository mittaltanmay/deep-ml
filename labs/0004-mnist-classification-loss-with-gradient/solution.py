import numpy as np
import math
def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    n,m=preds.shape
    target_logits=[]
    loss=[]
    for i in range(n):
        temp=[0]*m
        temp[target[i]]=1.0
        target_logits.append(temp)
        loss.append(-1*np.log((preds[i][target[i]]).item()))
    grad=preds-target_logits
    Loss=np.array(loss,dtype=float)
    if reduction=="mean":
        result=np.mean(Loss).item()
    elif reduction=="sum":
        result=np.sum(Loss).item()
    else:
        return (loss,grad)
    return (result,grad)
