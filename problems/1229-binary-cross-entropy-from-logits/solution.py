import torch
import math
def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    loss=[]
    for i in range(logits.shape[0]): 
        temp=max(logits[i],0) - logits[i]*targets[i] + math.log(1+math.exp(-1*abs(logits[i])))
        loss.append(temp)
    Loss=torch.tensor(loss,dtype=torch.float32)
    return (torch.mean(Loss)).item()