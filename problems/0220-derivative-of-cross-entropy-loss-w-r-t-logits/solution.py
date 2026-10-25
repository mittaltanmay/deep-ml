import torch
import torch.nn.functional as F

def cross_entropy_derivative(logits: torch.Tensor, target: int) -> torch.Tensor:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs tensor
        target: Index of the true class
        
    Returns:
        Gradient tensor
    """
    p=F.softmax(logits,dim=0)
    y=torch.zeros((logits.shape[0],),dtype=torch.float32)
    y[target]=1.0
    return p-y