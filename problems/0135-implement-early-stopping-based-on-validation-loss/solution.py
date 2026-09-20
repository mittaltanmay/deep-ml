import torch
from typing import Tuple

def early_stopping(val_losses: torch.Tensor, patience: int, min_delta: float) -> Tuple[int, int]:
    """
    Determine when to stop training early based on validation losses.
    
    Args:
        val_losses: A 1D tensor of validation losses for each epoch
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum decrease in loss to qualify as an improvement
    
    Returns:
        Tuple of (stop_epoch, best_epoch)
    """
    p=patience
    best_i=0
    best_loss=val_losses[0];
    for i in range(1,len(val_losses)):
        if val_losses[i]<best_loss-min_delta:
            best_i=i
            best_loss=val_losses[i]
            p=patience
        else:
            p-=1
        if p==0:
            return (i,best_i)

    return (len(val_losses)-1,best_i)   
    