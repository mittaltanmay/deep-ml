import torch

def batch_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""
    mean=X.mean(dim=(0,2,3),keepdim=True)
    variance=X.var(dim=(0,2,3),keepdim=True,unbiased=False)
    X_norm=(X-mean)/torch.sqrt(variance+epsilon)
    gamma=gamma.view(1,-1,1,1)
    beta=beta.view(1,-1,1,1)
    out=gamma*X_norm+beta
    return out