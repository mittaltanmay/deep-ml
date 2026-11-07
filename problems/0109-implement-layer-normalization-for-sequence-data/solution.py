import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    mean=X.mean(dim=2,keepdim=True)
    var=X.var(dim=2,keepdim=True,unbiased=False)
    gamma=gamma.view(1,1,-1)
    beta=beta.view(1,1,-1)
    X_norm=(X-mean)/torch.sqrt(var+epsilon)
    return X_norm*gamma+beta
    