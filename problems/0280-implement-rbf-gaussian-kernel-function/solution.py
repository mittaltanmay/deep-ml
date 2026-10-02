import torch
import math
def rbf_kernel(X1: torch.Tensor, X2: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute the RBF (Gaussian) kernel matrix between X1 and X2.
    
    Args:
        X1: First set of samples with shape (n1, d)
        X2: Second set of samples with shape (n2, d)
        gamma: Kernel coefficient (controls kernel width)
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """
    n=X1.shape[0]
    m=X2.shape[0]
    ans=[]
    for i in range(n):
        temp=[]
        for j in range(m):
            dist=X1[i]-X2[j]
            dist=torch.square(dist)
            d=torch.sum(dist)
            temp.append(math.exp(-1*gamma*d.item()))
        ans.append(temp)
    return torch.tensor(ans)
    
