import torch
import torch.nn.functional as F
def softmax_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix of the softmax function using PyTorch.
    
    Args:
        x: Input tensor
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here - you can use torch.autograd.functional.jacobian
    # or compute it directly from softmax output
    softmax=F.softmax(x,dim=-1)
    n=x.shape[0]
    derivative=torch.zeros((n,n),dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            if i==j:
                derivative[i][j]=softmax[i]*(1-softmax[j])
            else:
                derivative[i][j]=-1*softmax[i]*(softmax[j])
    return derivative