import torch

def explained_variance_ratio(X: torch.Tensor) -> torch.Tensor:
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features) as a torch.Tensor
    
    Returns: 
        torch.Tensor of explained variance ratios sorted in descending order
    """
    # Your code here
    Data=X-X.mean(dim=0)
    covariance=torch.cov(Data.T)
    eigv,eign=torch.linalg.eigh(covariance)
    ans=[]
    Sum=(torch.sum(eigv)).item()
    for i in range(len(eigv)):
        temp=eigv[i].item()/Sum
        if temp<0:
            temp*=-1
        ans.append(temp)
    ans.sort(reverse=True)
    return torch.tensor(ans)