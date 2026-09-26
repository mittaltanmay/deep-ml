import torch

def pca_reconstruction_error(X: torch.Tensor, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features) as a torch.Tensor
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    Data=X-X.mean(dim=0)
    covariance=torch.cov(Data.T,correction=0)
    eigv,eign=torch.linalg.eigh(covariance)
    ans=0.0
    rem=len(eigv)-n_components
    for i in range(rem):
        ans+=eigv[i].item()
    return ans/len(eigv)
