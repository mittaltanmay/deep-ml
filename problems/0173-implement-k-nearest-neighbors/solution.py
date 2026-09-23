import torch

def k_nearest_neighbors(points: torch.Tensor, query_point: torch.Tensor, k: int) -> torch.Tensor:
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: Tensor of shape (n, d) representing n points in d-dimensional space
        query_point: Tensor of shape (d,) representing the query point
        k: Number of nearest neighbors to return
    
    Returns:
        Tensor of shape (k, d) containing the k nearest neighbor points
        When distances are tied, points appearing earlier in the input tensor come first.
    """
    nearest=[]
    n,d=points.shape
    for i in range(n):
        dist=(torch.sqrt
        (torch.sum
        (torch.square(points[i]-query_point))
        )).item()
        temp=[]
        temp.append(dist)
        temp.append(i)
        for j in range(d):
            temp.append(points[i][j])
        nearest.append(temp)
    nearest.sort()
    ans=[]
    for i in range(n):
        if len(ans)==k:
            break;
        ans.append(nearest[i][2:])
    return torch.tensor(ans)