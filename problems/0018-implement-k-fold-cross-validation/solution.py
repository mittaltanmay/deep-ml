import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    data=np.arange(0,n_samples)
    if(shuffle):
        np.random.shuffle(data)
    ans=[]
    fold_size=n_samples//k
    for i in range(0,n_samples,fold_size):
        fold=data[i:i+fold_size]
        rem1=data[0:i]
        rem2=data[i+fold_size:n_samples]
        rem=np.concatenate((rem1,rem2))
        ans.append((rem.tolist(),fold.tolist()))
    return ans