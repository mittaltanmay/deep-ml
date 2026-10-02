import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    n,m=X_minority.shape
    k_actual=min(k,n-1)
    if n_synthetic==0:
        return np.empty((0,m))
    if k_actual==0:
        return np.empty((0,m))
    ans=[]
    for i in range(n_synthetic):
        ind=np.random.randint(0,n)
        x_i=X_minority[ind]
        neighbour=[]
        for j in range(n):
            if(j==ind):
                continue
            else:
                dist=(x_i[0]-X_minority[j][0])**2+(x_i[1]-X_minority[j][1])**2
                dist=(dist)**0.5
                neighbour.append([dist,j])
        neighbour.sort()
        j=np.random.randint(0,k_actual)
        x_nn=X_minority[neighbour[j][1]]
        gap=np.random.random()
        ans.append(x_i+gap*(x_nn-x_i))
    return np.array(ans)

