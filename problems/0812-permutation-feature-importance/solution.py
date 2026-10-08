import numpy as np
def pred(X,weights,bias):
    return [
        sum(X[i][j]*weights[j] for j in range(len(weights)))+bias
        for i in range(len(X))
    ]

def permutation_importance(X, y, weights, bias, permutations):
    """
    Compute permutation feature importance for a linear regression model.

    Args:
        X: 2D array-like of shape (n_samples, n_features)
        y: 1D array-like of shape (n_samples,)
        weights: 1D array-like of shape (n_features,)
        bias: float, the intercept
        permutations: list of length n_features; permutations[j] is a list of
                      permutation index arrays to apply to column j

    Returns:
        List of feature importances (length n_features).
    """
    pred_real=pred(X,weights,bias)
    y_mean=sum(y)/len(y)
    ss_r=sum((y[i]-pred_real[i])**2 for i in range(len(y)))
    ss_tot=sum((y[i]-y_mean)**2 for i in range(len(y)))
    r2_real=1-(ss_r/ss_tot)
    importance=[]
    for j in range(len(X[0])):
        scores=[]
        for perm in permutations[j]:
            x_perm=[row[:] for row in X]
            for i in range(len(X)):
                x_perm[i][j]=X[perm[i]][j]
            pred_perm=pred(x_perm,weights,bias)
            ss_r=sum((y[i]-pred_perm[i])**2 for i in range(len(y)))
            scores.append(1-(ss_r/ss_tot))
        scores_mean=sum(scores)/len(scores)
        importance.append(r2_real-scores_mean)
    return importance
