import torch
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    """
    Given a 2D tensor X and integer degree, return a new tensor of all polynomial
    feature combinations (with constant term), sorted for each sample from
    smallest to largest.
    """
    X = torch.as_tensor(X)

    n_samples, n_features = X.shape
    features = []

    features.append(torch.ones(n_samples, dtype=torch.float32))

    for d in range(1, degree + 1):
        for combination in combinations_with_replacement(range(n_features), d):
            selected = X[:, combination]
            feature = torch.prod(selected, dim=1)
            features.append(feature)

    result = torch.stack(features, dim=1)
    result, _ = torch.sort(result, dim=1)

    return result