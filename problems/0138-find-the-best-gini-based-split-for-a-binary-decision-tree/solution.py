import torch
from typing import Tuple
def gini(x:torch.tensor):
    count={}
    for i in range(len(x)):
        count[x[i].item()]=count.get(x[i].item(),0)+1
    gini_impurity=0
    for key,value in count.items():
        gini_impurity+=(value/(len(x)))**2
    return 1-gini_impurity

def find_best_split(X: torch.Tensor, y: torch.Tensor) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    n,m=X.shape
    best_split=-1
    best_gini=float("inf")
    best_feature=0
    for feature in range(m):
        for threshold in X[:, feature]:
            left=[]
            right=[]
            for j in range(n):
                if X[j][feature]<=threshold:
                    left.append(y[j].item())
                else:
                    right.append(y[j].item())
            left_side=torch.tensor(left)
            right_side=torch.tensor(right)
            y_left=gini(left_side)
            y_right=gini(right_side)
            weighted=(len(left)/n)*y_left+(len(right)/n)*y_right
            if weighted<best_gini:
                best_gini=weighted
                best_split=threshold
                best_feature=feature
    return (best_feature,best_split.item())