import torch

def gini_impurity(y: torch.Tensor) -> float:
    """
    Calculate Gini Impurity for a tensor of class labels.

    :param y: 1D Tensor of class labels (integer type)
    :return: Gini Impurity rounded to three decimal places
    """
    count={}
    for i in range(len(y)):
        count[y[i].item()]=count.get(y[i].item(),0)+1
    gini=0
    for key,value in count.items():
        gini=gini+(value/len(y))**2
    return 1-gini