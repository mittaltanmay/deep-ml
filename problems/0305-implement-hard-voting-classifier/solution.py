import torch

def hard_voting_classifier(predictions: torch.Tensor) -> torch.Tensor:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D tensor where predictions[i][j] is classifier i's prediction for sample j
                     Shape: (n_classifiers, n_samples)
        
    Returns:
        1D tensor of final predictions using majority vote
    """
    count={}
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            count[predictions[i][j]]=count.get(predictions[i][j],0)+1
    maxi=0
    pred=-1
    for key,value in count.items():
        if value>maxi:
            pred=key
            value=maxi
        elif value==maxi:
            maxi=min(maxi,key)
    return ans