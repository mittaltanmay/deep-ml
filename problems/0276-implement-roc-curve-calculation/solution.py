import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    x=[0.0]
    y=[0.0]
    pos=0
    neg=0
    for i in range(len(y_true)):
        if y_true[i]==1:
            pos+=1
        else:
            neg+=1
    sorted_scores=list(set(y_scores))
    sorted_scores.sort(reverse=True)
    for i in range(len(sorted_scores)):
        thresh=sorted_scores[i]
        pred=[1 if score>=thresh else 0 for score in y_scores]
        tp=0
        fp=0
        for j in range(len(y_scores)):
            if y_true[j]==1 and pred[j]==1:
                tp+=1
            elif y_true[j]==0  and pred[j]==1:
                fp+=1
        x.append((fp/neg)*1.0)
        y.append((tp/pos)*1.0)
    return (x,y)
