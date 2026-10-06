import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
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
    area=0.0
    for i in range(1,len(x)):
        area+=((y[i]+y[i-1])*(x[i]-x[i-1]))
    return area/2