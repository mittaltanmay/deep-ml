import numpy as np

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the average hinge loss for SVM classification.
    
    Args:
        y_true: Array of true labels (-1 or +1)
        y_pred: Array of predicted scores (raw SVM scores)
    
    Returns:
        Average hinge loss rounded to 4 decimal places
    """
    hinge=0.0
    for i in range(len(y_true)):
        hinge+=max(0,1-y_true[i]*y_pred[i])
    return hinge/len(y_pred)