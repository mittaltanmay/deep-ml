import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    X=np.array(x,dtype=np.float32)
    X_poly=np.column_stack([X**i for i in range(degree+1)])
    # W=np.zeros((degree+1),dtype=np.float32)
    A=(X_poly.T @ X_poly)
    B=X_poly.T @ y
    return  (np.linalg.solve(A,B)).tolist()   
