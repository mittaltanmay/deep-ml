import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        # Your code here
        self.p=p
        if(p<0 or p>=1):
            raise ValueError("valueError")
        self.mask=None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        # Your code here
        if(training==True):
            self.mask=np.random.binomial(1,1-self.p,x.shape)
            return np.array(x * self.mask/(1-self.p))
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        if self.mask is None:
            raise ValueError("ValueError")
        return np.array(grad * self.mask/(1-self.p))
        
        