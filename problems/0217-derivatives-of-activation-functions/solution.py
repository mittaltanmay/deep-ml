import torch

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here - use autograd!
    # Hint: Create tensors with requires_grad=True, apply activation, call .backward()
    tensor=torch.tensor([x],dtype=torch.float32,requires_grad=True)
    sig = torch.sigmoid(tensor)
    sig.backward()
    d_sig = tensor.grad.item()
    
    # 2. Reset gradient and compute Tanh derivative
    tensor.grad.zero_()
    tanh_val = torch.tanh(tensor)
    tanh_val.backward()
    d_tanh = tensor.grad.item()
    
    # 3. Reset gradient and compute ReLU derivative
    tensor.grad.zero_()
    relu_val = torch.relu(tensor)
    relu_val.backward()
    d_relu = tensor.grad.item()
    return {"sigmoid":d_sig,"tanh":d_tanh,"relu":d_relu}
