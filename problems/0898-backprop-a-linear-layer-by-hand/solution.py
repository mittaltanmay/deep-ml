import torch

def linear_backward(grad_output, x, W):
    # TODO: return (grad_input, grad_W, grad_b) for y = x @ W.T + b
    return (grad_output @ W,grad_output.T @ x, grad_output.sum(dim=0))
