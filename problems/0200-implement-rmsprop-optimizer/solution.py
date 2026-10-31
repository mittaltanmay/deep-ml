import torch

def rmsprop_update(params: torch.Tensor, grads: torch.Tensor, cache: torch.Tensor,
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform RMSProp optimization update.
    
    Args:
        params: Tensor of parameter values
        grads: Tensor of gradients for each parameter
        cache: Tensor of cache values (moving average of squared gradients)
        lr: Learning rate
        beta: Decay rate for moving average
        epsilon: Small constant for numerical stability
    
    Returns:
        Tuple of (updated_params, updated_cache)
    """
    v_t=beta*cache+(1-beta)*(torch.square(grads))
    p_t=params-((lr)/(torch.sqrt(v_t)+epsilon))*grads
    return (p_t,v_t)