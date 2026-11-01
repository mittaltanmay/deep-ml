import torch

def adamw_update(w: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor, t: int, lr: float, beta1: float, beta2: float, epsilon: float, weight_decay: float) -> tuple:
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter tensor (torch.Tensor)
      g: gradient tensor (torch.Tensor)
      m: first moment tensor (torch.Tensor)
      v: second moment tensor (torch.Tensor)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    # grad = torch.as_tensor(grad, dtype=torch.float64)
    # m = torch.as_tensor(m, dtype=torch.float64)
    # v = torch.as_tensor(v, dtype=torch.float64)
    m=beta1*m+(1-beta1)*g
    v=beta2*v+(1-beta2)*torch.square(g)
    m_t=m/(1-beta1**t)
    v_t=v/(1-beta2**t)
    w-=lr*weight_decay*w
    w-=lr*(m_t/(torch.sqrt(v_t)+epsilon))
    return w,m,v
    