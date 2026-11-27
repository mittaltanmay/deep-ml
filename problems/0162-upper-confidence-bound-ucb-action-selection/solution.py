import torch

def ucb_action(counts: torch.Tensor, values: torch.Tensor, t: int, c: float) -> int:
    """
    Choose an action using the UCB1 formula.
    Args:
      counts (torch.Tensor): Number of times each action has been chosen
      values (torch.Tensor): Average reward of each action
      t (int): Current timestep (starts from 1)
      c (float): Exploration coefficient
    Returns:
      int: Index of action to select
    """
    n=counts.shape[0]
    ln=torch.tensor([t],dtype=torch.float32)
    ucb=torch.zeros((n,),dtype=torch.float32)

    for i in range(n):
      ucb[i]=values[i]+c*(torch.sqrt(torch.log(ln)/(1e-8+counts[i]))).item()
    return (torch.argmax(ucb)).item()
    