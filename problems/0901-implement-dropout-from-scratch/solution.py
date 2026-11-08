import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    if training==False:
        return x
    else:
        mask=torch.empty_like(x)
        mask.bernoulli_(p=1-p)
        x=x*(mask/(1-p))
        return x
