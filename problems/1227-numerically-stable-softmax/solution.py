import torch

def softmax(t, dim):
    max_val=torch.max(t,dim=dim,keepdim=True).values
    t=t-max_val
    exp=torch.exp(t)
    Sum=torch.sum(exp,dim=dim,keepdim=True)
    softmax=exp/Sum
    return softmax

                
