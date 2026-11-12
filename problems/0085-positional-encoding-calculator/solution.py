import torch

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    # Your code here
    if position==0 or d_model<=0:
        return -1

    output=[]
    for i in range(position):
        temp=[]
        for j in range(d_model//2):
            den=(10000)**(2*j/d_model)
            angle=i/den
            sin=(torch.sin(torch.tensor([angle],dtype=torch.float16))).item()
            cos=(torch.cos(torch.tensor([angle],dtype=torch.float16))).item()
            temp.append(sin)
            temp.append(cos)
        output.append(temp)
    return torch.tensor(output,dtype=torch.float16)