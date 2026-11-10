import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    position_embedding=torch.zeros(seq_len,d_model,dtype=torch.float32)
    for i in range(seq_len):
        for j in range(d_model):
            div_term=math.exp(-math.log(10000)*((2*(j//2))/d_model))
            if j%2==0:
                position_embedding[i][j]=math.sin(i*div_term)
            else:
                position_embedding[i][j]=math.cos(i*div_term)
    return position_embedding

