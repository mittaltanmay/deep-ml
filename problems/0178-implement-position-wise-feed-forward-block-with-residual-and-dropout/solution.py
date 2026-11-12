import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNBlock(nn.Module):
    def __init__(self, d_model, d_hidden, dropout_p=0.1):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_hidden)
        self.linear2=nn.Linear(d_hidden,d_model)
        self.dropout_layer=nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        # Your code here
        residual=x
        out=self.linear1(x)
        out=F.relu(out)
        out=self.linear2(out)
        out=self.dropout_layer(out)
        return out+residual

        