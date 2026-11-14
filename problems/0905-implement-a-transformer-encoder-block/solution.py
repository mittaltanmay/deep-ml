import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: norm1, attn, norm2, mlp, dropout
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.attn=nn.MultiheadAttention(d_model,num_heads,dropout=dropout,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Linear(d_ff,d_model)
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        # TODO: pre-LN attention sublayer, then pre-LN MLP sublayer, both with residual
        norm_x=self.norm1(x)
        attn_value,attn_weights=self.attn(norm_x,norm_x,norm_x)
        dropout_attn=self.dropout(attn_value)

        residual1=dropout_attn+x

        norm_residual=self.norm2(residual1)
        ffn=self.mlp(norm_residual)
        drop_ffn=self.dropout(ffn)

        return residual1+drop_ffn


