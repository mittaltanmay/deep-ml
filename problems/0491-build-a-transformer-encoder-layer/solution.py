import numpy as np
def softmax(x, axis=-1):
    # Subtracting the max along the axis prevents overflow (numerical stability trick)
    kwarg = {"axis": axis, "keepdims": True}
    exps = np.exp(x - np.max(x, **kwarg))
    return exps / np.sum(exps, **kwarg)
def relu(x):
    return np.maximum(0, x)
def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    output=[]
    Q=X @ weights["W_q"]
    K=X @ weights["W_k"]
    V=X @ weights["W_v"]
    d_model=weights["W_q"].shape[1]
    d_head=d_model//num_heads
    for i in range(num_heads):
        Q_i=Q[:,:,i*d_head:(i+1)*d_head]
        K_i=K[:,:,i*d_head:(i+1)*d_head]
        V_i=V[:,:,i*d_head:(i+1)*d_head]

        attention=(Q_i @ K_i.transpose(0,2,1))/(d_head)**0.5
        score=softmax(attention,-1)
        output.append(score @ V_i)

    output_tensor=np.concatenate(output,axis=-1)
    output_tensor=output_tensor @ weights["W_o"]
    residual1=output_tensor+X

    mean=np.mean(residual1,axis=-1,keepdims=True)
    var=np.var(residual1,axis=-1,keepdims=True)
    residual1=(residual1-mean)/np.sqrt(var+eps)

    gamma1=weights["gamma1"].reshape(1,1,-1)
    beta1=weights["beta1"].reshape(1,1,-1)

    residual1=residual1*gamma1 + beta1

    ff_i=residual1
    hd1=ff_i @ weights["W1"] + weights["b1"].reshape(1,1,-1)
    hd1_reLU=relu(hd1)
    hd2=hd1_reLU @ weights["W2"] + weights["b2"].reshape(1,1,-1)

    residual2=residual1+hd2

    gamma2=weights["gamma2"].reshape(1,1,-1)
    beta2=weights["beta2"].reshape(1,1,-1)

    mean=np.mean(residual2,axis=-1,keepdims=True)
    var=np.var(residual2,axis=-1,keepdims=True)

    residual2=(residual2-mean)/np.sqrt(var+eps)
    return residual2*gamma2 + beta2