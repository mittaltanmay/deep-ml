import numpy as np
from scipy.special import softmax
def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	d_k=K.shape[1]
	attention=np.dot(Q,K.T)/d_k**0.5
	if isinstance(mask,np.ndarray):
		for i in range(len(mask)):
			for j in range(len(mask[i])):
				if mask[i][j]==0:
					attention[i][j]=float('-inf')
	attentions=softmax(attention,axis=1)
	return (attentions @ V,attentions)
