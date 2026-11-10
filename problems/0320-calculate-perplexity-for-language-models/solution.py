import torch

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    probability=torch.tensor(probabilities,dtype=torch.float32)
    n=len(probabilities)
    prob=torch.sum(torch.log(probability))
    ppw=torch.exp(-1*(prob/n))
    return ppw.item()