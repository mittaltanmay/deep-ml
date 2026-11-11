def most_frequent_pair(sequences):
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """
    recurring={}
    for i in range(len(sequences)):
        for j in range(len(sequences[i])-1):
            pair=(sequences[i][j],sequences[i][j+1])
            recurring[pair]=recurring.get(pair,0)+1
    maxi=0
    bpe=None
    for key,value in recurring.items():
        if maxi<value:
            bpe=key
            maxi=value
    return bpe
    
