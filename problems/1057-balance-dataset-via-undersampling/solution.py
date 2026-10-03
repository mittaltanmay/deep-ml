def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    freq={}
    for i in range(len(data)):
        freq[data[i][1]]=freq.get(data[i][1],0)+1
    mini=100
    for key,value in freq.items():
        mini=min(mini,value)
    ans=[]
    taken={}
    for i in range(len(data)):
        if taken.get(data[i][1],0)==mini:
            continue
        else:
            ans.append(data[i])
            taken[data[i][1]]=taken.get(data[i][1],0)+1
    return ans        

