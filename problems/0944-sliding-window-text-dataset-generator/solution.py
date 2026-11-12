def sliding_window_dataset(token_ids: list[int], max_length: int, stride: int) -> list:
    """
    Generate (input, target) training pairs using a sliding window.

    Args:
        token_ids: List of integer token IDs
        max_length: Length of each input/target chunk (context window size)
        stride: Step size between consecutive windows

    Returns:
        List of (input_chunk, target_chunk) tuples, each chunk as a list of ints.
    """
    ans=[]
    for i in range(0,len(token_ids),stride):
        st=i
        end=i+max_length
        if(end<len(token_ids)):
            input=token_ids[i:i+max_length]
            output=token_ids[i+1:i+max_length+1]
            ans.append((input,output))
    return ans