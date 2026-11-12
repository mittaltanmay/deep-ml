import numpy as np

def get_batch(data: np.ndarray, block_size: int, batch_size: int, seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed)
    batchSize=rng.integers(0,len(data)-block_size,size=batch_size)
    input=[]
    output=[]
    for i in batchSize:
        input.append(data[i:i+block_size])
        output.append(data[i+1:i+1+block_size])
    np_input=np.array(input)
    np_output=np.array(output)
    output=np.stack((np_input,np_output),axis=0)
    return output