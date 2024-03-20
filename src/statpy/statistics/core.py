import numpy as np

# binning
def bin(data, b, weights=None):
    assert b is not None
    if b == 1:
        return data
    w = np.ones(len(data)) if weights is None else weights
    Nb = len(data) // b # cut off data of last incomplete bin
    binned_data = []
    for i in range(Nb):
        mean = np.average(data[i*b:(i+1)*b], axis=0, weights=w[i*b:(i+1)*b])
        binned_data.append(mean)
    return np.array(binned_data) 