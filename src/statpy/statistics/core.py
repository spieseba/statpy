import numpy as np

# binning
def bin(data, binsize, weights=None):
    assert binsize is not None
    if binsize == 1:
        return data
    N = len(data)
    w = np.ones(N) if weights is None else weights
    Nb = N // binsize # cut off data of last incomplete bin
    binned_data = []
    for i in range(Nb):
        mean = np.average(data[i*binsize:(i+1)*binsize], axis=0, weights=w[i*binsize:(i+1)*binsize])
        binned_data.append(mean)
    return np.array(binned_data) 