import numpy as np

# binning
def bin(data, b, *argv):
    if b == 1:
        return data
    if len(argv) != 0:
        w = argv[0]
    Nb = len(data) // b # cut off data of last incomplete bin
    bata = []
    for i in range(Nb):
        if len(argv) != 0:
            mean = np.average(data[i*b:(i+1)*b], weights=w[i*b:(i+1)*b], axis=0)
        else:
            mean = np.mean(data[i*b:(i+1)*b], axis=0)
        bata.append(mean)
    return np.array(bata) 