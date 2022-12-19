import numpy as np

def bin(data, b, *argv):
    if len(argv) != 0:
        w = argv[0]
    Nb = len(data) // b # cuts off data which do not build a complete bin
    bata = []
    for i in range(Nb):
        if len(argv) != 0:
            mean = np.average(data[i*b:(i+1)*b], weights=w[i*b:(i+1)*b], axis=0)
        else:
            mean = np.mean(data[i*b:(i+1)*b], axis=0)
        bata.append(mean)
    return np.array(bata) 