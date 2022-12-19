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

#def bin(data, binsize):
#    data_binned = [] 
#    for i in range(0, len(data), binsize):
#        data_avg = 0; append_bin = True
#        for j in range(0,binsize):
#            try:
#                data_avg += data[i+j]
#            except IndexError:
#                append_bin = False
#                continue
#        if append_bin:
#            data_binned.append(data_avg / binsize)
#    return np.array(data_binned)

