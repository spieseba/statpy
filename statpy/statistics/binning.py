import numpy as np

# produce binned copy of data ensemble with bins of size binsize 
def bin(data, binsize):
    data_binned = [] 
    for i in range(0, len(data), binsize):
        data_avg = 0; append_bin = True
        for j in range(0,binsize):
            try:
                data_avg += data[i+j]
            except IndexError:
                append_bin = False
                continue
        if append_bin:
            data_binned.append(data_avg / binsize)
    return np.array(data_binned)