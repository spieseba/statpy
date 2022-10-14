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


def binned_stdErrs(data, binsize_max):
    stdErr_binned = []
    for binsize in range(1, binsize_max):
        data_binned = bin(data, binsize)
        stdErr_binned.append( np.std(data_binned, ddof=1) / np.sqrt(len(data_binned)) )
    return np.array(stdErr_binned)
