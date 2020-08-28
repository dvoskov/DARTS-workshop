import numpy as np
import h5py
import pandas as pd

def get_rates(rates_filename):
    a = np.genfromtxt(rates_filename, names=True, skip_header = 9)
    b = np.genfromtxt(rates_filename, names=True, skip_header = 4, skip_footer = a.size + 4)
    c = np.genfromtxt(rates_filename, names=True, skip_header = 6, skip_footer = a.size + 2)
    d = list(b.dtype.names)
    e = 5*list(c.dtype.names)
    for i in range(0,len(e)):
        d[i+2]=e[i] + d[i+2]
    d[-3] = 'P1WBHP'
    d[-1] = 'P1BPR'
    a.dtype.names = tuple(d)
    data = pd.DataFrame(a)
    return data




