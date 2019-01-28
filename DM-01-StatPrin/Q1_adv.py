import os
import time
from itertools import product
from multiprocessing import Process, Pool, Queue

import numpy as np
import pandas as pd
import sklearn
from joblib import dump, load


# draw from [1, n]
def draw(n):
    int = np.random.randint(1, n+1)
    return int

# Q1: A ------------------------------
def get_k(n):
    xs = []
    k = 0
    while len(set(xs)) == len(xs):
        x = draw(n)
        xs.append(x)
        k += 1
    return k

def get_tp(m, n):
    ts = time.time()
    for i in range(m):
        get_k(n)
    tp = time.time() - ts
    return {'m': m, 'n': n, 'tp': tp}


if __name__ == '__main__':
    m_range = [300, 3000, 5000, 8000, 10000]
    n_range = range(1, 1100001, 100000)
    params = product(m_range, n_range)

    with Pool(16) as pool:
        results = pool.starmap(get_tp, params)
    dump(results, 'q1.joblib')


#plt.plot(n_range, tps[0], n_range, tps[1])
#plt.xlabel('N')
#plt.ylabel('Time')
#plt.grid(True)
#plt.show()




