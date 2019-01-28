# import modules
import os
import time
from itertools import product
from multiprocessing import Process, Pool, Queue

import numpy as np
import pandas as pd
import sklearn
from joblib import dump, load


# Q2: A ------
def draw(n):
    x = np.random.randint(1, n+1)
    return x

def get_k(n):
    xs = set()
    k = 0
    while len(xs) < n:
        k += 1
        x = draw(n)
        xs.add(x)
    return k

np.random.seed(42)
n = 300
get_k(n)

# Q2: B -------
def get_ks(m, n):
    ks = []
    for i in range(m):
        ks.append(get_k(n))
    return ks

m = 400
n = 300
np.random.seed(42)
ts = time.time()
ks = get_ks(m, n)
tp = time.time() - ts
print(f"Use {tp: .2f} sec")

# plot CDF
n_bins = 50
fig, ax = plt.subplots(figsize=(8, 4))
n, bins, patches = ax.hist(ks, n_bins, density=True, histtype='step',
                           cumulative=True, label='Empirical')
# tidy up the figure
ax.grid(True)
ax.legend(loc='right')
ax.set_title('CDF')
ax.set_xlabel('k')
ax.set_ylabel('Cumlative density')
#plt.show()

## Q2: C --------
# expected k
m = 400
n = 300
np.random.seed(42)
ks = get_ks(m, n)
E_k = np.asarray(ks).sum()/len(ks)
print(f"Expected k is {E_k: .2f}")

## how long it takes for m = 400, n = 300
## already get in previous steps

# time increase
def get_tp(m, n):
    ts = time.time()
    ks = []
    for i in range(m):
        ks.append(get_k(n))
    tp = time.time() - ts
    return {'m': m, 'n': n, 'tp': tp}

# multiprocessing
if __name__ == '__main__':
    np.random.seed(42)
    m_range = [400, 1500, 2700, 3800, 5000]
    n_range = range(1, 22001, 2000)
    params = product(m_range, n_range)    

    with Pool(8) as pool:
        results = pool.starmap(get_tp, params)
    dump(results, 'q2.joblib')

# load q2.joblib
tps = load('q2.joblib')
ys = []
for m in m_range:
    y = []
    for n in n_range:
        tp = {tp['n']: tp['tp'] for tp in tps if tp['m'] == m}
        y.append(tp[n])
    ys.append(y)

# plot
plt.plot(n_range, ys[0], n_range, ys[1], n_range, ys[2], n_range, ys[3], n_range, ys[4])
plt.xlabel('N')
plt.ylabel('Time')
plt.grid(True)
plt.show()



