import os
import time
from itertools import product
from multiprocessing import Process, Pool, Queue

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from joblib import dump, load



# Q1: A ------------------------------
def draw(n):
    int = np.random.randint(1, n+1)
    return int

def get_k(n):
    xs = []
    k = 0
    while len(set(xs)) == len(xs):
        x = draw(n)
        xs.append(x)
        k += 1
    return k

n = 5000
np.random.seed(42)
print(f"k is {get_k(n)}")

# Q1: B --------------------------
def get_ks(n, m):
    ks = []
    for i in range(m):
        ks.append(get_k(n))
    return ks

n = 5000
m = 300 # number of repeat
np.random.seed(42)
ks = get_ks(n, m)

# plot CDF
n_bins = 100
fig, ax = plt.subplots(figsize=(8, 4))
n, bins, patches = ax.hist(ks, n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
#n, bins, patches = plt.hist(ks, n_bins, facecolor='blue', alpha=0.5)

ax.grid(True)
ax.legend(loc='right')
ax.set_title('CDF')
ax.set_xlabel('k')
ax.set_ylabel('Cumlative density')
#plt.show()

# Q1: C --------------
n = 5000
m = 300 # number of repeat
np.random.seed(42)
ks = get_ks(n, m)
E_k= np.asarray(ks).sum() / len(ks)
print(f"Expected k is {E_k: .2f}")

# Q1: D -------
n = 5000
m = 300 # number of repeat
np.random.seed(42)
ts = time.time()
ks = get_ks(n, m)
tp = time.time() - ts
print(f"Use {tp: .2f} sec")

# plot time increase
if __name__ == '__main__':
    m_range = [300, 3000, 5000, 8000, 10000]
    n_range = range(1, 1100001, 100000)
    params = product(m_range, n_range)

    with Pool(8) as pool:
        results = pool.starmap(get_tp, params)
    dump(results, 'q1.joblib')
       
# plot
tps = load('q1.joblib')
ys = []
for m in m_range:
    y = []
    tp = {tp['n']: tp['tp'] for tp in tps if tp['m'] == m}
    for n in n_range:
        y.append(tp[n])
    ys.append(y)

plt.plot(n_range, ys[0], n_range, ys[1], n_range, ys[2], n_range, ys[3], n_range, ys[4])
plt.xlabel('N')
plt.ylabel('Time')
plt.grid(True)
#plt.show()




