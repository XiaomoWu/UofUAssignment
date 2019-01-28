#%%  read D1-D4 -----------------

# docs_char: a list of the input documents, each element is a long str
# docs_word: each element is a list of words
docs_char = []

for i in range(1, 5):
    with open(f"D{i}.txt") as f:
        line = f.readline()
        docs_char.append(line)
 
#%% create k-gram ---------------
def get_kgram_char(k, doc):
    kgram = []
    for i in range(len(doc) - (k-1)):
        kgram.append(doc[i:(i+k)])
    return kgram

import itertools
docs = [set(get_kgram_char(2, doc)) for doc in docs_char]
V = set(itertools.chain.from_iterable(docs))
doc1 = set(get_kgram_char(2, docs_char[0]))
doc2 = set(get_kgram_char(2, docs_char[1]))
docs = docs[0:2]


#%% output answer ----------------
from hashlib import sha1
# size of doc
len(doc1) # 266
len(doc2) # 265

# size of V
N = len(V) # 309

# number of hash functions

# minhash
# x: list (set)
# salt: a random number
def minhash(doc, salt):
    doc_hash = [sha1((w + str(salt)).encode()).hexdigest() for w in doc]
    return min(doc_hash)

# simulate!
import datetime
def simulate():
    for T in [20, 60, 150, 300, 600, 1000, 2000, 5000, 10000]:
        ts = datetime.datetime.now()
        np.random.seed(42)
        hit = 0
        salts = np.random.randn(T)
        for t in range(T):
            salt = str(salts[t])
            doc_minhash = [minhash(doc, salt) for doc in docs]
            if len(set(doc_minhash)) == 1:
                hit += 1
        td = datetime.datetime.now() - ts
        print(f"Empirical JS: {(hit / T): .3f} @ T={T}, {td}")
simulate()

