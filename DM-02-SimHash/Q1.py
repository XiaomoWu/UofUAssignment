#%%  read D1-D4 -----------------

# docs_char: a list of the input documents, each element is a long str
# docs_word: each element is a list of words
docs_char = []
docs_word = []

for i in range(1, 5):
    with open(f"D{i}.txt") as f:
        line = f.readline()
        docs_char.append(line)
        docs_word.append(line.split())
 
#%% create k-gram ---------------
def get_kgram_char(k, doc):
    kgram = []
    for i in range(len(doc) - (k-1)):
        kgram.append(doc[i:(i+k)])
    return kgram

def get_kgram_word(k, doc):
    kgram = []
    for i in range(len(doc) - (k-1)):
        kgram.append(' '.join(doc[i:(i+k)]))
    return kgram

#%% get similarity ------------
def get_js(s1, s2):
    s1_inter_s2 = s1.intersection(s2)
    s1_union_s2 = s1.union(s2)
    return len(s1_inter_s2) / len(s1_union_s2)

#%% output answer ----------------
# Q1.A
# 2-gram, char
[len(set(get_kgram_char(2, doc))) for doc in docs_char]

# 3-gram, char
[len(set(get_kgram_char(3, doc))) for doc in docs_char]

# 2-gram, word
[len(set(get_kgram_word(2, doc))) for doc in docs_word] 


# Q1.B
# 2-gram char
comb_idx = combinations(range(1, 5), 2)
docs_char_2gram = [set(get_kgram_char(2, doc)) for doc in docs_char]
for idx, comb in zip(comb_idx, combinations(docs_char_2gram, 2)):
    f"{idx}, {get_js(comb[0], comb[1]): .3f}"

# 3-gram char
comb_idx = combinations(range(1, 5), 2)
docs_char_3gram = [set(get_kgram_char(3, doc)) for doc in docs_char]
for idx, comb in zip(comb_idx, combinations(docs_char_3gram, 2)):
    f"{idx}, {get_js(comb[0], comb[1]): .3f}"

# 2-gram word
comb_idx = combinations(range(1, 5), 2)
docs_word_2gram = [set(get_kgram_word(2, doc)) for doc in docs_word]
for idx, comb in zip(comb_idx, combinations(docs_word_2gram, 2)):
    f"{idx}, {get_js(comb[0], comb[1]): .3f}"