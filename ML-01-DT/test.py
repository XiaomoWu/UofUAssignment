import numpy as np

train = pd.read_csv(DATA_DIR + 'train.csv')
Stest = pd.read_csv(DATA_DIR + 'small_train.csv')
rules = [{'attr': ['smell', 'time'], 'value': ['none', 'two'], 'label': False},
 {'attr': ['smell', 'time'], 'value': ['none', 'one'], 'label': True},
 {'attr': ['smell', 'variety'], 'value': ['sweet', 'alp'], 'label': True},
 {'attr': ['smell', 'variety'], 'value': ['sweet', 'haden'], 'label': True},
 {'attr': ['smell', 'variety'], 'value': ['sweet', 'keitt'], 'label': False}]
n_test = test.shape[0]
n_hit = 0
n_rules = len(rules)

# obs: each row of test
# rule: each rule in rules
# if obs == rule, n_hit += 1
for i in range(n_test): 
    print('i:', i)
    obs = test.iloc[i]
    for j in range(n_rules):
        rule = rules[j]
        print('j:', j)

        if obs['label'] == rule['label'] \
            and all(obs.loc[rule['attr']] == rule['value']):
            n_hit += 1
            break
            

def foo():
    1 + 1
    return

foo()