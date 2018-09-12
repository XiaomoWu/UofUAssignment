
import numpy as np
import pandas as pd
import copy


# Import & some CONST
DATA_DIR = "C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-01-DT/experiment-data_new/data_new/"


class ID3:

    def __init__(self):
        pass
    
    # train_id3: training with ID3
    def train_id3(self, train = None, fpath = '', max_depth = 500):

        if not fpath and data is None:
            raise Exception("Must pass either a path to a data file or a pandas DataFrame object")    

        # train: the training set
        # trees: stack of current working tree, will be cleared at end
        # tree: an empty element of trees
        # rules: final results!
        # attr_name: all attr names excluding the label
        if train is None:
            self.train = pd.read_csv(fpath)
        else:
            self.train = train
        trees = [] 
        self.depth = 0
        self.tree = {'attr': [], 'value': [], 'parent_entropy': [], 'data': []} 
        self.rules = [] # 
        self.attr_name = set(self.train.columns[1:])
        self.max_depth = max_depth

        # if all labels are identical, return the label immediately
        unique_label = self.train['label'].value_counts()
        if unique_label.size == 1:
            self.rules.append(unique_label.index[0])
            return

        # if all labels are not identical, start growing trees
        else:
            return self._grow_trees(trees)

    # test_id3: test id3
    def test_id3(self, test = None, fpath = ''):

        if not fpath and test is None:
            raise Exception("Must pass either a path to a data file or a pandas DataFrame object")        
        
        if len(self.rules) == 0:
            raise Exception("No rules to be applied, please first train the classifier")
        
        # test: test set
        # rules: rules derived from traning set
        # n_test: # row of test
        # n_hit: # of hit
        # n_rules: # of rules
        # acc = n_hit / n_test

        if test is None:
            self.test = pd.read_csv(fpath)
        else:
            self.test = test
        rules = self.rules
        
        n_test = self.test.shape[0]
        n_rules = len(rules)
        n_hit = 0

        # obs: each row of test
        # rule: each rule in rules
        # if obs == rule, n_hit += 1
        for i in range(n_test): 
            #print('i:', i)
            obs = self.test.iloc[i]
            for j in range(n_rules):
                rule = rules[j]
                #print('j:', j)

                if obs['label'] == rule['label'] \
                    and all(obs.loc[rule['attr']] == rule['value']):
                    n_hit += 1
                    break

            print('# %s / # %s' % (n_hit, i + 1))

        return n_hit / n_test


    # grow_tree: a recursive implementation of ID3
    def _grow_trees(self, trees):
        
        # if there's no root, generate root
        if len(trees) == 0 and len(self.rules) == 0:
            parent_tree = self.tree
            parent_entropy = self._get_entropy(self.train['label'])
            parent_attr = []
            child_attr = self.attr_name
            data = self.train

            # new_node:  the attr with higest gain
            new_node = self._get_win_attr(data, child_attr, parent_entropy)
            trees.extend(self._update_parent_tree(parent_tree, new_node))
        
            # inc depth by 1
            #self.depth = 1

            return self._grow_trees(trees)

        # if all possible trees have been found, quit
        elif len(trees) == 0 and len(self.rules) > 0:
            return
     
        # if trees are not empty, keep adding nodes
        elif len(trees) > 0:
            
            new_trees = []
            
            # if not reach max, keep growing; otherwise, cut
            for t in trees:
                # print progress
                print('tree depth: %s,  # rules: %s' % (len(t['attr']), len(self.rules)))

                print(t['attr'])

                if len(t['attr']) < self.max_depth:
                    parent_tree = t
                    parent_entropy = parent_tree['parent_entropy'][-1]
                    parent_attr = set(parent_tree['attr'])
                    child_attr = self.attr_name - parent_attr
                    data = parent_tree['data']

                    # new_node:  the attr with higest gain
                    new_node = self._get_win_attr(data, child_attr, parent_entropy)
                    new_trees.extend(self._update_parent_tree(parent_tree, new_node))

                    return self._grow_trees(new_trees)

                elif len(t['attr']) == self.max_depth:
                    label = t['data']['label'].value_counts().index[0]
                    self.rules.append({'attr': t['attr'], 'value': t['value'], 'label': label})


    # get_entropy: given a label (Series) or a subset of it, return its entropy (float)
    def _get_entropy(self, label):
        value_count = pd.DataFrame({'count': label.value_counts()})
        value_count = value_count.assign(pct = lambda df: (df['count'] / df['count'].sum()))
        entropy = ((-1) * value_count['pct'] * np.log2(value_count['pct'])).sum()
        return entropy


    # get_win_attr: return the attr with the highest gain, together with its entropy and gain
    def _get_win_attr(self, data, child_attr, parent_entropy):
        # try every possible attr in child_attr
        gain_list = [] # attributes and their gain
        for attr in child_attr:
            # attr_values: values of A, their pct, and entropy 
            attr_values = pd.DataFrame({'pct': data[attr].value_counts(normalize = True)}) 

            for v in attr_values.index:
                attr_values.loc[v, 'entropy'] = self._get_entropy(data.loc[data[attr] == v, 'label'])
            attr_entropy = sum(attr_values['pct'] * attr_values['entropy'])
            gain_list.append({'attr': attr, 'entropy': attr_entropy, 'gain': (parent_entropy - attr_entropy)})

        gain_list = pd.DataFrame(gain_list)
        win_attr, entropy, gain = gain_list.sort_values('gain', ascending = False).iloc[0]
    
        return {'win_attr': win_attr, 'entropy': entropy, 'data': data}


    # update_parent_tree: add new_node to parent tree
    def _update_parent_tree(self, parent_tree, new_node):

        win_attr = new_node['win_attr']
        parent_entropy = new_node['entropy']
        parent_data = new_node['data']

        new_parent_tree = [] # a list of new tree, split on diff values of a SAME attr

        for v in parent_data[win_attr].unique():
            parent = copy.deepcopy(parent_tree)
            parent['attr'].append(win_attr)
            parent['value'].append(v)
            parent['data'] = parent_data.loc[parent_data[win_attr] == v]
            parent['parent_entropy'].append(parent_entropy)

            u = parent['data']['label'].value_counts()
            # if the branch is "pure" move to rules
            if len(u) == 1: parent['label'] = u.index[0]
            # there's no data for this tree, return the most common
            if len(u) == 0: parent['label'] = u[u == u.max()].index[0]
            if len(u) in [0, 1]:
                del parent['data']
                del parent['parent_entropy']
                self.rules.append(parent)
            # else, keep growing tree
            else:
                new_parent_tree.append(parent)

        return new_parent_tree

id3 = ID3()
id3.train_id3(fpath = DATA_DIR + 'small_train.csv', max_depth = 9)
id3.rules
#acc = x.test_id3(fpath = DATA_DIR + 'test.csv') # acc = 1.0





