import numpy as np
import pandas as pd
import copy
from collections import OrderedDict



class ID3():
    def __init__(self, fpath, max_depth):
        # max_depth: defaults to 100
        # train: traning set
        # feature_names (list): names of all features
        self.max_depth = 100
        self.traindata = pd.read_csv(fpath)
        self.feature_names = list(self.traindata)[1:]
        
    def train(self):
        # create empty root_node
        label = self.traindata['label']
        entropy_priori = self._calc_entropy(label)
        branch = OrderedDict()
        root_node = Node(entropy = entropy_priori, branch = branch)


        # recursively add children to root_node
        self._grow_tree(root_node)

        return root_node

    def _grow_tree(self, node):
        # create a subset of train from branch, only features not included in branch are kept
        #branch = OrderedDict([('time', 'two'), ('variety', 'alp')])
        kept_feature_names = list(set(self.feature_names) - set(node.branch))
        if len(node.branch) == 0:
            train = self.traindata
        else:
            train = pd.merge(self.traindata, pd.DataFrame(node.branch, index = [0])).loc[:, ['label'] + kept_feature_names]

        # update node:
        # - select feature
        # - update entropy
        # - add rule
        node.feature, node.entropy = self._select_feature(train, node.entropy)

        rule = train.loc[:, ['label'] + [node.feature]]
        rule['prob'] = 1
        rule = pd.DataFrame((rule.groupby([node.feature, 'label'])['prob'].sum().groupby(level = 0).transform(lambda x: x / x.sum())))
        node.rule = rule

        print("# node:", node.serial_n)
        
        # create /add child node
        # call _grow_tree recursively
        if node.entropy != 0:
            for value in train[node.feature].unique():
                branch_child = copy.deepcopy(node.branch)
                branch_child[node.feature] = value


                child_node = Node(entropy = node.entropy, branch = branch_child)
                node.children.append(child_node)

                self._grow_tree(child_node)

    def test(self, root_node):
        pass

    # _select_feature: given a train dataset, select the best feature
    # INPUT: a filtered training set
    # OUTPUT: the best feature's name (string)
    def _select_feature(self, train, entropy):
        entropy_prior = entropy

        feature_gain = pd.DataFrame(columns = ['entropy', 'gain'])
        for feature in list(train)[1:]:

            value_entropy = pd.DataFrame(train[feature].value_counts(normalize = True))

            for value in value_entropy.index:
                label = train.loc[train[feature] == value, 'label']
                value_entropy.loc[value, 'entropy'] = self._calc_entropy(label)

            entropy = np.average(value_entropy['entropy'], weights = value_entropy[feature])
            gain = entropy_prior - entropy

            feature_gain.loc[feature] = [entropy, gain]

        best_feature = feature_gain['gain'].sort_values(ascending = False).index[0]
        best_feature_entropy = feature_gain.loc[best_feature, 'entropy']

        return best_feature, best_feature_entropy


    # _calc_entropy: given a series of label, calculate entropy
    # INPUT: series, a pandas Series
    # RETURN: a num
    def _calc_entropy(self, series):
        #series = label
        value_count = series.value_counts(normalize = True)
        entropy = sum([-i * np.log2(i) for i in value_count])
        return entropy


class Node():
    # serial_n will be changed everytime a new node created
    serial_n = -1

    def __init__(self, entropy, branch):
        # feature (string): name of the feature to split on
        # entropy (num): when instantiated, the entropy is from its parents (BEFORE dertermining the feature), then the entropy will be updated (AFTER determining the feature) 
        # children (list): [child_node_1, child_node_2...]
        # branch (list): [{'feature1': value1}, {'feature2': value2}...]
        # if it's the end node, give the rule

        self.feature = None
        self.entropy = entropy
        self.branch = branch
        self.children = []
        self.seial_n = self._inc_serial_n()
        self.rule = {}

    def _inc_serial_n(self):
        Node.serial_n += 1
        return Node.serial_n



data_dir = "C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-01-DT/"

#train = pd.read_csv(data_dir + "small_train.csv")
#feature_names = list(train)[1:]
node = Node(entropy = 0, branch = [])

tree = ID3(data_dir + "train.csv", max_depth = 10)
x = tree.train()
