import numpy as np
import pandas as pd
from copy import deepcopy
from utils import get_data, init_weight, shuffle_samples

class BaseClf:
    def predict(self, X, report = True):
        """
        Parameters
        -------------
        X : DataFrame, columns = ['y', 'X']

        Returns
        --------
        pred: dict.
            pred['y_pred']: list, shape = (n_samples,)
            pred['acc']: accuracy
            pred['p']: precision
            pred['r']: recall
            pred['f']: f score
        """
        assert (type(X) == pd.DataFrame) & ('X' in X), "Input must be a DataFrame with column 'X'"

        y = X['y']
        y_pred = []
        if hasattr(self, 'weight'):
            w = self.weight
        for i in range(X.shape[0]):
            x = X.loc[i, 'X']
            if type(self) == SVM:
                if np.dot(w, x) > 0:
                    y_pred.append(1)
                else:
                    y_pred.append(-1)
            elif type(self) == Logistic:
                prob = 1 / (1 + np.exp(-np.dot(w, x)))
                if prob > 0.5:
                    y_pred.append(1)
                else:
                    y_pred.append(-1)
            elif type(self) == NB:
                lil = self.lil
                posts = pd.Series(index = lil['y'], data = [0, 0])
                for lil_y in lil['y']:
                    prior = lil.loc[lil.y == lil_y, 'priors'].values[0]
                    lil_x = lil.loc[lil.y == lil_y, 'likelihood'].values[0]
                    assert len(lil_x) == len(x) - 1, "dim(likelihood) != (n_feature - 1)"
                    post = np.sum(lil_x * x[1:]) + np.log(prior)
                    posts[lil_y] = post
                y_pred.append(posts.sort_values(ascending = False).index[0])
            elif type(self) == Tree:
                assert len(self.rules) > 0, "No rules to be applied, please first train the classifier"
        
                rules = self.rules
                n_rules = len(rules)

                hit = 0
                for j in range(n_rules):
                    rule = rules[j]
                    if all(np.array([x[int(i)] for i in rule['attr']]) == rule['value']):
                        y_pred.append(rule['label'])
                        hit += 1
                        break
                if hit == 0:
                   print("No fitting rule found!")
                   y_pred.append(1)

           

        if report & (type(self) != SVMTree):
            tp, fp, fn = 0, 0, 0
            for i in range(len(y)):
                if (y[i] == y_pred[i]) & (y_pred[i] == 1): tp += 1
                if (y[i] != y_pred[i]) & (y_pred[i] == 1): fp += 1
                if (y[i] != y_pred[i]) & (y_pred[i] == -1): fn += 1

            acc = (sum(y == y_pred) / len(y))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print("Accuracy: %1.3f    Precision: %1.3f    Recall: %1.3f     F: %1.3f" % (acc, p, r, f))

        return {'acc': acc, 'p': p, 'r': r, 'f': f, 'y_pred': y_pred}


class SVM(BaseClf):
    """
    Attributes
    -----------
    weight : array-like, shape = (1, n_features) 
    data_train : DataFrame, shape = (n_samples, n_features)
    """

    def fit(self, fpath = None, data = None, n_features = None, n_epoch = 1, r = 1, c = 1, **kargs):
        """
        Parameters
        -------------
        fpath: str. file dir for trainning set
        data: DataFrame. tranining set, needed only if fpath not given
        n_features: int. # features, needed only if fpath not given
        r: float. learning rate
        c: float. tradeoff between regularizer and loss
        n_epoch: int. max epoch

        Returns
        --------
        self: object
        """
        assert (r <= 1), "Learning rate must be no more than one!"

        if data is None: # if data not given, read from file
            n_features, n_samples, data_train = get_data(fpath)
        else: # else get data directly
            data_train = data
            n_samples = data_train.shape[0]
            x_temp = data_train.iloc[0]['X']
            if type(x_temp) == list:
                n_features = len(x_temp)
            else:
                n_features = x_temp.shape[0]

        # 1. initialize weight, r
        w = init_weight(n_features)
        r_0 = r

        # 2. for epoch = 1...T
        for epoch in range(1, n_epoch + 1):
        # shuffle traning set
            data = shuffle_samples(data_train, epoch)
            data.index = range(data.shape[0])
            self.data_train = data
        # for each example, update weight
            r = r_0 / epoch
            for t in range(n_samples):
                y = data.loc[t, 'y']
                x = data.loc[t, 'X']
                if type(x) == list: x = np.array(x)
                assert (w.shape == x.shape), "dim(w) != dim(x)"
                loss = y * np.dot(w, x)
                if loss <= 1:
                    w = (1 - r) * w + r * c * y * x
                else:
                    w = (1 - r) * w
        # print objective
            jw = 0.5 * np.dot(w, w) + c * max(0, 1 - loss)
            print("Epoch = %s   J(w) = %1.4f" % (epoch, jw))

        # 3. return w
        self.weight = w

        #return self
   

class SVMTree(BaseClf):
    """
    Attributes
    -----------
    rules: list, shape = (n_trees,)
        each element is a decision tree

    """
    def fit(self, fpath = None, data = None, n_features = None, max_depth = 10, n_trees = 200, r = 1, c = 1, n_epoch = 1, **kargs):
        """
        Parameters
        -------------
        max_depth : (TREE) int. tree depth
        n_trees : (TREE) int. # transformed features
        r : (SVM) learning rate 
        c : (SVM) tradeoff
        n_epoch : # epoch 

        Return
        -------
        self
        """

        # get data_train
        if data is None:
            n_features, n_samples, data_train = get_data(fpath)
        else:
            data_train = data
            n_samples = data_train.shape[0]
        self.data_train = data_train

        # transform features (train 200 trees)
        rules = []
        for i in n_trees:
            print("n_trees: %s / %s" % (i - min(n_trees) + 1, len(n_trees)))
            data = shuffle_samples(data_train, seed = i, frac = 0.1)
            tree = Tree()
            tree.fit(data = data, n_features = n_features, max_depth = max_depth, verbose = False)
            rules.append(tree.rules)


        self.rules = rules
        return self

    def transform(self, data, rules_list):
        """
        Return
        -------
        X_trans: DataFrame('y', 'X')
        """
        X_trans = []
        for i in range(data.shape[0]):
            x = data.loc[i, 'X']
            x_trans = []
            for rules in rules_list:
                hit = 0
                for rule in rules:
                    if all(np.array([x[int(i)] for i in rule['attr']]) == rule['value']):
                        x_trans.append(rule['label'])
                        hit += 1
                        break
                if hit == 0:
                    print("No fitting rules found!")
                    x_trans.append(1) # if no rule match, default to "1"
            x_trans.append(1) # biase term
            X_trans.append({'X': x_trans})
        X_trans = pd.DataFrame(X_trans)

        if 'y' in data:
            X_trans['y'] = data['y']
        return X_trans


class Tree(BaseClf):
    """
    Attributes
    -----------
    rules : list of dicts.
        each dict is a rule, shape  {'attr', 'value', 'label'}
    """
    
    def fit(self, fpath = None, data = None, n_features = None, max_depth = 50, verbose = True, **kargs):
        """
         train: the training set
         trees: stack of current working tree, will be cleared at end
         tree: an empty element of trees
         rules: final results!
         attr_name: all attr names excluding the label
        """
        if data is None:
            n_features, n_samples, data_train = get_data(fpath)
        else:
            data_train = data
            n_samples = data_train.shape[0]
        self.data_train = data_train

        trees = [] 
        self.tree = {'attr': [], 'value': [], 'parent_entropy': [], 'data': []} 
        self.rules = [] # 
        #self.attr_name = set(self.data_train.columns[1:])
        self.attr_name = set(list(range(n_features)))
        self.max_depth = max_depth
        self.verbose = verbose

        # if all labels are identical, return the label immediately
        unique_label = self.data_train['y'].value_counts()
        if unique_label.size == 1:
            self.rules.append(unique_label.index[0])
            return

        # if all labels are not identical, start growing trees
        else:
            return self._grow_trees(trees)


    # grow_tree: a recursive implementation of ID3
    def _grow_trees(self, trees):
        
        # if there's no root, generate root
        if len(trees) == 0 and len(self.rules) == 0:
            parent_tree = self.tree
            parent_entropy = self._get_entropy(self.data_train['y'])
            parent_attr = []
            child_attr = self.attr_name

            data = self.data_train['X']
            data = pd.DataFrame(np.array([list(i) for i in data]))
            data['y'] = self.data_train['y'].tolist()

            # new_node:  the attr with higest gain
            new_node = self._get_win_attr(data, child_attr, parent_entropy)
            trees.extend(self._update_parent_tree(parent_tree, new_node))
        
            return self._grow_trees(trees)

        # if all possible trees have been found, quit
        elif len(trees) == 0 and len(self.rules) > 0:
            return
     
        # if trees are not empty, keep adding nodes
        elif len(trees) > 0:
            
            new_trees = []
            
            for t in trees:

                # print progress
                if self.verbose:
                    print('tree depth: %s,  # rules: %s' % (len(t['attr']), len(self.rules)))

                # if not reach max, keep growing; otherwise, cut
                if len(t['attr']) < self.max_depth:

                    parent_tree = t
                    parent_entropy = parent_tree['parent_entropy'][-1]
                    parent_attr = set(parent_tree['attr'])
                    child_attr = self.attr_name - parent_attr
                    data = parent_tree['data']

                    # new_node:  the attr with higest gain
                    new_node = self._get_win_attr(data, child_attr, parent_entropy)
                    new_trees.extend(self._update_parent_tree(parent_tree, new_node))

                # if reach max, return
                elif len(t['attr']) == self.max_depth:
                    label = t['data']['y'].value_counts().index[0]
                    self.rules.append({'attr': t['attr'], 'value': t['value'], 'label': label})

            return self._grow_trees(new_trees)

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
                attr_values.loc[v, 'entropy'] = self._get_entropy(data.loc[data[attr] == v, 'y'])
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
            parent = deepcopy(parent_tree)
            parent['attr'].append(win_attr)
            parent['value'].append(v)
            parent['data'] = parent_data.loc[parent_data[win_attr] == v]
            parent['parent_entropy'].append(parent_entropy)

            u = parent['data']['y'].value_counts()
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


class Logistic(BaseClf):
    """
    Attributes
    -----------
    weight : array-like, shape = (1, n_features) 
    data_train : DataFrame, shape = (n_samples, n_features)
    """
    def fit(self, fpath = None, data = None, n_features = None, n_epoch = 1, r = 1, sigma = 1, **kargs):
        """
        Parameters
        -------------
        fpath: str. file dir for trainning set
        data: DataFrame. tranining set, needed only if fpath not given
        n_features: int. # features, needed only if fpath not given
        r: float. learning rate
        sigma: float. Tradeoff
        n_epoch: int. epoch

        Returns
        --------
        self: object
        """

        # 0. get training set
        if data is None:
            n_features, n_samples, data_train = get_data(fpath)
        else:
            data_train = data
            n_samples = data_train.shape[0]

            x_temp = data_train.iloc[0]['X']
            if type(x_temp) == list:
                n_features = len(x_temp)
            else:
                n_features = x_temp.shape[0]

        # 1. initialize weight, r
        w = init_weight(n_features)
        r_0 = r

        # 2. for each epoch:
        for epoch in range(1, n_epoch + 1):
            jw = []
            # (1) shuffle training set
            data_train = shuffle_samples(data_train, epoch)
            # (2) update weight
            #r = r_0 / epoch # diminishing r
            for i in range(n_samples):
                y = data_train.iloc[i, 0]
                x = data_train.iloc[i, 1]
                grad = (2/sigma**2) * w - (y * x) / (1 + np.exp(y * np.dot(w, x)))
                assert (w.shape == x.shape), "dim(w) != dim(x)"
                assert (grad.shape == w.shape), "dim(w) != dim(gradient)"
                w = w - r * grad
                jw.append((1/sigma**2) * np.dot(w, w) + np.log(1 + np.exp(-y * np.dot(w, x))))
            # print objective
            jw = np.mean(jw)
            print("Epoch = %s   J(w) = %1.4f" % (epoch, jw))

        # 3. return w
        self.weight = w

        return self


class NB(BaseClf):
    """
    Attributes
    -----------
    lil : DataFrame, shape = (n_samples, n_features-1) 
        lil contains priors and likelihood
    data_train : DataFrame, shape = (n_samples, n_features)
    """

    def fit(self, fpath = None, data = None, n_features = None, smooth = 1, **kargs):
        # get data
        if data is None:
            n_features, n_samples, data_train = get_data(fpath)
        else:
            data_train = data
            n_samples = data_train.shape[0]

            x_temp = data_train.iloc[0]['X']
            if type(x_temp) == list:
                n_features = len(x_temp)
            else:
                n_features = x_temp.shape[0]


        self.data_train = data_train

        # get priors
        priors = data_train['y'].value_counts(normalize = True)
        
        # update likelihood
        lil =[]
        for y in priors.index:
            prior = priors[y]
            lil_y = np.zeros(n_features - 1)
            for i in range(n_samples):
                x = data_train.iloc[i, 1]
                y_i = data_train.iloc[i, 0]
                if y_i == y:
                    for j in range(1, n_features):
                        if x[j] > 0:
                            lil_y[j-1] += 1
            # smooth likelihood
            lil_y = [l if l > 0 else l + smooth for l in lil_y] 
            n_y = sum(lil_y)
            lil_y = np.log([l / n_y for l in lil_y])
            lil.append({'y': y, 'priors': prior, 'likelihood': lil_y})

        # return
        lil = pd.DataFrame(lil, columns = ['y', 'priors', 'likelihood'])
        self.lil = lil
        return self



