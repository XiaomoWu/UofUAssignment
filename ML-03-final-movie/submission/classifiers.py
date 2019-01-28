import time
import os
import random
from copy import deepcopy
from operator import itemgetter

from utils import *
from scipy.sparse import csr_matrix
from scipy import stats
import numpy as np
import pandas as pd

class BaseClf:
    def __init__(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    @staticmethod
    def score_pred(ys, y_preds):
        """report {acc, pre, recall, f}
        """
        tp, fp, fn = 0, 0, 0
        for y, y_pred in zip(ys, y_preds):
            if (y == y_pred) & (y_pred == 1): tp += 1
            if (y != y_pred) & (y_pred == 1): fp += 1
            if (y != y_pred) & (y_pred == -1): fn += 1

        acc = (sum(ys == y_preds) / len(ys))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print("Accuracy: %1.3f    Precision: %1.3f    Recall: %1.3f     F: %1.3f" % (acc, precision, recall, f))

        return {'acc': acc, 'precision': precision, 'recall': recall, 'f': f, 'y_preds': y_preds}
        

class LinearClf(BaseClf):

    @timeit("Pred")
    def predict(self, data, weights):
        """
        Parameters
        ------------
        data : DataFrame, columns = ['y', 'x'] or ['x']
        weights : sparse matrix, shape = (1, n_features)

        Returns
        --------
        {y_pred, acc, p, r, f} - if 'y' in data.columns
        {y_pred} - if 'y' NOT in data.columns
        """

        # make prediction
        assert (all(data.index == range(data.shape[0]))), "all(data.index != range(data.shape[0])"
        from scipy.sparse import csr_matrix
        assert (type(weights) == csr_matrix), "type(weight) != csr_matrix"

        y_preds = []
        for i in range(data.shape[0]):
            x = data.loc[i, 'x']
            if (weights * x.T)[0, 0] > 0:
                y_preds.append(1)
            else:
                y_preds.append(-1)

        # return
        if 'y' not in data.columns:
            return {'y_preds': y_preds}
        else:
            ys = data['y']
            return self.score_pred(ys, y_preds)


class KNN(BaseClf):
    def __init__(self, method='cos', k=3):
        super().__init__(method=method, k=k)

    @timeit('Fit KNN')
    def fit(self, data=None):
        # check input must be compact
        assert (data.shape[0] == 1), "Input data must be compact!"

        # get training data
        data_train = data
        n_samples, n_features = get_n_samples_features_from_df(data)

        # normalize 'x'
        #x = data_train['x'].apply(self.normalize)
        x = data_train.loc[0, 'x']
        norm = csr_matrix(1 / np.sqrt(x.multiply(x).sum(axis = 1)))
        x = x.multiply(norm)
        data_train.loc[0, 'x'] = x

        # return
        self.data_train = data_train
        self.n_samples = n_samples
        self.n_features = n_features

        return self

    @timeit('Pred KNN')
    def predict(self, data):
        """
        Returns
        --------
        {y_pred, acc, p, r, f} - if 'y' in data.columns
        {y_pred} - if 'y' NOT in data.columns
        """
        # get param
        k = self.k
        data_test = data
        data_train = self.data_train
        n_samples_train = self.n_samples
        n_samples_test = data_test.loc[0, 'x'].shape[0]

        # check data_train must be compact
        assert (data_test.shape[0] == 1), "Input data must be compact!"

        # get x_train and y_train
        x_test = data_test.loc[0, 'x']
        norm = csr_matrix(1 / np.sqrt(x_test.multiply(x_test).sum(axis = 1)))
        x_test = x_test.multiply(norm)
        y_train = data_train.loc[0, 'y']
        x_train = data_train.loc[0, 'x']

        # make prediction
        y_preds = []

        # cosine distance
        if self.method == 'cos':
            prod = (x_train * x_test.T).tocsc()
            for i in range(n_samples_test): # each exmaple in data_test
                col_i = pd.Series(prod.getcol(i).toarray().reshape(n_samples_train))
                k_nbr_index = col_i.sort_values(ascending=False).iloc[0:k].index
                k_nbr_ys = y_train[k_nbr_index]
                k_nbr_y = 1 if k_nbr_ys.sum() > 0 else -1
                y_preds.append(k_nbr_y)

        # euclidean distance
        elif self.method == 'euclidean':
            # subtract each x_test example from x_train
            for i_test in range(n_samples_test)[0:5]:
                dists_for_i_in_test = []
                print("Predicting row %i of the test set" % i_test)
                for i_train in range(n_samples_train):
                    x_test_i = x_test.getrow(i_test)
                    x_train_i = x_train.getrow(i_train)
                    subtract = x_train_i - x_test_i
                    dist = np.sqrt((subtract * subtract.T)[0, 0])
                    dists_for_i_in_test.append(dist)
                dists_for_i_in_test = pd.Series(dists_for_i_in_test)
                k_nbr_index = dists_for_i_in_test.sort_values().iloc[0:k].index
                k_nbr_ys = y_train[k_nbr_index]
                k_nbr_y = 1 if k_nbr_ys.sum() > 0 else -1
                y_preds.append(k_nbr_y)


        # return
        if 'y' not in data.columns:
            return {'y_preds': y_preds}
        else:
            ys = data.loc[0, 'y']
            # check
            assert (ys.shape[0] == len(y_preds)), "ys.shape[0] != len(y_preds)"

            return self.score_pred(ys, y_preds)
         
    @staticmethod
    def normalize(x):
        """normalize x (divided by its norm)

        Parameters
        ------------
        x : csr_matrix, shape = (1, n_features)
        """

        norm = np.sqrt((x * x.T)[0, 0])
        return x / norm


class NB(BaseClf):
    """
    Attributes
    -----------
    jll : DataFrame, shape = (n_samples, n_features-1) 
        jll contains priors and likelihood
    unique_ys : list
        unique labels
    """
    def __init__(self, smooth=1):
        super().__init__(smooth=smooth)
        

    @timeit('Fit NB')
    def fit(self, data = None):
        # check input must be compact
        assert (data.shape[0] == 1), "Input data must be compact!"

        #get params
        smooth = self.smooth
        print("Training NB: smooth = %s" % (smooth))

        # get data
        n_samples, n_features = get_n_samples_features_from_df(data)
        self.data_train = data

        # get x
        x = data.loc[0, 'x']
        ## set non-zeros to one
        x = set_none_zero(x, 1)

        # get priors
        y = pd.Series(data.loc[0, 'y'])
        priors = y.value_counts(normalize = True)
        unique_ys = priors.index.values # [-1, 1]

        # update likelihood
        ls = [] # log-likihoods, list of P(x_i | y)
        # for each y, update its posteriors
        for unique_y in unique_ys: 
            prior = priors[y]
            index_x_given_y = y.loc[y == unique_y].index.values
            x_given_y = x[index_x_given_y,]
            x_rowsum_given_y = np.squeeze(np.asarray(x_given_y.sum(axis=0), dtype = np.int64))
            #l = x_rowsum_given_y / x_rowsum_given_y.sum() # liklihoods, shape = (1, n_features)
            l = (x_rowsum_given_y + smooth) / (index_x_given_y.shape[0] + smooth * n_features) # liklihoods w/ smooth, shape = (1, n_features)
            #ll = np.log(l) # take log liklihood, we don't do this in tranining
            ls.append(l)
        
        # jll: DataFrame, joint liklihood. 
        #   columns = [liklihoods, priors]
        #   index = [-1, 1]
        #   shape = (2, 2)
        jl = pd.DataFrame({'ls': ls, 'priors': priors}) 

        self.jl = jl
        self.unique_ys = unique_ys


    @timeit('Pred NB')
    def predict(self, data):
        # get x
        x = data.loc[0, 'x']

        ## normalize 'x'
        x = set_none_zero(x, 1) # set None-zeros to 1
        x_inverse = (x * -1) + np.array([1]) # set 1 to 0, and 0 to 1


        y = pd.Series(data.loc[0, 'y'])
        jl = self.jl

        # check input
        assert (x.shape[1] == len(jl['ls'].iloc[0])), "test data's n_features != train data" # n_features should be identical

        # make prediction
        compare = {} # store prob of each unique_y
        for unique_y in self.unique_ys:
            prior = jl.loc[unique_y, 'priors']
            ls = jl.loc[unique_y, 'ls'] # ls is an "array"
            lls = np.log(ls)
            lls_inverse = np.log(1 - ls)

            posteri = np.log(prior) + x.dot(lls) + np.squeeze(np.asarray(x_inverse.dot(lls_inverse))) # posteri if classified as unique_y
            compare[unique_y] = posteri

        # get the label of largest prob
        compare = pd.DataFrame(compare)
        compare = compare.T
        y_preds = compare.idxmax().values

        # return
        if 'y' not in data.columns:
            return {'y_preds': y_preds}
        else:
            ys = data.loc[0, 'y']
            return self.score_pred(ys, y_preds)


class Perceptron(LinearClf):
    """
    Attributes
    -----------
    weights : csr_matrix, shape = (1, n_features) 
    data_train : DataFrame, shape = (n_samples, n_features)
    """

    # set default params
    def __init__(self, r=1, margin=0, n_epoch=1):
        # get user param
        super(Perceptron, self).__init__(r=r, margin=margin, n_epoch=n_epoch)


    @timeit('Fit Perceptron')
    def fit(self, data=None):
        # get params
        r = self.r
        margin = self.margin
        n_epoch = self.n_epoch
        assert (data.shape[0] > 1), "Input data CANNOT be compact!"
        print("Training Perceptron: r = %s   margin = %s    n_epoch = %s" % (r, margin, n_epoch))

        # get  data
        n_samples, n_features = get_n_samples_features_from_df(data)
        self.data_train = data
        x = data.loc[0, 'x']
        y = data.loc[0, 'y']

        # initialize weight
        w = init_weight(n_features, all_zeros=True)
        wa = 0 # average weights
        r_0 = r
        weights = []

        for epoch in range(1, n_epoch+1):
            # shuffle traning set
            data = shuffle_samples(data, seed=epoch)
            assert all(data.index == range(n_samples)), "Index of data_train should be range(n_samples)"

            # for each example, update weight
            mistake_n = 0
            r = r_0 / epoch
            for i in range(n_samples):
                y = data.loc[i, 'y']
                x = data.loc[i, 'x']

                assert (w.shape == x.shape), "dim(w) != dim(x)"
                assert (type(y) == np.int64), "type(y) != np.int64"
                
                ywx = (y * w * x.T)[0, 0]
                if ywx <= margin:
                    mistake_n += 1
                    w += r * y * x
                wa += w

            # if input are compact
            #for i in range(n_samples):
            #    x_i = x.getrow(i)
            #    y_i = y[i]

            #    assert (w.shape == x_i.shape), "dim(w) != dim(x)"

            #    ywx = (y_i * w * x_i.T)[0, 0]
            #    if ywx <= margin:
            #        mistake_n += 1
            #        w += r * y_i * x_i
            #    wa += w

            # print mistake # for each epoch
            print('Epoch = %s   Mistake # = %s' % (epoch, mistake_n))

            # at the end of each epoch, append wa to weights
            weights.append(deepcopy(wa))

        # 3. return weights
        self.weights = weights
        return self


class Logistic(LinearClf):
    """
    Attributes
    -----------
    weights : csr_matrix, shape = (1, n_features) 
    data_train : DataFrame, shape = (n_samples, n_features)
    """

    def __init__(self, r=1, sigma=1, n_epoch=1):
        super().__init__(r=r, sigma=sigma, n_epoch=n_epoch)


    @timeit('Fit Logistic')
    def fit(self, data = None):
        """
        Parameters
        -------------
        data: DataFrame. 
            tranining set, needed only if fpath not given

        r: float. learning rate

        sigma: float. s.e. of normal distribution

        n_epoch: int. max epoch

        Returns
        --------
        self: object
        """
        # get params
        n_epoch = self.n_epoch
        r = self.r
        sigma = self.sigma

        assert (r <= 1), "Learning rate must be no more than one!"
        print("Training Logistic: r = %s   sigma = %s   n_epoch = %s" % (r, sigma, n_epoch))

        # get data
        n_samples, n_features = get_n_samples_features_from_df( data)
        self.data_train = data

        # 1. initialize weight, r
        w = init_weight(n_features)
        r_0 = r

        # 2. for each epoch:
        weights = []
        for epoch in range(1, n_epoch + 1):
            # (1) shuffle training set
            data = shuffle_samples(data, seed=epoch)

            # (2) update weight
            r = r_0 / epoch # diminishing r
            assert all(data.index == range(n_samples)), "index of data_train should be range(n_samples)"
            jw = []
            for i in range(n_samples):
                y = data.loc[i, 'y']
                x = data.loc[i, 'x']
                assert (w.shape == x.shape), "dim(w) != dim(x)"
                assert (type(y) == np.int64), "type(y) != np.int64"

                power = (y * w * x.T)[0, 0]
                if power > 500:
                    grad = (2/sigma**2) * w
                    jw.append((1/sigma**2) * (w * w.T)[0, 0])
                else:
                    grad = (2/sigma**2) * w - (y * x) / (1 + np.exp(power))     
                    np.exp((y * w * x.T)[0, 0])
                    if power < -500:
                        jw.append((1/sigma**2) * (w * w.T)[0, 0] - power)
                    else:
                        jw.append((1/sigma**2) * (w * w.T)[0, 0] + np.log(1 + 1 / np.exp(power)))
                w = w - r * grad

            # print objective
            print("Epoch = %s   J(w) = %1.4f" % (epoch, np.mean(jw)))

            # at the end of each epoch, append w to weights
            weights.append(deepcopy(w))

        # 3. return weights
        self.weights = weights

        return self


class Bagging(BaseClf):
    def __init__(self, clfs, n_bags=3, n_samples_frac=0.1, n_features_frac=1):
        super().__init__(clfs=clfs, n_bags=n_bags, n_samples_frac=n_samples_frac, n_features_frac=n_features_frac)

    @timeit('Fit Bagging')
    def fit(self, data=None):
        """
        Returns
        ---------
        clfs : list
            contains the trained classifiers
        """
        # get params
        n_bags = self.n_bags
        clfs = self.clfs
        n_samples_frac = self.n_samples_frac
        n_features_frac = self.n_features_frac
        print(f"n_bags={n_bags}  sample_frac={n_samples_frac}")

        # for each epoch, boostrap a subsample, then train a clf
        clfs_trained = []
        for n_bag in range(1, n_bags+1):
            print("n_bag: %s / %s" % (n_bag, n_bags))
            # bootstrap data
            data_subsample = shuffle_samples(data, seed=n_bag, n_samples_frac=n_samples_frac)
            with suppress_stdout():
                clf = random.sample(clfs, 1)[0]
                clf.fit(data_subsample)
            clfs_trained.append(clf)
            
        self.clfs = clfs_trained

    def predict(self, data=None):
        clfs = self.clfs
        y_preds_bags =[]
        for clf in clfs:
            weights = clf.weights[-1]
            with suppress_stdout():
                y_preds_bags.append(clf.predict(data = data, weights = weights)['y_preds'])

        # make final y_preds
        y_preds = []
        for preds in zip(*y_preds_bags):
            y_pred = stats.mode(preds)[0][0]
            y_preds.append(y_pred)

        # return
        if 'y' not in data.columns:
            return {'y_preds': y_preds}
        else:
            ys = data['y']
            return self.score_pred(ys, y_preds)


class SVM(LinearClf):
    """
    Attributes
    -----------
    weights : csr_matrix, shape = (1, n_features) 

    data : DataFrame, shape = (n_samples, n_features)

    r: float. learning rate

    c: float. tradeoff between regularizer and loss

    n_epoch: int. max epoch
    """
    def __init__(self, r=1, c=1, n_epoch=1):
        super().__init__(r=r, c=c, n_epoch=n_epoch)

    @timeit('Fit SVM')
    def fit(self, data = None):
        """
        Parameters
        -------------
        fpath: str. 
            file directory of trainning set

        data: DataFrame. 
            tranining set, needed only if fpath not given

        Returns
        --------
        self: object
        """
        r = self.r
        c = self.c
        n_epoch = self.n_epoch

        assert (r <= 1), "Learning rate must be no more than one!"
        print("Training SVM: r = %s   c = %s   n_epoch = %s" % (r, c, n_epoch))

        # get training data
        n_samples, n_features = get_n_samples_features_from_df(data)
        self.data_train = data

        # 1. initialize weight, r
        w = init_weight(n_features)
        r_0 = r

        # 2. for epoch = 1...T
        weights = []
        for epoch in range(1, n_epoch + 1):
        # shuffle traning set
            data = shuffle_samples(data, seed=epoch)

        # for each example, update weight
            r = r_0 / epoch
            assert all(data.index == range(n_samples)), "data_train.index != range(n_samples)!"
            jw = []
            for i in range(n_samples):
                y = data.loc[i, 'y']
                x = data.loc[i, 'x']
                assert (w.shape == x.shape), "dim(w) != dim(x)"
                assert (type(y) == np.int64), "type(y) != np.int64"
                sign = (y * w * x.T)[0, 0]
                if sign <= 1:
                    w = (1 - r) * w + r * c * y * x
                else:
                    w = (1 - r) * w
                jw.append((0.5 * w * w.T)[0, 0] + c * max(0, 1 - sign))

            # print objective
            print("Epoch = %s   J(w) = %1.4f" % (epoch, np.mean(jw)))

            # at the end of each epoch, append w to weights
            weights.append(deepcopy(w))

        # 3. return weights
        self.weights = weights
        #print("--------------------------")

        return self