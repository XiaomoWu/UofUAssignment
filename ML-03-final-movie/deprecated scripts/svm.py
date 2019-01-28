import os
from copy import deepcopy
import pandas as pd
import numpy as np
from utils import *
from base import *

class SVM(LinearClf):
    """
    Attributes
    -----------
    weights : csr_matrix, shape = (1, n_features) 

    data_train : DataFrame, shape = (n_samples, n_features)

    r: float. learning rate

    c: float. tradeoff between regularizer and loss

    n_epoch: int. max epoch
    """
    @timeit('Fit SVM')
    def fit(self, fpath = None, data = None):
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
        data_train, n_samples, n_features = self.init_input(fpath, data)
        self.data_train = data_train

        # 1. initialize weight, r
        w = init_weight(n_features)
        r_0 = r

        # 2. for epoch = 1...T
        weights = []
        for epoch in range(1, n_epoch + 1):
        # shuffle traning set
            data = shuffle_samples(data_train, seed=epoch)

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
   
