import os
from copy import deepcopy
import pandas as pd
import numpy as np
from utils import *

class Logistic(LinearClf):
    """
    Attributes
    -----------
    weights : csr_matrix, shape = (1, n_features) 
    data_train : DataFrame, shape = (n_samples, n_features)
    """

    @timeit('Fit Logistic')
    def fit(self, fpath = None, data = None):
        """
        Parameters
        -------------
        fpath: str. 
            file directory of trainning set

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

        # get training data
        data_train, n_samples, n_features = self.init_input(fpath, data)
        self.data_train = data_train

        # 1. initialize weight, r
        w = init_weight(n_features)
        r_0 = r

        # 2. for each epoch:
        weights = []
        for epoch in range(1, n_epoch + 1):
            # (1) shuffle training set
            data = shuffle_samples(data_train, seed=epoch)

            # (2) update weight
            r = r_0 / epoch # diminishing r
            assert all(data.index == range(n_samples)), "data_train.index != range(n_samples)!"
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

