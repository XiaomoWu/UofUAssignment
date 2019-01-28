from copy import deepcopy
from scipy.sparse import *
import numpy as np
import pandas as pd
import time
import os
from base import LinearClf
from utils import *


# set working path
home_dir = os.path.expanduser("~")
os.chdir(home_dir + '/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-03-final-movie')


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
    def fit(self, fpath = None, data = None):
        # get params
        r = self.r
        margin = self.margin
        n_epoch = self.n_epoch
        print("Training Perceptron: r = %s   margin = %s    n_epoch = %s" % (r, margin, n_epoch))

        # get training data
        data_train, n_samples, n_features = self.init_input(fpath, data)
        self.data_train = data_train

        # initialize weight
        w = init_weight(n_features)
        wa = 0 # average weights
        r_0 = r
        weights = []

        for epoch in range(1, n_epoch+1):
            # shuffle traning set
            data = shuffle_samples(data_train, seed=epoch)
            assert all(data.index == range(n_samples)), "data_train.index != range(n_samples)!"

            # for each example, update weight
            mistake_n = 0
            #r = r_0 / epoch
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

            # print mistake # for each epoch
            print('Epoch = %s   Mistake # = %s' % (epoch, mistake_n))

            # at the end of each epoch, append wa to weights
            weights.append(deepcopy(wa))

        # 3. return weights
        self.weights = weights
        return self


