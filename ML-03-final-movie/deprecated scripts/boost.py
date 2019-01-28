from algo_perceptron import *
from scipy.sparse import *
from copy import deepcopy
import numpy as np
import pandas as pd
import time
import os

# AdaBoost
class AdaBoost:
    def __init__(self):
        pass

    def train(self, algo = 'simple', r = 1, margin = 0, train_data = None, train_fpath = None, epoch_n = 1):
        # import training set
        self.train_data = pd.read_pickle(train_fpath)
        self.train_data_n = self.train_data.shape[0]
        self.feature_n = max(self.train_data['vector'].apply(lambda r: r.shape[1]))

        dist_list = []
        alpha_list = []
        dist_list.append([1 / self.train_data_n] * self.train_data_n)
        classifier_list = []

        for epoch in range(1, epoch_n + 1):
            print('Epoch: %s' % epoch)
            # train weak classifier
            classifier = Perceptron()
            classifier.train(type = algo, r = r, train_data = self.train_data, verbose = False)
            epsilon, y_pred = classifier.predict(test_data = self.train_data, verbose = False)
            classifier_list.append(classifier.weight[-1])
            epsilon = 1 - epsilon
            if_correct = y_pred * self.train_data['label']
            # update dist
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            alpha_list.append(alpha)
            last_dist = dist_list[-1]
            #dist_new = []
            #for i in range(self.train_data_n):
            #    dist_new.append(last_dist[i] * np.exp(- alpha * if_correct[i]))

            dist_new = np.array(last_dist) * np.exp(- alpha * if_correct)
            dist_new = [float(d) / sum(dist_new) for d in dist_new]
            dist_list.append(dist_new)
            print('epsilon: %1.4f     alpha: %1.4f\n' % (epsilon, alpha))

        self.dist_list = dist_list
        self.alpha_list = alpha_list
        self.classifier_list = classifier_list

    def predict(self, test_data = None, test_fpath = None):
        if test_data is None:
            self.test_data = pd.read_pickle(test_fpath)
        else:
            self.test_data = test_data
        self.test_data_n = self.test_data.shape[0]

        # generate H_final
        y_pred = np.array([float(0)] * self.test_data_n)
        classifier_n = len(self.classifier_list)
        for i in range(classifier_n):
            y_pred_i = self.alpha_list[i] * self.predict_one(weight_data = self.classifier_list[i])
            y_pred += y_pred_i
        y_pred = [1 if y >= 0 else -1 for y in y_pred]

        # use H_final to predict
        acc =  1- sum(y_pred != self.test_data['label']) / self.test_data_n
        print('Acc of H_final: %1.4f' % acc)

    def predict_one(self, weight_data = None):
        #weight = weight_data[-1]
        w = weight_data['w']
        b = weight_data['b']
        y_pred_list = []
        for i in range(self.test_data_n):
            y_i = self.test_data.loc[lambda df: df['id'] == i, 'label'].values[0]
            x_i = self.test_data.loc[lambda df: df['id'] == i, 'vector'].values[0]
            y_pred = 1 if (x_i @ w.T)[0, 0] > 0 else -1
            y_pred_list.append(y_pred)
        return np.array(y_pred_list)



ad = AdaBoost()
ad.train(train_fpath = 'train_data2_sm.pkl', epoch_n = 5, algo = 'average', r = 1, margin = 0.1)
ad.predict(test_fpath = 'test_data2_sm.pkl')




