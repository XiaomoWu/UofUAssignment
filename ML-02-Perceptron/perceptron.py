from collections import OrderedDict
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dir = "C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-02-Perceptron/dataset/"
os.chdir(dir)

class Perceptron:
    def __init__(self):
        pass

    # train 
    def train(self, train_fpath = None, train_data = None, type = 'simple', eta = 1, init_weight = 'random', margin = 0, epoch_n = 1, verbose = True):
        # import training set
        if train_data is None:
            self.feature_n, self.train_data_n, self.train_data = self._get_data(train_fpath)
        else:
            self.train_data = train_data
            self.train_data_n = train_data.shape[0]
            self.feature_n = train_data.shape[1] - 1

        # initialize weight
        w_0, b_0 = self._init_weight(self.feature_n, type = init_weight)

        # train perceptron
        if type == 'simple':
            self.weight = self._train_simple(w_0, b_0, eta = eta, epoch_n = epoch_n, verbose = verbose)
        elif type == 'decay':
            self.weight = self._train_decay(w_0, b_0, eta = eta, epoch_n = epoch_n, verbose = verbose)
        elif type == 'margin':
            self.weight = self._train_margin(w_0, b_0, eta = eta, margin = margin, epoch_n = epoch_n, verbose = verbose)
        elif type == 'average':
            self.weight = self._train_average(w_0, b_0, eta = eta, epoch_n = epoch_n, verbose = verbose)
        elif type == 'aggresive':
            self.weight = self._train_aggresive(w_0, b_0, margin = margin, epoch_n = epoch_n, verbose = verbose)
        #return self.weight

    # test perceptron
    def test(self, test_fpath = None, test_data = None, weight = None, test_type = None):
        if test_data is None:
            self.test_data = self._get_data(test_fpath)[2]
        else:
            self.test_data = test_data

        self.test_data_n = self.test_data.shape[0]

        if weight is None:
            weight = self.weight[-1]
        else:
            weight = weight
        if test_type == 'average':
            w = weight['a']
            b = weight['ba']
        else:
            w = weight['w']
            b = weight['b']

        hit_n = 0
        
        for i in range(self.test_data_n):
            y_i = self.test_data.iloc[i, 0]
            x_i = self.test_data.iloc[i, 1:]
            if y_i * (np.dot(w, x_i) + b) >= 0:
                hit_n += 1

        acc = hit_n / self.test_data_n
        self.accuracy = acc
        #print('Accuracy is %s (%s/%s)' % (acc, hit_n, self.test_data_n))

        return acc
            
    # cross validation
    def cv(self, type, init_weight, eta = None, test_type = None, margin = None, epoch_n = 10, verbose = False):
        cv1 = self._get_data('CVSplits/training00.data')[2] # 150
        cv2 = self._get_data('CVSplits/training01.data')[2] # 150
        cv3 = self._get_data('CVSplits/training02.data')[2] # 150
        cv4 = self._get_data('CVSplits/training03.data')[2] # 150
        cv5 = self._get_data('CVSplits/training04.data')[2] # 150

        cv = [cv1, cv2, cv3, cv4, cv5]

        cv_acc = []
        for i in range(5):
            cv_temp = deepcopy(cv)
            test = cv_temp.pop(i)
            train = cv_temp
            train = pd.concat(train, ignore_index = True)

            # train epoch_n times
            self.train(train_data = train, type = type, eta = eta, init_weight = init_weight, margin = margin, epoch_n = epoch_n, verbose = verbose)

            # test on the fifth fold
            weight = self.weight[-1]
            acc = self.test(test_data = test, weight = weight, test_type = test_type)
            print('CV %s, acc: %1.4f' % (i, acc))

            cv_acc.append(acc)
        
        avg_cv_acc = np.mean(cv_acc)
        print('Avg acc: %1.4f \n' % (avg_cv_acc))
        return np.mean(avg_cv_acc)


    # Simple perceptron
    def _train_simple(self, w_0, b_0, eta, epoch_n, verbose):
        w = deepcopy(w_0) # w_0 is an array (mutable)
        b = b_0 # b is a float (immutable)
        weight_list = [{'epoch': 1, 't': 0, 'mistake_n': 0, 'w': deepcopy(w), 'b': b}]


        for epoch in range(1, epoch_n + 1):
            t = 0
            mistake_n = 0
            index = [i for i in range(self.train_data_n)]

            np.random.shuffle(index)

            for i in index:
                t += 1
                y_i = self.train_data.iloc[i, 0]
                x_i = np.array(self.train_data.iloc[i, 1:])        
                
                y_pred = 1 if np.dot(x_i, w) >= 0 else -1

                if y_i * y_pred <= 0:
                    mistake_n += 1
                    w += eta * y_i * x_i
                    b += eta * y_i

                weight_list.append({'epoch': epoch, 't': t, 'mistake_n': mistake_n, 'w': deepcopy(w), 'b': deepcopy(b)})
            if verbose == True:
                print('Train epoch: %s, Mistake #: %s' % (epoch, mistake_n))

        return weight_list

    # Decaying perceptron
    def _train_decay(self, w_0, b_0, eta, epoch_n, verbose):
        w = deepcopy(w_0)
        b = b_0
        weight_list = [{'epoch': 1, 't': 0, 'mistake_n': 0, 'w': deepcopy(w), 'b': b}]

        for epoch in range(1, epoch_n + 1):
            t = 0
            mistake_n = 0
            index = [i for i in range(self.train_data_n)]

            np.random.shuffle(index)

            for i in index:
                t += 1
                y_i = self.train_data.iloc[i, 0]
                x_i = self.train_data.iloc[i, 1:]           
                y_pred = 1 if np.dot(x_i, w) >= 0 else -1


                if y_i * y_pred <= 0:
                    mistake_n += 1
                    w += (eta / t) * y_i * x_i
                    b += (eta / t) * y_i

                weight_list.append({'epoch': epoch, 't': t, 'mistake_n': mistake_n, 'w': deepcopy(w), 'b': deepcopy(b)})
            if verbose == True:
                print('Train epoch: %s, Mistake #: %s' % (epoch, mistake_n))

        return weight_list

    
    # margin perceptron
    def _train_margin(self, w_0, b_0, eta, margin, epoch_n, verbose):
        w = w_0
        b = b_0
        weight_list = [{'epoch': 1, 't': 0, 'mistake_n': 0, 'w': deepcopy(w), 'b': b}]

        for epoch in range(1, epoch_n + 1):
            t = 0
            mistake_n = 0
            index = [i for i in range(self.train_data_n)]
            np.random.shuffle(index)

            for i in index:
                t += 1
                y_i = self.train_data.iloc[i, 0]
                x_i = self.train_data.iloc[i, 1:]           
                y_pred = 1 if np.dot(x_i, w) >= 0 else -1

                if y_i * y_pred <= margin:
                    mistake_n += 1
                    w += (eta / t) * y_i * x_i
                    b += (eta / t) * y_i

                weight_list.append({'epoch': epoch, 't': t, 'mistake_n': mistake_n, 'w': deepcopy(w), 'b': deepcopy(b)})

            if verbose == True:
                print('Train epoch: %s, Mistake #: %s' % (epoch, mistake_n))
        return weight_list

    # average perceptron
    def _train_average(self, w_0, b_0, eta, epoch_n, verbose):
        w = deepcopy(w_0) 
        b = b_0 

        a = deepcopy(w_0)
        ba = b_0
        weight_list = [{'epoch': 1, 't': 0, 'mistake_n': 0, 'w': deepcopy(w), 'b': b, 'a': 0, 'ba': 0}]

        for epoch in range(1, epoch_n + 1):
            t = 0
            mistake_n = 0
            index = [i for i in range(self.train_data_n)]

            np.random.shuffle(index)

            for i in index:
                t += 1
                y_i = self.train_data.iloc[i, 0]
                x_i = self.train_data.iloc[i, 1:]      
                y_pred = 1 if np.dot(x_i, w) >= 0 else -1
            
                if y_i * y_pred <= 0:
                    mistake_n += 1
                    w += eta * y_i * x_i
                    b += eta * y_i

                a += w
                ba += b
                weight_list.append({'epoch': epoch, 't': t, 'mistake_n': mistake_n, 'w': deepcopy(w), 'b': b, 'a': deepcopy(a), 'ba': ba})
            if verbose == True:
                print('Train epoch: %s, Mistake #: %s' % (epoch, mistake_n))
        return weight_list

    # aggresive perceptron
    def _train_aggresive(self, w_0, b_0, margin, epoch_n, verbose):
        w = deepcopy(w_0) 
        b = b_0
        weight_list = [{'epoch': 1, 't': 0, 'mistake_n': 0, 'w': deepcopy(w), 'b': b}]

        for epoch in range(1, epoch_n + 1):
            t = 0
            mistake_n = 0
            index = [i for i in range(self.train_data_n)]

            np.random.shuffle(index)

            for i in index:
                t += 1
                y_i = self.train_data.iloc[i, 0]
                x_i = self.train_data.iloc[i, 1:]      
                y_pred = 1 if np.dot(x_i, w) >= 0 else -1
            
                if y_i * y_pred <= margin:
                    mistake_n += 1
                    eta = (margin - y_i * np.dot(w, x_i)) / (np.dot(x_i, x_i) + 1)
                    w += eta * y_i * x_i
                    b += eta * y_i

                weight_list.append({'epoch': epoch, 't': t, 'mistake_n': mistake_n, 'w': deepcopy(w), 'b': deepcopy(b)})
            if verbose == True:
                print('Train epoch: %s, Mistake #: %s' % (epoch, mistake_n))
        return weight_list

    # OUTPUT: a dict with keys of "w" and "b". 
    # e.g. {'w': [0, 0, 0], 'b': 0}
    def _init_weight(self, n, type):
        if type == 'zero':
            w = np.zeros(n)
            b = 0
            return w, b
        elif type == 'random':
            w_and_b = 0.02 * np.random.random_sample(n + 1, ) - 0.01
            w = w_and_b[:-1]
            b = w_and_b[-1]
            return w, b

    # import train/test
    # OUTPUT: list of dicts or pandas
    # e.g. [{'label': 1, 1: 12, 3: 4}] 
    def _get_data(self, fpath, output_format = 'pandas'):
        with open(fpath) as f:
            lines = f.readlines()

        feature_n = 0
        train_data = []
        train_data_n = len(lines)

        for line in lines:
            line = line.strip().split()
            label = int(line.pop(0))
            line = dict([(int(i.split(':')[0]), float(i.split(':')[1])) for i in line])
            line_max_feature_n = max(list(line))
            feature_n = max(feature_n, line_max_feature_n)
            line[0] = label
            line = OrderedDict(sorted(line.items()))
            train_data.append(line)

        if output_format == 'dict':
            data_dict = []
            for line in train_data:
                line['label'] = line.pop(0)
                line.move_to_end('label', last = False)
                data_dict.append(line)
            return feature_n, train_data_n, data_dict
        elif output_format == 'pandas':
            data_pd = []
            for line in train_data:
                valid_feature_index = list(line)[:-1]
                for i in range(1, feature_n + 1):
                    if i not in valid_feature_index:
                        line[i] = 0
                line = OrderedDict(sorted(line.items()))
                line['label'] = line.pop(0)
                line.move_to_end('label', last = False)
                data_pd.append(line)
            data_pd = pd.DataFrame(data_pd)
            return feature_n, train_data_n, data_pd

    # plot epoch vs. acc
    def plot(self):
        epoch_n = max([i['epoch'] for i in self.weight])
        test_data = self._get_data('diabetes.dev')[2]

        acc_list = []
        for epoch in range(1, epoch_n + 1):
            w, b = [(w['w'], w['b']) for w in self.weight if w['epoch'] == epoch][-1]
            weight = {'w': w, 'b': b}
            acc = self.test(test_data = test_data, weight = weight)
            acc_list.append(acc)
            print('Plot Epoch: %s, Acc: %1.4f' % (epoch, acc))
        plt.plot(acc_list)
        plt.show()

# demo
p = Perceptron()
"""
# CV: simple
# pick eta == 0.01
np.random.seed(42)
for eta in [1, 0.1, 0.01]:
    print('Learning rate: %s' % eta)
    p.cv(type = 'simple', eta = eta, init_weight = 'random')
# CV: decay
# pick eta == 0.01
np.random.seed(42)
for eta in [1, 0.1, 0.01]:
    print('Learning rate: %s' % eta)
    p.cv(type = 'decay', eta = eta, init_weight = 'random')

# CV: margin
np.random.seed(42)
for eta in [1, 0.1, 0.01]:
    for margin in [1, 0.1, 0.01]:
        print('Learning rate: %s' % eta)
        print('Margin: %s' % margin)
        p.cv(type = 'margin', eta = eta, margin = margin, init_weight = 'random')

# CV: average
np.random.seed(42)
for eta in [1, 0.1, 0.01]:
    print('Learning rate: %s' % eta)
    p.cv(type = 'average', test_type = 'average', eta = eta, init_weight = 'random')

# CV: aggresive
np.random.seed(42)
for margin in [1, 0.1, 0.01]:
    print('margin: %s' % margin)
    p.cv(type = 'aggresive', margin = margin, init_weight = 'random')

# Number of Updates
np.random.seed(42)
p.train(train_fpath = 'diabetes.train', eta = 1, type = 'simple', init_weight = 'random', epoch_n = 20)
p.train(train_fpath = 'diabetes.train', eta = 0.01, type = 'decay', init_weight = 'random', epoch_n = 20)
p.train(train_fpath = 'diabetes.train', eta = 1, margin = 0.01, type = 'margin', init_weight = 'random', epoch_n = 20)
print(np.sum([w['mistake_n'] for w in p.weight if w['t'] == 750]))

p.train(train_fpath = 'diabetes.train', eta = 0.1, type = 'average', init_weight = 'random', epoch_n = 20)
p.train(train_fpath = 'diabetes.train', eta = 0.1, type = 'aggresive', init_weight = 'random', epoch_n = 20)
"""

