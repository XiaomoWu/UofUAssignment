import os
import sys
from copy import deepcopy
from contextlib import contextmanager
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, rand
import classifier

# os.chdir(os.path.expanduser("~") + '/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-04-Bayes')


def get_data(fpath):
    """
    Returns
    --------
    n_features: int. # features, including bias term (always one as the 1st element)
    n_samples: it. # samples.
    data: DataFrame, colums = ['X', 'y']
    """

    # read features as dict
    with open(fpath) as f:
        lines = f.readlines()
    line_list = []
    n_features = 0
    n_samples = len(lines)
    for line in lines:
        y = int(line.split(' ')[0].strip())
        X = [dict([map(int, tuple(l.strip().split(':')))]) for l in line.split(' ')[1:]]
        n_features = max(n_features, max([list(x.keys())[0] for x in X]))
        line_list.append({'X': X, 'y': y})
    n_features += 1 # include bias term
    
    # convert features from dict to sparse_matrix
    examples = []
    for line in line_list:
        X = np.zeros(n_features, dtype = 'float64')
        y = line['y']
        X[0] = 1 # bias term
        for d in line['X']:
            for k, v in d.items():
                X[k] = v # k >= 2
        examples.append({'X': X, 'y': y})

    # output
    return n_features, n_samples, pd.DataFrame(examples, columns = ['y','X'])
#n_features, n_samples, data = get_data('data/train.liblinear')

def init_weight(n_features):
    """
    Returns
    --------
    weight: array-like, shape(1, n_feature). May be sparse or not.
    """
    np.random.seed(42)
    weight = 0.02 * np.random.random_sample(n_features) - 0.01
    # weight = np.zeros(n_features, dtype = np.float64)
    return weight

def shuffle_samples(data, seed, frac = 1):
    data_shuffle = data.sample(frac = frac, random_state = seed, replace = True)
    return data_shuffle
#x = shuffle_samples(data)

def cv(clf, n_epoch = None, r = None, c = None, sigma = None, smooth = None, d = None):
    """
    Parameters
    -------------
    clf : Classifer object. Could be "SVM()".

    r_0, c, sigma, n_epoch: 
        Parameters to pass to clf to fit the model
    """

    # get data_train
    data_cv = []
    n_features = 0
    for filename in os.listdir('data/CVSplits'):
        n_features_new, n_samples, data = get_data('data/CVSplits/' + filename)
        data_cv.append(data)
        n_features = max(n_features, n_features_new)

    # cross_validation
    acc, recall, p, f = [], [], [], []
    for i in range(5):
        #print("CROSS VALIDATION: fold = %s" % i)
        d = deepcopy(data_cv)
        data_test = d.pop(i)
        data_fit = pd.concat(d, ignore_index = True)

        with suppress_stdout():
            clf.fit(data = data_fit, n_features = n_features, n_epoch = n_epoch, r = r, c = c, smooth = smooth, sigma = sigma)
            re = clf.predict(X = data_test)

        acc.append(re['acc'])
        recall.append(re['r'])
        p.append(re['p'])
        f.append(re['f'])

    print("CROSS VALIDATION SUMMARY: Accuracy: %1.3f    Recall: %1.3f    Precision: %1.3f     F: %1.3f" % (np.mean(acc), np.mean(recall), np.mean(p), np.mean(f)))

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout