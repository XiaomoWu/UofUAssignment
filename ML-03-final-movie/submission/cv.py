"""Functions for cross validation
"""
from utils import *

def cv_baseclf(cls, param_grid, data, index_cv_train, index_cv_test):
    """
    Parameters
    ------------
    cls : Classifier (not instance!)
    """
    for param in make_grid(param_grid):
        #loop through each split
        results_cv = []

        # do cross validation
        for idx_train, idx_test in zip(index_cv_train[0:1], index_cv_test[0:1]):
            d_train = data.loc[idx_train,]
            d_train.index = range(d_train.shape[0])
            
            d_test = data.loc[idx_test,]
            d_test.index = range(d_test.shape[0])

            # initialize classifier
            clf = cls(**param)

            # train classifier
            clf.fit(data = d_train)

            # predict 
            results_cv.append(clf.predict(data = d_test))


        score_cv_baseclf(results_cv, param)#cv(NB, param_grid_nb)


def cv_linearclf(cls, param_grid, data, index_cv_train, index_cv_test):
    """
    Parameters
    ------------
    cls : Classifier (not instance!)
    """
    for param in make_grid(param_grid):
        #loop through each split
        results_cv = []

        # do cross validation
        for idx_train, idx_test in zip(index_cv_train[0:1], index_cv_test[0:1]):
            d_train = data.loc[idx_train,]
            d_train.index = range(d_train.shape[0])
            d_test = data.loc[idx_test,]
            d_test.index = range(d_test.shape[0])

            # initialize & train classifier
            clf = cls(**param)
            clf.fit(data = d_train)

            weights_list = clf.weights
            assert (param['n_epoch'] == len(weights_list)), "n_epoch != len(weights_list)"

            # for each epoch make a prediction
            results = []
            for weights in weights_list:
                with suppress_stdout():
                    re = clf.predict(data = d_test, weights = weights)
                results.append(re)

            results_cv.append(results)

        score_cv_linearclf(results_cv, param)


def score_cv_baseclf(results_cv, params):
    """
    Parameters
    ------------
    results_cv : list
        len(results_cv) == n_splits
        each element of results_cv is a "dict" like {'acc', 'p', 'r', 'f'}

    params : dict
        {'r': 1, 'c': 1}

    Returns
    --------
    score : dict
        {'acc':, 'precision':, 'recall':, 'f':}
    """
    keys = results_cv[0].keys()

    score = {}
    for key in keys:
        if key != 'y_preds':
            score[key] = np.mean([r[key] for r in results_cv])

    print("CV summary: %s" % params)
    print(str(score))
    print('--------------------------')


def score_cv_linearclf(results_cv, params):
    """
    Parameters
    ------------
    results_cv : list
        len(results_cv) == n_splits
        each element of results_cv can be "dict"(n_epoch = 1) or "list of dicts" (n_epoch > 1)

    params : dict
        {'r': 1, 'c': 1}

    Returns
    --------
    score : dict
        {'acc':, 'precision':, 'recall':, 'f':}
    """
    n_epoch = len(results_cv[0]) # number of n_epoch 
    for epoch in range(n_epoch):
        result_cv = [r[epoch] for r in results_cv] # [{'acc':}, {'acc':}]
        keys = results_cv[0][0].keys()

        epoch += 1
        score = {'epoch': epoch}
        for key in keys:
            if key != 'y_preds':
                score[key] = np.mean([r[key] for r in result_cv])

        print("CV summary: %s" % params)
        print(str(score))
    print('--------------------------')
