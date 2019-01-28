import os
import sys
import time
import pickle
import itertools
from copy import deepcopy
from contextlib import contextmanager
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix, rand

os.chdir(os.path.expanduser("~") + '/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-03-final-movie')

# CONSTANTS
N_FEATURES = 74483

def timeit(task_name=None):
    def decorator(f):
        def wrapper(*args, **kwargs):
            ts = time.time()
            result = f(*args, **kwargs)
            tp = (time.time() - ts) / 60
            print("%s uses %0.1f min" % (task_name, tp))
            print("-----------------------------")
            return result
        return wrapper
    return decorator


@timeit('Get data')
def get_data(fpath, each_row_as_a_matrix=True):
    """
    Returns
    --------
    data: DataFrame, shape = (n_samples, 2), colums = ['y', 'x']
        column 'x' includes the bias term (1)
        each element of x is a ROW vector (csr_matrix)
        type(y) == np.int64
   
    n_samples: int. # samples.

    n_features: int. # features
    """

    # read features as dict
    ys = []
    xs = []
    with open(fpath) as f:
        for line in f:
            # get y
            y = int(line.strip().split(' ')[0])
            y = -1 if y == 0 else 1 
            ys.append(y)

            # get x
            x = line.strip().split(' ')[1:] # here x is a list of str
            x = {int(d.split(':')[0]) : int(d.split(':')[1]) for d in x}
            xs.append(x)

    # determin n_features
    # n_features = max(dict.key()) + 2
    # "1" for the bias, "1" for the case where the original feature index starts from 0.
    #n_features = 0
    #for x in xs:
    #    n_features = max(n_features, max(x.keys()))
    #n_features += 2 
    n_features = N_FEATURES
    n_samples = len(xs)

    # transform dict into sparse vector
    if each_row_as_a_matrix == True:
        xs_sparse = []
        for x in xs:
            x = dict2sparse(x, n_features)
            xs_sparse.append(x)
        data = pd.DataFrame({'y': ys, 'x': xs_sparse})
    else:
        xs_sparse = lil_matrix((n_samples, n_features))
        for i, x in enumerate(xs):
            for k, v in x.items(): # x = {1:1, 2:3, 9:1,...}
                xs_sparse[i, k] = v
        xs_sparse = xs_sparse.tocsr()
        data = pd.DataFrame({'y': [np.array(ys)], 'x': xs_sparse})

    return data, n_samples, n_features


def get_index_cv(data, n_splits=5):
    """
    Parameters
    ------------
    data : DataFrame
        can't be a compact one, i.e., each row should be a single example

    Returns
    --------
    index_train : ndarray
        each element is the train indices of a split

    index_test : ndarray
        each element is the test indices of a split
    """
    n_samples = data.shape[0]
    fold_sizes = np.ones(n_splits, dtype=np.int) * (n_samples // n_splits)
    fold_sizes[:(n_samples % n_splits)] += 1
    np.random.seed(42)
    index = np.array(data.index)
    np.random.shuffle(index)

    index_cv_train = []
    index_cv_test = []
    current = 0
    for fold_size in fold_sizes:
        current, stop = current, current + fold_size
        index_test = index[current:stop]
        index_train = list(set(index) - set(index_test))

        index_cv_test.append(index_test)
        index_cv_train.append(index_train)

        current = stop

    return index_cv_train, index_cv_test
        

def dict2sparse(dict, n_features):
    """Convert a dict to a sparse matrix

    Parameters
    ------------
    dict : dict, 
        the keys and values must be int/float. e.g. {1:1, 2:2, 3:3}

    n_features : int
        the length of the output sparse vector (including bias)

    Returns
    ---------
    vec_sparse : crr_matrix, shape = (1, n_features) (row vector)
        the 1st element of the vector is always 1 (the bias)
    """
    #vec = np.zeros(n_features)
    #vec[0] = 1 # the bias
    #for k, v in dict.items():
    #    vec[k+1] = v
    #vec_sparse = csr_matrix(vec)

    vec_lil = lil_matrix((1, n_features))
    vec_lil[0, 0] = 1
    for k, v in dict.items():
        vec_lil[0, k+1] = v
    vec_sparse = vec_lil.tocsr()
    
    return vec_sparse
#x = get_data('data/data-splits/data.train')[0]


def get_n_samples_features_from_df(df):
    """
    Parameters
    ------------
    df : DataFrame. columns = ['y', 'x']
        'x' must be sparse matrix
        'x' already contains the bias (1)

    Returns
    --------
    n_samples : int
    n_features : int
    """
    # if all examples are in one row
    if df.shape[0] == 1:
        n_samples, n_features = df.loc[0, 'x'].shape
    # if each row is an example
    else:
        n_samples = df.shape[0]
        n_features = df.iloc[0]['x'].shape[1]

    return n_samples, n_features


def init_weight(n_features, all_zeros=False):
    """
    Parameters
    ------------
    n_features : int.
        # features, including the bias

    all_zero : boolean (default=False)
        If True, all weights set to zero.

    Returns
    --------
    weights : sparse matrix, shape = (1, n_features), row vector 
    """
    if all_zeros:
        density = 0
    else: 
        density = 1
    np.random.seed(42)
    weights = rand(1, n_features, density=density, format='csr', dtype=np.float) * 0.01
    return weights
#w = init_weight(5)


def shuffle_samples(data, seed=42,  n_samples_frac=1, replace=True):
    """
    Parameters
    -------------
    data : DataFrame.
        usually columns = ['y', 'x']
       
    seed : int. Random seed

    frac : np.float
        1 = return all rows

    replace : boolean (default=True)
        Whether to replace in sampling.
    
    Returns
    ---------
    data_shuffle : DataFrame.
        the shuffled dataset
    """

    data_shuffle = data.sample(frac = n_samples_frac, random_state = seed, replace = replace)
    data_shuffle.index = range(data_shuffle.shape[0])
    return data_shuffle


def shuffle_features(data, seed=42, n_features_frac=1):
    """
    Parameters
    ------------
    data : DataFrame, shape = (n_samples, 1), columns = ['x']
    """
    pass

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

@timeit('Saving data')
def sv(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


@timeit('Loading data')
def ld(name):
    with open(name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
    return obj


def make_grid(param_grid):
    """
    Parameters
    ------------
    param_grid : list
        each element is a grid, like:
        param_grid = [{
            'r': [0.1, 1],
            'c': [1, 10],
            'n_epoch': [2]}]

    Returns
    --------
    """
    for p in param_grid:
        items = sorted(p.items())
        keys, values = zip(*items)
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            yield params
#list(make_grid(param_grid))


def transform_and_save_input_as_pickle():
    """pickle all input data 
    """
     #create & save data
    data_train = get_data('data/data-splits/data.train')[0]
    data_train_compact = get_data('data/data-splits/data.train', each_row_as_a_matrix=False)[0]

    data_test = get_data('data/data-splits/data.test')[0]
    data_test_compact= get_data('data/data-splits/data.test', each_row_as_a_matrix=False)[0]

    data_eval = get_data('data/data-splits/data.eval.anon')[0]
    data_eval_compact = get_data('data/data-splits/data.eval.anon', each_row_as_a_matrix=False)[0]

    index_cv5_train, index_cv5_test = get_index_cv(data_train, n_splits = 5)
    index_cv3_train, index_cv3_test = get_index_cv(data_train, n_splits = 3)

    sv(data_train, 'data_train')
    sv(data_train_compact, 'data_train_compact')

    sv(data_test, 'data_test')
    sv(data_test_compact, 'data_test_compact')

    sv(data_eval, 'data_eval')
    sv(data_eval_compact, 'data_eval_compact')

    sv(index_cv5_train, 'index_cv5_train')
    sv(index_cv5_test, 'index_cv5_test')
    sv(index_cv3_train, 'index_cv3_train')
    sv(index_cv3_test, 'index_cv3_test')


def make_submission_data(y_preds, fpath):
    """
    Parameters
    ------------
    y_preds : predicted y of the eval set

    fpath : str.
        save dir for the csv file
    """

    # change "-1" in y_preds to "0"
    y_preds = [1 if y > 0 else 0 for y in y_preds]

    # get eval_ids
    eval_ids =[]
    with open('data/data-splits/data.eval.anon.id', 'r') as f:
        for line in f:
            eval_ids.append(line.strip())

    # make submission file
    with open(fpath, 'w') as f:
        f.write('example_id,label\n')
        for id, y_pred in zip(eval_ids, y_preds):
            line = ("%s,%s\n" % (id, y_pred))
            f.write(line)

    print("Write submission file done!")


def set_none_zero(matrix, value=1):
    """Given a sparse matrix, set its non-zero entries to value
    """
    matrix = matrix.tolil()
    rows, cols = matrix.nonzero()
    for row, col in zip(rows, cols):
        matrix[row, col] = value
    return matrix.tocsr()

def get_n_zero_x(matrix):
    """Given a sparse matrix, return the number of non-zeros elements in each row
    """
    n_cols = matrix.shape[1]
    x_colsum = np.squeeze(np.asarray(matrix.sum(axis=1)))
    n_zero_x = n_cols - x_colsum

    return n_zero_x

