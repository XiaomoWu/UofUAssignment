from base import BaseClf
from utils import *

class NB(BaseClf):
    """
    Attributes
    -----------
    jll : DataFrame, shape = (n_samples, n_features-1) 
        jll contains priors and likelihood
    unique_ys : list
        unique labels
    """
    @timeit('Fit NB')
    def fit(self, fpath = None, data = None):
        smooth = self.smooth

        print("Training NB: smooth = %s" % (smooth))

        # get dat
        data_train, n_samples, n_features = self.init_input(fpath, data)
        self.data_train = data_train

        # get priors
        y = pd.Series(data_train.loc[0, 'y'])
        priors = y.value_counts(normalize = True)
        unique_ys = priors.index.values # [-1, 1]

        # update likelihood
        lls = [] # log-likihoods, list of P(x_i | y)
        # for each y, update its posteriors
        for unique_y in unique_ys: 
            prior = priors[y]
            x = data_train.loc[0, 'x']
            index_x_given_y = y.loc[y == unique_y].index.values
            x_given_y = x[index_x_given_y,]
            x_rowsum_given_y = np.squeeze(np.asarray(x_given_y.sum(axis=0), dtype = np.int64))
            l = x_rowsum_given_y / x_rowsum_given_y.sum() # liklihoods, shape = (1, n_features)
            ll = np.log(l + smooth)# smooth, then take log liklihood
            lls.append(ll)
        
        # jll: DataFrame, joint liklihood. 
        #   columns = [liklihoods, priors]
        #   index = [-1, 1]
        #   shape = (2, 2)
        jll = pd.DataFrame({'lls': lls, 'priors': priors}) 

        self.jll = jll
        self.unique_ys = unique_ys


    @timeit('Pred NB')
    def predict(self, data):
        # get param
        x = data.loc[0, 'x']
        y = pd.Series(data.loc[0, 'y'])
        jll = self.jll

        # check input
        assert (x.shape[1] == len(jll['lls'].iloc[0])), "test data's n_features != train data" # n_features should be identical

        # make prediction
        compare = {} # store prob of each unique_y
        for unique_y in self.unique_ys:
            prior = jll.loc[unique_y, 'priors']
            ll = jll.loc[unique_y, 'lls']
            posteri = prior * x.dot(ll) # posteri if classified as unique_y
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

data_train = ld('data_train_compact')
data_test = ld('data_test_compact')
for smooth in [1, 3, 5, 7, 9]:
    nb = NB(smooth = smooth)
    nb.fit(data=data_train)
    nb.predict(data = data_test)


