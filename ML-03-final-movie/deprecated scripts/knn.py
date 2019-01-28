from operator import itemgetter
from base import BaseClf
from utils import *

class KNN(BaseClf):
    @timeit('Fit KNN')
    def fit(self, data=None):

        # get training data
        data_train = data
        n_samples, n_features = data.loc[0, 'x'].shape

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


        ## return
        #if 'y' not in data.columns:
        #    return {'y_preds': y_preds}
        #else:
        #    ys = data.loc[0, 'y']
        #    # check
        #    assert (ys.shape[0] == len(y_preds)), "ys.shape[0] != len(y_preds)"

        #    return self.score_pred(ys, y_preds)
         
    @staticmethod
    def normalize(x):
        """normalize x (divided by its norm)

        Parameters
        ------------
        x : csr_matrix, shape = (1, n_features)
        """

        norm = np.sqrt((x * x.T)[0, 0])
        return x / norm

data_train = ld('data_train_compact')
data_test = ld('data_test_compact')

for k in [1]:
    knn= KNN(k=k, method='cos') # computing time for euclidean is prohibitive. Transforming to dense then use sklearn is also prohibitive.
    knn.fit(data = data_train)
    knn.predict(data_test)

