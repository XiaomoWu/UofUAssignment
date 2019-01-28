from utils import *
from base import *
from classifiers import *

class Bagging(BaseClf):
    def __init__(self, clf=Pereptron, n_bag=3):
        super().__init__(clf=clf, n_bags=n_bags)


    @timeit('Fit Bagging')
    def fit_and_predicate(self, data_train=None, data_test=None):
        # get params
        n_bags = self.n_bags
        clf = self.clf

        # for each epoch, boostrap a subsample, then make a prediction
        y_preds_bags = []
        for n_bag in range(1, n_bags+1):
            print("n_bag: %s / %s" % (n_bag, n_bags))
            data = shuffle_samples(data_train, seed=n_bag, frac=0.1)
            print("data n_samples %s" % data.shape)
            clf.fit(data = data_train)
            weights = clf.weights[0]
            y_preds_bags.append(clf.predict(data = data_test, weights = weights)['y_preds'])

        # make final y_preds
        y_preds = []
        for preds in zip(*y_preds_bags):
            y_preds.append(np.mean(preds))

        # return
        if 'y' not in data_test.columns:
            return {'y_preds': y_preds}
        else:
            ys = data_test['y']
            return self.score_pred(ys, y_preds)

data_train = ld('data_train')
data_test = ld('data_test')
bagging = Bagging(clf = Perceptron(r=1), n_bags = 1)
bagging.fit_and_predicate(data_train, data_test)
