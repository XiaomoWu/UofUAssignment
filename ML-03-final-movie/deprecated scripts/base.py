from utils import *

class BaseClf:
    def __init__(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    def score_pred(self, ys, y_preds):
        """report {acc, pre, recall, f}
        """
        tp, fp, fn = 0, 0, 0
        for y, y_pred in zip(ys, y_preds):
            if (y == y_pred) & (y_pred == 1): tp += 1
            if (y != y_pred) & (y_pred == 1): fp += 1
            if (y != y_pred) & (y_pred == -1): fn += 1

        acc = (sum(ys == y_preds) / len(ys))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print("Accuracy: %1.3f    Precision: %1.3f    Recall: %1.3f     F: %1.3f" % (acc, precision, recall, f))

        return {'acc': acc, 'precision': precision, 'recall': recall, 'f': f, 'y_preds': y_preds}
        
    def init_input(self, fpath=None, data=None):
        """
        get data, n_samples, n_features
        """
        # get training data
        if data is None: # if data not given, read from file
            data, n_samples, n_features = get_data(fpath)
        else: # else get data directly
            n_samples, n_features = get_n_samples_features_from_df(data)

        return data, n_samples, n_features

class LinearClf(BaseClf):

    @timeit("Pred")
    def predict(self, data, weights):
        """
        Parameters
        ------------
        data : DataFrame, columns = ['y', 'x'] or ['x']
        weights : sparse matrix, shape = (1, n_features)

        Returns
        --------
        {y_pred, acc, p, r, f} - if 'y' in data.columns
        {y_pred} - if 'y' NOT in data.columns
        """

        # make prediction
        assert (all(data.index == range(data.shape[0]))), "all(data.index != range(data.shape[0])"
        from scipy.sparse import csr_matrix
        assert (type(weights) == csr_matrix), "type(weight) != csr_matrix"

        y_preds = []
        for i in range(data.shape[0]):
            x = data.loc[i, 'x']
            if (weights * x.T)[0, 0] > 0:
                y_preds.append(1)
            else:
                y_preds.append(-1)

        # return
        if 'y' not in data.columns:
            return {'y_preds': y_preds}
        else:
            ys = data['y']
            return self.score_pred(ys, y_preds)

