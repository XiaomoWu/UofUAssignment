from scipy import stats
from utils import ld, sv, transform_and_save_input_as_pickle, make_submission_data, suppress_stdout
from classifiers import BaseClf, KNN, NB, Perceptron, Logistic, Bagging, SVM

# Prepare data ---------------------------
# transform and save as pickle
print("Reading input data, transforming them into sparse matrix, and saving as pickle")
#transform_and_save_input_as_pickle()

# read the pickle files back
data_train = ld('data_train')
data_train_compact = ld('data_train_compact')

data_test = ld('data_test')
data_test_compact = ld('data_test_compact')

data_eval = ld('data_eval')
data_eval_compact = ld('data_eval_compact')

# run KNN --------------------------------
knn = KNN(k = 25)
knn.fit(data_train_compact)
print("Predict the TEST set")
knn.predict(data_test_compact)
print("Predict the EVAL set")
y_preds = knn.predict(data_eval_compact)['y_preds']
make_submission_data(y_preds, 'knn.csv')

# run NB ---------------------------------
nb = NB(smooth=10)
nb.fit(data_train_compact)
print("Predict the TEST set")
nb.predict(data_test_compact)
print("Predict the EVAL set")
y_preds = nb.predict(data_eval_compact)['y_preds']
make_submission_data(y_preds, 'nb_1214.csv')

# run Perceptron -----------------------
perceptron = Perceptron(r=0.1, margin=0.01, n_epoch=20)
perceptron.fit(data_train)
print("Predict the TEST set")
perceptron.predict(data_test, perceptron.weights[-1])
print("Predict the EVAL set")
y_preds = perceptron.predict(data_eval, perceptron.weights[-1])['y_preds']
make_submission_data(y_preds, 'perceptron.csv')

# run SVM
svm = SVM(r=0.01, c=1, n_epoch=17)
svm.fit(data_train)
print("Predict the TEST set")
svm.predict(data_test, svm.weights[-1])
print("Predict the EVAL set")
y_preds = svm.predict(data_eval, svm.weights[-1])['y_preds']
make_submission_data(y_preds, 'svm.csv')

# run Logistic -----------------------------
logistic = Logistic(r=0.01, sigma=100, n_epoch=10)
logistic.fit(data_train)
print("Predict the TEST set")
logistic.predict(data_test, logistic.weights[-1])
print("Preidict the EVAL set")
y_preds = logistic.predict(data_eval, logistic.weights[-1])['y_preds']
make_submission_data(y_preds, 'logistic.csv')

## run Bagging ------------------------------
# Cross validation
#for n_epoch in [20]:
#    for n_samples_frac in [0.25, 0.5]:
#        for n_bags in [5]:
#            print("n_epoch=%s   n_samples_frac=%s   n_bags=%s" % (n_epoch, n_samples_frac, n_bags))
#            clf = Perceptron(r=0.1, margin=0.01, n_epoch=n_epoch)
#            bagging = Bagging(clf, n_bags=n_bags, n_samples_frac=n_samples_frac)        
#            with suppress_stdout():
#                bagging.fit(data_train)
#            print("Predict the TEST set")
#            y_preds = bagging.predict(data_test)['y_preds']

# #get final outcome
#clfs = [Perceptron(r=0.1, margin=0.01, n_epoch=20)]
#bagging = Bagging(clfs, n_bags=50, n_samples_frac=0.3)        
#bagging.fit(data_train)
#print("Predict the TEST set")
#y_preds = bagging.predict(data_test)['y_preds']
#make_submission_data(y_preds, 'bagging.csv')

# Bagging: training
perceptron = Perceptron(r=0.1, margin=0.01, n_epoch=20)
perceptron.fit(data_train)

logistic = Logistic(r=0.01, sigma=100, n_epoch=10)
logistic.fit(data_train)

nb = NB(smooth=10)
nb.fit(data_train_compact)

# Bagging: predict the test
y_preds_perceptron = perceptron.predict(data_test, perceptron.weights[-1])['y_preds']
y_preds_logistic = logistic.predict(data_test, logistic.weights[-1])['y_preds']
y_preds_nb = nb.predict(data_test_compact)['y_preds']
y_preds_bag = [y_preds_perceptron, y_preds_logistic, y_preds_nb]

y_preds = []
for preds in zip(*y_preds_bag):
    y_pred = stats.mode(preds)[0][0]
    y_preds.append(y_pred)
y_preds = np.asarray(y_preds)

ys = data_test['y']
BaseClf.score_pred(ys, y_preds)

# make submission
y_preds_perceptron = perceptron.predict(data_eval, perceptron.weights[-1])['y_preds']
y_preds_logistic = logistic.predict(data_eval, logistic.weights[-1])['y_preds']
y_preds_nb = nb.predict(data_eval_compact)['y_preds']
y_preds_bag = [y_preds_perceptron, y_preds_logistic, y_preds_nb]

y_preds = []
for preds in zip(*y_preds_bag):
    y_pred = stats.mode(preds)[0][0]
    y_preds.append(y_pred)
y_preds = np.array(y_preds)

make_submission_data(y_preds, 'ensemble.csv')
