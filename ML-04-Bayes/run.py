import os
import time
import pickle
import multiprocessing as mp
from classifier import SVM, Logistic, NB, Tree, SVMTree
from utils import *

os.chdir(os.path.expanduser("~") + '/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-04-Bayes')

# get train/test data
data_train = get_data('data/train.liblinear')[2]
data_test = get_data('data/test.liblinear')[2]

# Fit SVM
r = 0.1
c = 10
n_epoch = 15
print("SVM: r = %s    c = %s    epoch = %s" % (r, c, n_epoch))
clf = SVM()
with suppress_stdout():
    clf.fit(data = data_train, n_epoch=n_epoch, r=r, c=c)
clf.predict(data_test)
print('------------------------------------------------------')

 #cross validation for Logistic
r = 1
sigma = 1000
n_epoch = 6
if 2*r/sigma**2 <= 1:
    print("Logistic: r = %s   sigma = %s    n_epoch = %s" % (r, sigma, n_epoch))
    clf = Logistic()
    clf.fit(data = data_train, n_epoch = n_epoch, r = r, sigma = sigma)
    clf.predict(data_test)
print('------------------------------------------------------')

# cross validation for NB
smooth = 1
print('NB: smooth = %s' % smooth)
clf = NB()
clf.fit(data = data_train, smooth = 1)
clf.predict(data_test)
print('------------------------------------------------------')



#def make_trees(fpath, i, chunk_size, max_depth):
#    process_name = mp.current_process().name
#    print(f"Process name: {process_name}")
#    clf = SVMTree()
#    clf.fit(fpath, n_trees = range(i * chunk_size, (i+1) * chunk_size), max_depth = max_depth)
#    rules = clf.rules
#    with open(f'test_depth{max_depth}_part{i}.pkl', 'wb') as f:
#        pickle.dump(rules, f)

#n_trees = 200
#n_worker = 8
#max_depth = 10
#chunk_size = round(n_trees / n_worker)

# #train trees
#if __name__ == '__main__':

#    processes = []
#    fpath = 'data/train.libinear'
#    for i in range(n_worker):
#        p = mp.Process(target = make_trees, args = (fpath, i, chunk_size, max_depth))
#        processes.append(p)
#        p.start()
        
#    for p in processes:
#        p.join()


# #load back the trees and generate transformed X
#rules_200 = []
#for i in range(8):
#    with open('test_depth10_part%s.pkl' % i, 'rb') as f:
#        rules = pickle.load(f)
#        rules_200.extend(rules)
#svmtree = SVMTree()
#x_trans_test = svmtree.transform(get_data('data/test.liblinear')[2], rules_200)
#with open('test_depth10_X_trans_200_trees.pkl', 'wb') as f:
#    pickle.dump(x_trans_test, f)


# load back rules
print("Training SVMTree")
print("Here I directly load back the trained 200 tress, you never want to re-train it yourself!")
with open('train_depth10_X_trans_200_trees.pkl', 'rb') as f:
    x_trans_train = pickle.load(f)
with open('test_depth10_X_trans_200_trees.pkl', 'rb') as f:
    x_trans_test = pickle.load(f)

# cross validation for SVMTree
n_epoch = 10
r = 0.01
c = 10
print("SVMTree: r = %s    c = %s    epoch = %s" % (r, c, n_epoch))
with suppress_stdout():
    clf = SVM()
clf.fit(data = x_trans_train, n_epoch=n_epoch, r=r, c=c)
clf.predict(x_trans_test)



