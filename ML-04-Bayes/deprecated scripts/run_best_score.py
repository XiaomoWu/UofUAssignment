from classifier import SVM, Logistic, NB, Tree, SVMTree
from utils import cv, get_data

## cross validation for SVM
#r = 0.1
#c = 10
#n_epoch = 15
#print("SVM: r = %s    c = %s    epoch = %s" % (r, c, n_epoch))
#clf = SVM()
#clf.fit('data/train.liblinear', n_epoch = n_epoch, r = r, c = c)
#clf.predict(get_data('data/test.liblinear')[2])

# #cross validation for Logistic
# r = 
# s = 
# n_epoch = 
#print("Logistic: r = %s   sigma = %s    n_epoch = %s" % (r, sigma, n_epoch))
#clf = Logistic()
#clf.fit('data/train.liblinear', n_epoch = n_epoch, r = r, sigma = sigma)
#clf.predict(get_data('data/test.liblinear')[2])

## cross validation for NB
#smooth = 1
#print('NB: smooth = %s' % smooth)
#clf = NB()
#clf.fit('data/train.liblinear', smooth = 1)
#clf.predict(get_data('data/test.liblinear')[2])

# corss validation for SVMTree
r = 
c = 
d = 
n_epoch = 
print("SVMTree: r = %s    c = %s    d = %s    epoch = %s" % (r, c, d, n_epoch))
clf = SVMTree()
clf.fit('data/train.liblinear', r = r, c = c, d = d, n_epoch = n_epoch)
clf.predict(get_data('data/test.liblinear')[2])