from classifier import SVM, Logistic, NB, Tree, SVMTree
from utils import cv, get_data

## cross validation for SVM
#print("Starting Cross Validation for SVM")
#for r in [1e0, 1e-1, 1e-2, 1e-3, 1e-4]: # best r = 0.1
#    for c in [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]: # best c = 10
#        for n_epoch in range(1, 30):
#            print("SVM: r = %s    c = %s    epoch = %s" % (r, c, n_epoch))
#            clf = SVM()
#            cv(clf, r = r, c = c, n_epoch = n_epoch)

# #cross validation for Logistic
#print("Starting Cross Validation for Logistic")
#for r in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]: # best r = 1
#    for sigma in [1e4, 1e3, 1e2, 1e1, 1e0, 1e-1]: # best sigma = 1
#        for n_epoch in range(1, 31, 5):
#            print("Logistic: r = %s   sigma = %s    n_epoch = %s" % (r, sigma, n_epoch))
#            clf = Logistic()
#            cv(clf, r = r, sigma = sigma, n_epoch = n_epoch)

## cross validation for NB
#print("Starting Cross Validation for NB")
#for smooth in [2, 1.5, 1, 0.5]: # best smooth = 1
#    print('NB: smooth = %s' % smooth)
#    clf = NB()
#    cv(clf, smooth = smooth)

