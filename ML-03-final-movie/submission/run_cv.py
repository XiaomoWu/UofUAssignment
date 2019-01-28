import numpy as np
import itertools
from utils import *
from cv import *
from classifiers import *


data_train = ld('data_train')
data_train_compact = ld('data_train_compact')
data_test = ld('data_test')
data_test_single_marix = ld('data_train_compact')

index_cv5_train = ld('index_cv5_train')
index_cv5_test = ld('index_cv5_test')
index_cv3_train = ld('index_cv3_train')
index_cv3_test = ld('index_cv3_test')

param_grid_svm = [{
    'r': [0.01],
    'c': [1],
    'n_epoch': [40]}]

param_grid_logistic = [{
    'r': [0.1],
    'sigma': [1000],
    'n_epoch': [20]}]

param_grid_perceptron = [{
    'r': [0.1],
    'n_epoch': [3],
    'margin': [0.01]}]

param_grid_knn = [{
    'k': [3]}]

param_grid_nb = [
    {'smooth': [1]}]


cv_linearclf(SVM, param_grid_svm, data_train, index_cv5_train, index_cv5_test)
#cv_linearclf(Logistic, param_grid_logistic, data_train, index_cv5_train, index_cv5_test)
#cv_linearclf(Perceptron, param_grid_perceptron, data_train, index_cv5_train, index_cv5_test)


