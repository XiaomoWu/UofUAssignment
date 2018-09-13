# codes to get the results in my report

import numpy as np
import pandas as pd
from ID3 import ID3

# Import & some CONST
DATA_DIR = "C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-01-DT/experiment-data_new/data_new/"

##### Experiments
# 1.(b)
id3 = ID3()
id3.train_id3(fpath = DATA_DIR + 'train.csv')
id3.test_id3(fpath = DATA_DIR + 'train.csv') # the answer is 1

# 1.(c)
id3.test_id3(fpath = DATA_DIR + 'test.csv') # the answer is 1

# 1.(d)
rules = id3.rules
max([len(r['attr']) for r in rules]) # the answer is 6



#### CORSS-VALIDATION

# 2.(a)

# import validation set
cv1 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold1.csv')
cv2 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold2.csv')
cv3 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold3.csv')
cv4 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold4.csv')
cv5 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold5.csv')
cv = [cv1, cv2, cv3, cv4, cv5]

# for each depth in [1~5], run cross-validation and return the average accuracy and sd.
for max_depth in [1, 2, 3, 4, 5, 10, 15]:
    accs = []
    for i in range(5):
        cv_temp = copy.deepcopy(cv)
        test = cv_temp[i]
        del cv_temp[i]
        train = pd.concat(cv_temp, ignore_index = True)

        id3 = ID3()
        id3.train_id3(data = train, max_depth = max_depth, verbose = False)
        accs.append(id3.test_id3(data = test))
    avg_acc = np.mean(accs)
    sd_acc = np.std(accs)
    print('max_depth: %s avg_acc: %s, sd_acc: %s' % (max_depth, avg_acc, sd_acc))

# 2.(b)
# choose a max_depth of 2
id3.train_id3(fpath = DATA_DIR + 'train.csv', max_depth = 2)
id3.test_id3(fpath = DATA_DIR + 'test.csv') # answer is 0.9774


