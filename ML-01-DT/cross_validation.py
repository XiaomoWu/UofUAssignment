import numpy as np
import pandas as pd
from build_id3 import ID3

DATA_DIR = "C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-01-DT/experiment-data_new/data_new/"

# import validation set
cv1 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold1.csv')
cv2 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold2.csv')
cv3 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold3.csv')
cv4 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold4.csv')
cv5 = pd.read_csv(DATA_DIR + 'CVfolds_new/fold5.csv')
cv = [cv1, cv2, cv3, cv4, cv5]


id3 = ID3()
id3.train_id3(data = cv[0], max_depth = 1)