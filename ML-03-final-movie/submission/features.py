from scipy.sparse import *
import pandas as pd
import spacy
import time
import os
import io


# import NLP library
nlp = spacy.load('en_core_web_lg')

# set working path
home_dir = os.path.expanduser("~")
os.chdir(home_dir + '/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-03-final-movie')

# generate my feature ----
def make_my_feature():
    # training set ----
    start_time = time.time()
    with io.open('data/raw-data/train.rawtext', encoding = 'utf8') as f:
        train_data = {'id':[], 'text':[]}
        lines = f.readlines()
        for i, line in enumerate(lines):
            train_data['id'].append(i)
            train_data['text'].append(nlp(line))
        train_data = pd.DataFrame(train_data)
    print("--- %s mimutes ---" % (time.time() - start_time) / 60)
    # add label
    label = pd.read_pickle('label_train.pkl')
    train_data['label'] = label
    # generate sentence embedding
    train_data['vector'] = train_data['text'].apply(lambda r: csr_matrix(r.vector))

    # save to external file
    train_data2_sm = train_data[['id', 'label', 'vector']]
    train_data2_sm.to_pickle('train_data2_sm.pkl')

    # test set ----
    start_time = time.time()
    with io.open('data/raw-data/test.rawtext', encoding = 'utf8') as f:
        test_data = {'id':[], 'text':[]}
        lines = f.readlines()
        for i, line in enumerate(lines):
            test_data['id'].append(i)
            test_data['text'].append(nlp(line))
        test_data = pd.DataFrame(test_data)
    print("--- %s mimutes ---" % ((time.time() - start_time) / 60))
    # add label
    label = pd.read_pickle('label_test.pkl')
    test_data['label'] = label
    # generate sentence embedding
    test_data['vector'] = test_data['text'].apply(lambda r: csr_matrix(r.vector))
    # save to external file
    test_data2_sm = test_data[['id', 'label', 'vector']]
    test_data2_sm.to_pickle('test_data2_sm.pkl')

    # eval set ----
    start_time = time.time()
    with io.open('data/raw-data/eval.rawtext', encoding = 'utf8') as f:
        eval_data = {'id':[], 'text':[]}
        lines = f.readlines()
        for i, line in enumerate(lines):
            eval_data['id'].append(i)
            eval_data['text'].append(nlp(line))
        eval_data = pd.DataFrame(eval_data)
    print("--- %s mimutes ---" % ((time.time() - start_time) / 60))
    # add label
    eval_data['label'] = 1
    # add example_id
    eval_data['example_id'] = pd.read_pickle('id_eval.pkl')
    # generate sentence embedding
    eval_data['vector'] = eval_data['text'].apply(lambda r: csr_matrix(r.vector))
    # save to external file
    eval_data2_sm = eval_data[['id', 'example_id', 'label', 'vector']]
    eval_data2_sm.to_pickle('eval_data2_sm.pkl')


