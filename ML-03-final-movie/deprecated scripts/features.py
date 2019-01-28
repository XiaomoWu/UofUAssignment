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

# import given features ----
def import_given_feature():
    voc_n = 74481 # vocabulary size
    train_data = []
    with io.open('data/data-splits/data.train', encoding = 'utf8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split()
            label = int(line.pop(0))
            label = -1 if label == 0 else 1
            feature_dict = dict([(int(i.split(':')[0]), float(i.split(':')[1])) for i in line])
            # convert feature to sparse vector
            feature = lil_matrix((1, voc_n))
            for k, v in feature_dict.items():
                feature[0, k-1] = v
            feature = feature.tocsr()
            line = {'id': i, 'label': label, 'vector': feature}
            train_data.append(line)
    train_data = pd.DataFrame(train_data)
    # save 
    train_data['label'].to_pickle('label_train.pkl')
    train_data.to_pickle('train_data.pkl')

    test_data = []
    with io.open('data/data-splits/data.test', encoding = 'utf8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split()
            label = int(line.pop(0))
            label = -1 if label == 0 else 1
            feature_dict = dict([(int(i.split(':')[0]), float(i.split(':')[1])) for i in line])
            # convert feature to sparse vector
            feature = lil_matrix((1, voc_n))
            for k, v in feature_dict.items():
                feature[0, k-1] = v
            feature = feature.tocsr()
            line = {'id': i, 'label': label, 'vector': feature}
            test_data.append(line)
    test_data = pd.DataFrame(test_data)
    # save 
    test_data['label'].to_pickle('label_test.pkl')
    test_data.to_pickle('test_data.pkl')

    eval_data = []
    with io.open('data/data-splits/data.eval.anon', encoding = 'utf8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split()
            label = int(line.pop(0))
            label = -1 if label == 0 else 1
            feature_dict = dict([(int(i.split(':')[0]), float(i.split(':')[1])) for i in line])
            # convert feature to sparse vector
            feature = lil_matrix((1, voc_n))
            for k, v in feature_dict.items():
                feature[0, k-1] = v
            feature = feature.tocsr()
            line = {'id': i, 'label': label, 'vector': feature}
            eval_data.append(line)
    eval_data = pd.DataFrame(eval_data)
    with io.open('data/data-splits/data.eval.anon.id', encoding = 'utf8') as f:
        eval_data_id = [f.strip() for f in f.readlines()]
    eval_data['example_id'] = eval_data_id
    # save 
    eval_data['example_id'].to_pickle('eval_id.pkl')
    eval_data.to_pickle('eval_data.pkl')


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

