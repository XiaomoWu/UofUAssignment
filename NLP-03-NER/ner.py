'''
1) "Perm , Russia", how to parse?
2) exclude previous learnt rule from rules_temp
'''

import os
import pandas as pd
from copy import deepcopy

#import os
dir_path = 'C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/NLP-03-NER/'
os.chdir(dir_path)

class NER:
    def __init__(self):
        self.prob = 0.8
        self.freq = 5
        self.seed_rules = self._get_rules('seedrules.txt')
        self.train_data = self._get_train('train.txt')
        self.test_data = self._get_train('test.txt')
        self.unique_class = self.seed_rules['class'].unique()

        self.np_rules = pd.DataFrame()
        self.context_rules = pd.DataFrame()
        self.decision_rules = pd.DataFrame()

    def train(self):
        train_data = self.train_data
        for iter in [1]:
            # step 2
            if iter == 1:
                np_rules_temp = deepcopy(self.seed_rules)
                self.np_rules = deepcopy(np_rules_temp)
            else:
                np_rules_temp = self.np_rules.loc[self.np_rules['iter'] == (iter - 1)]
            # step 3
            train_data = self._apply_rules(train_data, np_rules_temp, 'np')
            # step 4, 5
            context_rules_temp = self._induce_rules(train_data, 'context', iter)
            self.context_rules = pd.concat([self.context_rules, context_rules_temp], ignore_index = True)

            # print newly learned context rules
            print('\nITERATION #%s: NEW CONTEXT RULES\n' % iter)
            for i in context_rules_temp.index.tolist():
                (word, label, prob, freq) = context_rules_temp.loc[i, ['word', 'class', 'prob', 'freq']]
                print('CONTEXT Contains(%s) -> %s (prob=%1.3f ; freq=%1.0f)' % (word, label, prob, freq))

            # step 6
            train_data = self._apply_rules(train_data, context_rules_temp, 'context')
            # step 7
            np_rules_temp = self._induce_rules(train_data, 'np', iter)
            # step 8
            self.np_rules = pd.concat([self.np_rules, np_rules_temp], sort = False, ignore_index = True)

            # print newly learned np rules
            print('\nITERATION #%s: NEW SPELLING RULES\n' % iter)
            for i in np_rules_temp.index.tolist():
                (word, label, prob, freq) = np_rules_temp.loc[i, ['word', 'class', 'prob', 'freq']]
                print('SPELLING Contains(%s) -> %s (prob=%1.3f ; freq=%1.0f)' % (word, label, prob, freq))

        self.decision_rules = pd.concat([self.seed_rules, self.np_rules, self.context_rules], sort = False, ignore_index = True)
        self.train_data = train_data
        '''
        '''


    def _apply_rules(self, instances, rules, type):
        '''
        instances are train/test data to be labeled
        type: 'context' or 'np'
        '''

        # for each rule in rules, apply it
        for i in rules.index.tolist():
            rule_word = rules.loc[i, 'word']
            rule_class = rules.loc[i, 'class']

            # for each instance in instances
            for j in instances.index[instances['class'] == ''].tolist():
                if instances.loc[j, 'class'] == '':
                    np_or_context = instances.loc[j, type]
                    if rule_word in np_or_context:
                        instances.loc[j, 'class'] = rule_class
            
        return instances

    def _induce_rules(self, instances, type, iter = None):
        rules = {}
        labeled_instances = instances.loc[instances['class'] != '']
        if type == 'np':
            old_rule_words = (self.np_rules['word'].unique() if self.np_rules.shape[0] > 0 else [])
        elif type == 'context':
            old_rule_words = (self.context_rules['word'].unique() if self.context_rules.shape[0] > 0 else [])

        for i in labeled_instances.index:
            np_or_context = labeled_instances.loc[i, type]
            label = labeled_instances.loc[i, 'class']
            for word in np_or_context:
                # exclude old rules
                if word not in old_rule_words:
                    if word not in rules:
                        rules[word] = dict((k, 0) for k in self.unique_class)
                        rules[word]['freq'] = 0
                    rules[word][label] += 1
                    rules[word]['freq'] += 1
                    for k in self.unique_class:
                        prob = rules[word][k] / rules[word]['freq']
                        rules[word].update({k + '_prob': prob})
        df = pd.DataFrame(rules).transpose()

        # select rules according to freq and prob
        induce_rules = []
        for k in self.unique_class:

            d = df.loc[(df[k] >= self.freq) & (df[k + '_prob'] >= self.prob)]
            for i in d.index:
                word = i
                label = k
                prob = d.loc[i, k + '_prob']
                freq = d.loc[i, k]
                induce_rules.append({'iter': iter, 'type': type, 'word': word, 'class': k, 'prob': prob, 'freq': freq})

        df = pd.DataFrame(induce_rules)

        # sort & output rules
        df = df.sort_values(by = ['class', 'prob', 'freq', 'word'], ascending = [True, False, False, True]).groupby('class').head(2).sort_values(by = ['prob', 'freq', 'word'], ascending = [False, False, True])

        return df

    def _get_train(self, fpath):
        with open(fpath) as f:
            train = []
            lines = f.read().split('\n\n')
            for line in lines:
                type1 = line.split('\n')[0].split(':')[0].strip().lower()
                type1_arg = line.split('\n')[0].split(':')[1].strip().split(' ')
                type1_arg = [t.strip() for t in type1_arg if t != ',']
                
                type2 = line.split('\n')[1].split(':')[0].strip().lower()
                type2_arg = line.split('\n')[1].split(':')[1].strip().split(' ')
                type2_arg = [t.strip() for t in type2_arg if t != ',']
                train.append({type1: type1_arg, type2: type2_arg, 'class': ''})

        df = pd.DataFrame(train)
        df.to_csv('temp.csv', index = False)
        return df

    def _get_rules(self, fpath):
        with open(fpath) as f:
            lines = f.readlines()
        seed_rules = []

        print('SEED DECISION LIST\n')

        for line in lines:
            print('%s (prob = -1.000 ; freq = -1)' % (line.strip('\n')))
            left = line.split('->')[0]
            right = line.split('->')[1]
            label = right.strip()
            type = left.split(' ')[0].strip()
            if type == 'SPELLING':
                type = 'np'
            elif type == 'CONTEXT':
                type = 'context'
            else:
                type = 'unknow'
            word = left.split(' ')[1].split('(')[1].strip(' )')
            seed_rules.append({'iter': 0, 'type': type, 'word': word, 'class': label, 'prob': 1, 'freq': -1})
        df = pd.DataFrame(seed_rules).reindex(['iter', 'type', 'word', 'class', 'prob', 'freq'], axis = 1)



        return df

ner = NER()
#ner._get_rules('seedrules.txt')
#ner._get_train('train.txt')
#ner.train_dat
#ner.test_data
#labeled_instances = ner._apply_rules(ner.train_data, ner.seed_rules, 'np')
#induce = ner._induce_rules(labeled_instances, 'context')
ner.train()
#print(ner.decision_rules)
#ner.train_data.to_csv('df.csv', index = False)


