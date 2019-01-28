import os
import sys
import pandas as pd
from copy import deepcopy

#import os
#dir_path = 'C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/NLP-03-NER/'
#os.chdir(dir_path)

class NER:
    def __init__(self, fpath_seed, fpath_train, fpath_test):
        self.prob = 0.8
        self.freq = 5
        self.seed_rules = self._get_rules(fpath_seed)
        self.train_data = self._get_train(fpath_train)
        self.test_data = self._get_train(fpath_test)
        self.unique_class = self.seed_rules['class'].unique()

        self.np_rules = pd.DataFrame()
        self.context_rules = pd.DataFrame()
        self.decision_rules = pd.DataFrame()

    def train(self):
        # train_data_np: labels only produced by NP rules
        train_data_context = deepcopy(self.train_data)
        train_data_np = deepcopy(self.train_data)

        for iter in [1, 2, 3]:
            # step 2: initialization
            if iter == 1:
                np_rules_temp = deepcopy(self.seed_rules)
                self.np_rules = deepcopy(np_rules_temp)
            else:
                np_rules_temp = self.np_rules.loc[self.np_rules['iter'] == (iter - 1)]

            # step 3: apply NP rules
            train_data_np = self._apply_rules(train_data_np, np_rules_temp)

            # step 4, 5: induce Context rules
            context_rules_temp = self._induce_rules(train_data_np, 'context', iter)

            self.context_rules = pd.concat([self.context_rules, context_rules_temp], ignore_index = True)

            # print newly learned context rules
            print('\nITERATION #%s: NEW CONTEXT RULES\n' % iter)
            for i in context_rules_temp.index.tolist():
                (word, label, prob, freq) = context_rules_temp.loc[i, ['word', 'class', 'prob', 'freq']]
                print('CONTEXT Contains(%s) -> %s (prob=%1.3f ; freq=%1.0f)' % (word, label, prob, freq))

            # step 6: apply context rules
            train_data_context = self._apply_rules(train_data_context, context_rules_temp)

            # step 7, 8: induce NP rules
            np_rules_temp = self._induce_rules(train_data_context, 'np', iter)

            self.np_rules = pd.concat([self.np_rules, np_rules_temp], ignore_index = True)

            # print newly learned np rules
            print('\nITERATION #%s: NEW SPELLING RULES\n' % iter)
            for i in np_rules_temp.index.tolist():
                (word, label, prob, freq) = np_rules_temp.loc[i, ['word', 'class', 'prob', 'freq']]
                print('SPELLING Contains(%s) -> %s (prob=%1.3f ; freq=%1.0f)' % (word, label, prob, freq))

        self.decision_rules = pd.concat([self.seed_rules, self.np_rules, self.context_rules], ignore_index = True)

        # print final decison rules.
        print('\nFINAL DECISION LIST\n')
        for i in self.decision_rules.index.tolist():
            (type, word, label, prob, freq) = self.decision_rules.loc[i, ['type', 'word', 'class', 'prob', 'freq']]
            type = 'SPELLING' if type == 'np' else 'CONTEXT'
            print('%s Contains(%s) -> %s (prob=%1.3f ; freq=%1.0f)' % (type, word, label, prob, freq))

    def test(self):
        test = self._get_train('test.txt')
        rules = deepcopy(self.decision_rules)

        # apply decision rules
        labeled_test = self._apply_rules(test, rules)

        # print test labeling results
        print('\nAPPLYING FINAL DECISION LIST TO TEST INSTANCES\n')
        for i in labeled_test.index.tolist():
            context = ' '.join(labeled_test.loc[i, 'context'])
            np = ' '.join(labeled_test.loc[i, 'np'])
            label = labeled_test.loc[i, 'class']
            label =  label if label != '' else 'NONE'

            print('CONTEXT: %s' % context)
            print('NP: %s' % np)
            print('CLASS: %s\n' % label)

    def _apply_rules(self, instances, rules):
        '''
        instances are train/test data to be labeled
        type: 'context' or 'np'
        '''

        # for each rule in rules, apply it
        for i in rules.index.tolist():
            rule_word = rules.loc[i, 'word']
            rule_class = rules.loc[i, 'class']
            rule_type = rules.loc[i, 'type']


            # for each unlabeled instance
            for j in instances.index[instances['class'] == ''].tolist():
                np_or_context = instances.loc[j, rule_type]
                if rule_word in np_or_context:
                    instances.loc[j, 'class'] = rule_class
            
        return instances

    def _induce_rules(self, instances, type, iter = None):
        rules = []
        labeled_instances = instances.loc[instances['class'] != '']
        if type == 'np':
            old_rule_words = (self.np_rules['word'].unique() if self.np_rules.shape[0] > 0 else [])
        elif type == 'context':
            old_rule_words = (self.context_rules['word'].unique() if self.context_rules.shape[0] > 0 else [])

        for i in labeled_instances.index.tolist():
            np_or_context = labeled_instances.loc[i, type]
            label = labeled_instances.loc[i, 'class']
            for word in np_or_context:
                if word not in old_rule_words: # exclude old rules
                    if word not in [r['word'] for r in rules]: # initial new words
                        d = dict((k, 0) for k in self.unique_class)
                        d['word'] = word
                        d['freq'] = 0
                        rules.append(d)
                    d = [r for r in rules if r['word'] == word][0]
                    d[label] += 1
                    d['freq'] += 1
                    for k in self.unique_class:
                        prob = d[k] / d['freq']
                        d[k + '_prob'] = prob
        
        #print([r for r in rules if r['word'] == 'president'])
        df = pd.DataFrame(rules)

        # select rules according to freq and prob
        induce_rules = []
        for k in self.unique_class:

            d = df.loc[(df[k] >= self.freq) & (df[k + '_prob'] >= self.prob)]
            for i in d.index:
                word = d.loc[i, 'word']
                label = k
                prob = d.loc[i, k + '_prob']
                freq = d.loc[i, 'freq']
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
                type1_arg = [t.strip() for t in type1_arg]
                
                type2 = line.split('\n')[1].split(':')[0].strip().lower()
                type2_arg = line.split('\n')[1].split(':')[1].strip().split(' ')
                type2_arg = [t.strip() for t in type2_arg]
                train.append({type1: type1_arg, type2: type2_arg, 'class': ''})

        df = pd.DataFrame(train)
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
        #df = pd.DataFrame(seed_rules).reindex(['iter', 'type', 'word', 'class', 'prob', 'freq'], axis = 1)
        df = pd.DataFrame(seed_rules).loc[:, ['iter', 'type', 'word', 'class', 'prob', 'freq']]
        return df


if __name__ == '__main__':
    args = sys.argv[1:]
    fpath_seed = args[0]
    fpath_train = args[1]
    fpath_test = args[2]
    ner = NER(fpath_seed, fpath_train, fpath_test)
    ner.train()
    ner.test()
    




