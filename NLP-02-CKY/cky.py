import sys
import getopt
import numpy as np
import pandas as pd

# CKY Parser class
class CKY():
    def __init__(self, pcfg = 'pcfg.txt', sentences = 'sentences.txt'):
        self.pcfg = self._get_pcfg(pcfg)
        self.sentence_list = self._get_sentence_list(sentences)

    # parse: main method to parse all sentences
    def parse(self, prob = False, verbose = False):
        for sentence in self.sentence_list:
            chart = self._parse_one(sentence)
            self._output(chart, prob, verbose)

    # _parse_one: parse one sentence, called by parse
    # sentence: the sentence to parse
    # n = number of words in the sentence
    # chart: 2D table to store all intermediate results
    def _parse_one(self, sentence):
        n = len(sentence)
        chart = pd.DataFrame(index=sentence, columns = sentence)

        # c: column number
        # r: row number
        # iterate over each column
        for c in range(n):
            # initialize [c,c]
            cell_cc = self._add_position(sentence[c], c, c)
            chart.iloc[c,c] = self._look_up_pcfg(cell_cc, c, c, 1)

            # iterate from row c-1 to row 1
            for r in range(c - 1, -1, -1):
                cell_rc = []
                # iterate for each possible span within words[r, c]
                for k in range(r, c):
                    cell_rk = chart.iloc[r, k]
                    cell_kc = chart.iloc[k + 1, c]

                    # cell_rkc: it's a "Cartesian product" of all posible rule combinations (may not be valid) by cell_rk and cell_kc, e.g. [{'NP_0_0 VP_1_1': {'prob': 0.3, 'right': 'trust_0_0 fish_1_1'}},...]
                    cell_rkc = [{' '.join([i['left'], j['left']]): {'prob': i['prob'] * j['prob'], 'right': ' '.join([i['right'], j['right']])}} for i in cell_rk for j in cell_kc]
                    # for all possible rules in cell_rkc, select those are valid (has entry in pcfg).
                    for cell in cell_rkc:
                        cell_rkc_new = self._look_up_pcfg(list(cell.keys())[0], r, c, list(cell.values())[0]['prob'])
                        if cell_rkc_new:
                            cell_rc += cell_rkc_new

                # assign cell_rc to chart
                chart.iloc[r, c] = cell_rc
        
        return(chart)

    # INPUT: the chart of a parsed sentence
    # OUTPUT: see requirement
    def _output(self, chart, prob, verbose):
        sentense = ' '.join(list(chart.index))
        n = len(chart)
        if prob == True:
            n_s = 1
        else:
            n_s = len([i for i in chart.iloc[0, n - 1] if i['left'][0].split('_')[0] == 'S'])
        print('PARSING SENTENCE: %s' % sentense)
        print('NUMBER OF PARSES FOUND: %s' % n_s)
        print('TABLE:')
        for r in range(n):
            for c in range(r, n):
                if verbose == True:
                    if prob == True:
                        o = sorted(['%s[%s] (%1.4f)' % (i['left'], i['right'], i['prob']) for i in chart.iloc[r, c]])

                        # for cell[1,N], if there exits multiple S, only select the one with highest prob.
                        if r == 0 and c == n - 1:
                            all_s = [i for i in o if i[0:2] == 'S_']
                            all_s_prob = [float(i.split('(')[1].split(')')[0]) for i in all_s]
                            max = np.max(all_s_prob)
                            not_max_prob_s = [all_s[i] for i in [i for i, j in enumerate(all_s_prob) if j != max]]

                            o = sorted([i for i in o if i not in not_max_prob_s])

                        o = ' '.join(o)
                    else:
                        o = ' '.join(sorted(['%s ' % (i['left']) for i in chart.iloc[r, c]]))
                else:
                    if prob == True:
                        o = sorted(['%s(%1.4f)' % (self._remove_position(i['left']), i['prob']) for i in chart.iloc[r, c]])

                        # for cell[1,N], if there exits multiple S, only select the one with highest prob.
                        if r == 0 and c == n - 1:
                            all_s = [i for i in o if i[0:2] == 'S(']
                            less_prob_s = sorted(all_s, reverse = True)[1:]
                            o = [i for i in o if i not in less_prob_s]
                        o = ' '.join(o)
                    else:
                        o = ' '.join(sorted(['%s ' % (self._remove_position(i['left'])) for i in chart.iloc[r, c]]))
                # remove additional white spaces
                o = ' '.join(o.split())
                if len(o) == 0:
                    o = '-'
                print('cell[%s,%s]: %s' % (r, c, o))
        print()

    # INPUT: fpath of text sentences
    # OUTPUT: a list of lists, e.g. [['Will', 'you', 'marry', 'me'], ...]
    def _get_sentence_list(self, fpath):
        #fpath = 'sentences.txt'
        with open(fpath) as f:
            lines = f.readlines()
        sentence_list = [line.strip().split() for line in lines]
        return sentence_list


    # INPUT: fpath of pcfg
    # OUTPUT: a pandas dataframe, e.g. [{'left': 'S', 'right': 'NP', 'prob': 0.8}, ....]
    def _get_pcfg(self, fpath):
        #fpath = 'pcfg.txt'
        with open(fpath) as f:
            lines = f.readlines()
        pcfg = []
        for line in lines:
            left = line.split('->')[0].strip()
            right = ' '.join(line.split('->')[1].split()[0:-1])
            prob = float(line.strip().split('->')[1].split()[-1].strip())
            pcfg.append({'left': left, 'right': right, 'prob': prob})
        return pd.DataFrame(pcfg)

    # _look_up_pcfg: given the right of rule, return the left
    # INPUT: 
    # - right: a string of right, e.g. 'Utah_0_0' or 'NP_0_0 VP_1_3'
    # - start: start position in sentence
    # - end: end position in sentence
    # OUTPUT: a list whose key is the left and whose item is right & prob, e.g. [{'NP_2_5': 'art_2_2 NP_3_5', 'prob': 0.2}, ...]
    # - ATTEN! the prob is the updated prob
    def _look_up_pcfg(self, right, start, end, prob):
        right_pure = self._remove_position(right)
        match_rules = self.pcfg.loc[right_pure == self.pcfg['right']].to_dict('records')
        if match_rules:
            for m in match_rules:
                m['left'] = self._add_position(m['left'], start, end)
                m['right'] = right
                m['prob'] = round(m['prob'] * prob, 4)
            return match_rules

    # add position: given the start and end position, add subscript to symbols
    # INPUT: e.g. 'NP VP', (2, 3)
    # OUTPUT: e.g. 'NP_2_3 NP_2_3'
    def _add_position(self, symbols, start, end):
        return ' '.join([s + '_' + str(start) + '_' + str(end) for s in symbols.split()])

    # delete position: remove subscripts of symbols
    # INPUT: e.g. 'NP_2_3 NP_2_3'
    # OUTPUT: e.g. 'NP VP'
    def _remove_position(self, symbols):
        return ' '.join([s.split('_')[0] for s in symbols.split()])

#import os
#dir_path = 'C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/NLP-02-CKY/'
#os.chdir(dir_path)

#cky = CKY()
#cky.parse(prob = 'prob', verbose = 'verbose')
#cky.parse(prob = 'prob')

if __name__ == '__main__':
    args = sys.argv[1:]
    pcfg = args[0]
    sentences = args[1]
    prob = True if '-prob' in args else False
    verbose = True if '-verbose' in args else False
    cky = CKY(pcfg = pcfg, sentences = sentences)
    cky.parse(verbose = verbose, prob = prob)

    
    