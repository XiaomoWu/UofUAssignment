import copy

DATA_DIR = 'C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/NLP-02-morphology/'

class PARSER:
    def __init__(self):
        self.dict = self._get_dict()
        self.rules = self._get_rules()
        self.test = self._get_test()
        
    # main parse method
    def parse(self):

        for word in self.test:
            self.output = set()
            self._parse_one(word)
            self._format_output()
            for o in self.output:
                print(o)
            print()

    # main method called by parse
    def _parse_one(self, word):
        dict_match = self._search_dict(word)
        
        # IN dict
        if dict_match:
            for d in dict_match:
                pos = d['pos']
                self.output.add('%s %s ROOT=%s SOURCE=%s' % (word, pos, word, 'dictionary'))

        # NOT in dict
        else:
            # generate parent 
            parent = {'word_input': word,
                      'pos_input': '',
                      'word': word,
                      'pos': ''}

            # call _analyzer recursively
            return self._analyzer(parent)

    # recursive analyzer
    def _analyzer(self, parent):
        
        word_input = parent['word_input']
        pos_input = parent['pos_input']
        word = parent['word']
        pos = parent['pos']

        # search rules for word
        rule_match = self._search_rules(word)

        # if NOT in rule, return NONE and DEFAULT 
        if not rule_match:
            self.output.add('%s %s ROOT=%s SOURCE=%s' % (word_input, 'noun', word_input, 'default'))

        # if IN rule, search word_new in dict
        if rule_match:
            pos_input_new = copy.deepcopy(pos_input)
            for rm in rule_match:
                word_new = rm['word_new']
                pos_new = rm['pos_new']
                pos_pred = rm['pos'] # pos predicted by rule

                if not pos_input_new:
                    pos_input = pos = pos_pred

                parent_new = {'word_input': word_input,
                      'pos_input': pos_input,
                      'word': word_new,
                      'pos': pos_new}

                # continue ONLY if pos and its parent NOT conflict
                if pos_pred == pos:
                    # IN dict
                    dict_match = self._search_dict(word_new)
                    if dict_match:
                        for d in dict_match:
                            self.output.add('%s %s ROOT=%s SOURCE=%s' % (word_input, pos_input, word_new, 'morphology'))
                            
                    # NOT in dict
                    else:
                         self._analyzer(parent_new)
            

    # _search_rules: look up the rules, return {'pos', 'word_new', 'pos_new'}
    def _search_rules(self, word):
        rule_match = []
        for rule in rules:
            type = rule['type']
            affix = rule['affix']
            affix_n = len(affix)
            replace = rule['replace']

            if type == 'prefix':
                if word[ : affix_n] == affix:
                    word_new = replace + word[affix_n:]
                    pos_new = rule['pos_origin']
                    pos = rule['pos_derived']
                    rule_match.append({'word_new': word_new, 'pos_new': pos_new, 'pos': pos})

            elif type == 'suffix':
                if word[-affix_n : ] == affix:
                    word_new = word[ : -affix_n] + replace
                    pos_new = rule['pos_origin']
                    pos = rule['pos_derived']
                    rule_match.append({'word_new': word_new, 'pos_new': pos_new, 'pos': pos})

        return rule_match

    
    # _search_dict: look up dict, return a list of (True, matched), or (False, original)
    def _search_dict(self, word):
        dict_match = []

        for d in self.dict:
            if word == d['word']:
                dict_match.append({'word': word, 'pos': d['pos']})
        
        return dict_match


    # import rules
    def _get_rules(self):
        with open(DATA_DIR + 'rules.txt') as f:
            lines = f.readlines()
        rules = []
        for line in lines:
            line = line.strip().lower().split()
            if len(line) == 7:
                rules.append({'type': line[0], 'affix': line[1], 'replace': line[2].replace('-', ''), 'pos_origin': line[3], 'pos_derived': line[5]})
            else:
                raise Exception('Unexpected dict!')
        return rules

    # import dict
    def _get_dict(self):
        with open(DATA_DIR + 'dict.txt') as f:
            lines = f.readlines()
        dict = []
        for line in lines:
            line = line.strip().lower().split()
            if len(line) == 2:
                dict.append({'word': line[0], 'pos': line[1], 'root': line[0]})
            elif len(line) == 4:
                dict.append({'word': line[0], 'pos': line[1], 'root': line[3]})
            else:
                raise Exception('Unexpected dict!')
        return dict

    # import test
    def _get_test(self):
        with open(DATA_DIR + 'test.txt') as f:
            return [line.strip() for line in f.readlines()]

    # format output
    def _format_output(self):
        output = list(self.output)
        if len(output) > 1:
            self.output = [o for o in output if o.find('SOURCE=default') < 0]




x = PARSER()
x.parse()

