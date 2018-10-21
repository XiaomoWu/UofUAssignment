rules = ner.seed_rules
rules['class'].unique()
instances = ner.train_data
o = ner._apply_rules(ner.train_data, ner.seed_rules, 'np')

l = ['a', 'b', 'c']
d = dict(zip(l, 1))
unique_class = ner.seed_rules['class'].unique()
dict((k, 0) for k in unique_class)
o['class'].value_counts()
labeled_instances = o.loc[o['class'] != '']

labeled_instances = ner._apply_rules(ner.train_data, ner.seed_rules, 'np')
induce = ner._induce_rules(labeled_instances, 'context')

train_data

