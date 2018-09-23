#!/usr/bin/env python

### IMPORTANT:  This only works in Python 2.x

import DecisionTree
import pickle

dt = pickle.load(open('dt.db'))

root_node = pickle.load(open('root_node.db'))

root_node.display_decision_tree("     ");         

test_sample = ['exercising=>never', 'smoking=>heavy',                   
               'fatIntake=>heavy', 'videoAddiction=>heavy']             

classification = dt.classify(root_node, test_sample)                    

print ("Classification: ", classification)


