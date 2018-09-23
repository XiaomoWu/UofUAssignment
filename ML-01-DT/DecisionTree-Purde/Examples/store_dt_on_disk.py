#!/usr/bin/env python

### IMPORTANT:  This only works in Python 2.x

import DecisionTree
import pickle

training_datafile = "training.dat"

dt = DecisionTree.DecisionTree( training_datafile = training_datafile,
                                entropy_threshold = 0.1,
                                debug1 = 1                          
                              )
dt.get_training_data()

root_node = dt.construct_decision_tree_classifier()

pickle.dump(dt, open('dt.db', 'w'))

pickle.dump(root_node, open('root_node.db', 'w'))

