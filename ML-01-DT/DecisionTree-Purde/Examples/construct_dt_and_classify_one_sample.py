#!/usr/bin/env python

import DecisionTree
import sys

training_datafile = "training.dat"

dt = DecisionTree.DecisionTree( training_datafile = training_datafile,
                                entropy_threshold = 0.1,
                                max_depth_desired = 3,
                                debug1 = 1                          
                              )
dt.get_training_data()

#   UNCOMMENT THE FOLLOWING LINE if you would like to see the training
#   data that was read from the disk file:
#dt.show_training_data()

root_node = dt.construct_decision_tree_classifier()

#   UNCOMMENT THE FOLLOWING LINE if you would like to see the decision
#   tree displayed in your terminal window:
#root_node.display_decision_tree("   ")

test_sample = ['exercising=>never', 
               'smoking=>heavy', 
               'fatIntake=>heavy',
               'videoAddiction=>heavy']

classification = dt.classify(root_node, test_sample)

print("\n\n")
print("Classification Results: ", classification)

print("Number of nodes created: ", root_node.how_many_nodes())


###########################   cut  here  #############################

'''
# THE COMMENTED OUT CODE THAT IS SHOWN BELOW IS USEFUL FOR DEBUGGING THE
# PROBABILITY AND THE ENTROPY CALCULATORS:

prob = dt.prior_probability_for_class( 'benign' )
print("prior for benign: ", prob)
prob = dt.prior_probability_for_class( 'malignant' )
print("prior for malignant: ", prob)

prob = dt.probability_for_feature_value( 'smoking', 'heavy')
print(prob)
dt.determine_data_condition()

prob = dt.probability_for_feature_value_given_class('smoking', 'heavy', 'malignant')
print(prob)
prob = dt.probability_for_feature_value_given_class('smoking', 'medium', 'malignant')
print(prob)

prob = dt.probability_of_a_sequence_of_features_and_values(\
          ['smoking=>heavy', 'exercising=>regularly', 'fatIntake=>heavy'])
print(prob)

prob = dt.probability_of_a_sequence_of_features_and_values(\
          ['smoking=>heavy', 'exercising=>regularly', 'fatIntake=>heavy'])
print(prob)

prob = dt.probability_for_sequence_of_features_and_values_given_class(\
   ['smoking=>heavy', 'exercising=>regularly', 'fatIntake=>heavy'],'malignant')
print(prob)

prob = dt.probability_for_a_class_given_feature_value( 'benign', \
                                              'smoking', 'heavy')
print(prob)

prob = dt.probability_for_a_class_given_sequence_of_features_and_values(\
            'benign', ['smoking=>heavy', 'exercising=>regularly', 'fatIntake=>heavy'])
print(prob)

prob = dt.probability_for_a_class_given_sequence_of_features_and_values(\
            'malignant', ['smoking=>heavy', 'exercising=>regularly', 'fatIntake=>heavy'])
print(prob)

ent = dt.class_entropy_on_priors()
print("entroy on priors: ", ent)

ent = dt.class_entropy_for_a_given_feature_and_given_value('smoking','heavy')
print("class entroy on feature 'smoking' and value 'heavy': ", ent)

ent = dt.class_entropy_for_a_given_feature('smoking')
print("class entroy on feature 'smoking': ", ent)

ent = dt.class_entropy_for_a_given_sequence_of_features_values( \
        ['smoking=>heavy', 'exercising=>never', 'fatIntake=>low', 'videoAddiction=>none'])
print("class entroy on on a sequence of feature and values: ", ent)

best = dt.best_feature_calculator(['smoking=>heavy', 'exercising=>never'])
print("best feature to use next: ", best)

best = dt.best_feature_calculator(['smoking=>heavy'])
print("best feature to use next: ", best)

best = dt.best_feature_calculator([])
print("best feature to use next: ", best)

'''

