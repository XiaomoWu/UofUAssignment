
#----------------------------  Test Code Follows  -----------------------

fpath = 'C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-01-DT/DecisionTree-Purde/Examples/'

dt = DecisionTree( training_datafile = fpath + "training.dat",  
                    max_depth_desired = 2,
                    entropy_threshold = 0.1,
                    debug1 = 1,
                    )
dt.get_training_data()

dt.show_training_data()

prob = dt.prior_probability_for_class( 'benign' )
print("prior for benign: ", prob)
prob = dt.prior_probability_for_class( 'malignant' )
print("prior for malignant: ", prob)

prob = dt.probability_for_feature_value( 'smoking', 'heavy')
print(prob)

dt.determine_data_condition()

root_node = dt.construct_decision_tree_classifier()
root_node.display_decision_tree("   ")

test_sample = ['exercising=>never', 'smoking=>heavy', 'fatIntake=>heavy', 'videoAddiction=>heavy']
classification = dt.classify(root_node, test_sample)
print("Classification: ", classification)

test_sample = ['videoAddiction=>none', 'exercising=>occasionally', 'smoking=>never', 'fatIntake=>medium']
classification = dt.classify(root_node, test_sample)
print("Classification: ", classification)

print("Number of nodes created: ", root_node.how_many_nodes())
