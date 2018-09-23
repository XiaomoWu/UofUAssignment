import DecisionTree
import unittest

training_datafile = "training.dat"


class TestDecisionTreeClassification(unittest.TestCase):

    def setUp(self):
        print("Testing decision-tree classification on sample training file")
        self.dt = DecisionTree.DecisionTree(training_datafile = training_datafile)
        self.dt.get_training_data()
        self.root_node = self.dt.construct_decision_tree_classifier()

    def test_decision_tree_classification(self):
        test_sample = ['exercising=>never', 
               'smoking=>heavy', 
               'fatIntake=>heavy',
               'videoAddiction=>heavy']
        classification = self.dt.classify(self.root_node, test_sample)
        self.assertTrue( abs(classification['benign'] - 0.08978) < 0.01 )
        self.assertTrue( abs(classification['malignant'] - 0.91021) < 0.01 )

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestDecisionTreeClassification, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

