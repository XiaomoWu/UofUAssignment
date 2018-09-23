import DecisionTree
import unittest

training_datafile = "training.dat"


class TestEntropyCalculation(unittest.TestCase):

    def setUp(self):
        print("Testing entropy calculation on sample training file")
        self.dt = DecisionTree.DecisionTree(training_datafile = training_datafile)
        self.dt.get_training_data()

    def test_class_entropy_on_priors(self):
        ent = self.dt.class_entropy_on_priors()
        self.assertTrue( abs(ent - 0.88) < 0.01)

    def test_class_entropy_for_given_feature_and_value(self):
        ent = self.dt.class_entropy_for_a_given_feature_and_given_value('smoking','heavy') 
        self.assertTrue( abs(ent - 0.98) < 0.01)

    def test_class_entropy_for_given_sequence_of_feature_values(self):
        ent = self.dt.class_entropy_for_a_given_sequence_of_features_values( \
         ['smoking=>heavy', 'exercising=>never', 'fatIntake=>low', 'videoAddiction=>none'])            
        self.assertTrue( abs(ent - 0.903) < 0.01)

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestEntropyCalculation, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

