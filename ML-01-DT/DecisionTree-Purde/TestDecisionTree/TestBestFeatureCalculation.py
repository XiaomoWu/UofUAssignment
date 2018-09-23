import DecisionTree
import unittest

training_datafile = "training.dat"


class TestBestFeatureCalculation(unittest.TestCase):

    def setUp(self):
        print("Testing best-feature calculation on sample training file")
        self.dt = DecisionTree.DecisionTree(training_datafile = training_datafile)
        self.dt.get_training_data()

    def test_best_feature_calculation(self):
        best = self.dt.best_feature_calculator([])
        self.assertEqual(best[0], 'fatIntake')
        best = self.dt.best_feature_calculator(['smoking=>heavy'])
        self.assertEqual(best[0], 'fatIntake')
        best = self.dt.best_feature_calculator(['fatIntake=>medium', 'exercising=>never'])
        self.assertEqual(best[0], 'smoking')

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestBestFeatureCalculation, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

