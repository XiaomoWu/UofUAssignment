#!/usr/bin/env python

### setup.py

from distutils.core import setup

setup(name='DecisionTree',
      version='1.5',
      author='Avinash Kak',
      author_email='kak@purdue.edu',
      maintainer='Avinash Kak',
      maintainer_email='kak@purdue.edu',
      url='http://RVL4.ecn.purdue.edu/~kak/distDT/DecisionTree-1.5.html',
      download_url='http://RVL4.ecn.purdue.edu/~kak/distDT/DecisionTree-1.5.tar.gz?download',
      description='A pure-Python implementation for constructing a decision tree from multidimensional training data and for using the decision tree for classifying unlabeled data',
      long_description=''' 

**Version 1.5** should work with both Python 3.x and Python 2.x.

This module is a pure-Python implementation for constructing
a decision tree from multidimensional training data and
subsequently using the decision tree to classify future
data.  

Assuming you have arranged your training data in the form of
a table in a text file, all you have to do is to supply the
name of the training datafile to this module and it does the
rest for you without much effort on your part.

A decision tree classifier consists of feature tests that
are arranged in the form of a tree. You associate with the
root node a feature test that can be expected to maximally
disambiguate the different possible class labels for an
unlabeled data vector.  You then hang from the root node a
set of child nodes, one for each value of the feature that
you chose for the root node.  At each such child node, you
now select a feature test that is the most class
discriminative given that you have already applied the
feature test at the root node and observed the value for
that feature.  This process is continued until you reach the
leaf nodes of the tree.  The leaf nodes may either
correspond to the maximum depth desired for the decision
tree or to the case when you run out of features to test.

Typical usage syntax:

        dt = DecisionTree( training_datafile = "training.dat", debug1 = 1 )
        dt.get_training_data()
        dt.show_training_data()
        root_node = dt.construct_decision_tree_classifier()
        root_node.display_decision_tree("   ")
        test_sample = ['exercising=>never', 'smoking=>heavy', 
                       'fatIntake=>heavy', 'videoAddiction=>heavy']
        classification = dt.classify(root_node, test_sample)
        print "Classification: ", classification

          ''',

      license='Python Software Foundation License',
      keywords='data classification, decision trees, information analysis',
      platforms='All platforms',
      classifiers=['Topic :: Scientific/Engineering :: Information Analysis'],
      packages=['']
)
