           __________________________________________________

             CS 5350/6350 COMPETITIVE PROJECT: CLASSIFYING
                             MOVIE REVIEWS
           __________________________________________________





1 Introduction
==============

  Each example in this classification task is a movie review. The goal
  is to predict whether the review is a positive or a negative one. The
  data used for this task is based on the *Large Movie Review Dataset
  v1.0*[1].


2 Data
======

  The `data-splits' directory contains the following three data files
  (one training set and two test sets):

  1. `data-splits/data.train': This is the training set, in the usual
     liblinear format. There are 25000 training examples.

  2. `data-splits/data.test': This is the set of examples on which you
     will report results in your final report. There are 12500 test
     examples.

  3. `data-splits/data.eval.anon': These 12500 examples are all labeled
     positive in the provided data set. You should use your models to
     make predictions on each example and upload them to Kaggle. See
     below for the format of the upload. Half of these examples are used
     to produce the public leader board. The other half will be used to
     evaluate your results.

  In addition, the directory also contains a file called
  `data-splits/data.eval.anon.ids'. This file has as many rows as the
  `data.eval.anon' file. Each line consists of an example id, that
  uniquely identifies the evaluation example. The ids from this file
  will be used to match your uploaded predictions on Kaggle. (The train
  and test splits are also associated with ids, but we will not use
  them.)


  We have replicated the features that were used in the original
  paper. A movie review is featurized as a bag-of-words, where each
  feature is the number of times a particular word occurs in the
  review. Of course, most words don't occur in a single review. So while
  the dimensionality of the feature vector is the number of words (in
  this case 74481), most reviews correspond to very sparse vectors in
  this space.

  (Your implementations of learning algorithms may have to deal with
  this large dimensional space. Use sparse vectors as features,
  otherwise, you will run out of memory.)

  Note that as part of your project, you are welcome to try different
  feature sets. To help with this, we have also provided the raw text
  that we used to produce the features. These are in the directory
  called `raw-data'. This directory contains three files corresponding
  to the train, test and eval splits described above. Each line in each
  of these files is a movie review. The file `vocab.gz' in this
  directory is the list of unique words in the training set and each
  line in this file corresponds to a feature dimension for the
  featurized version of the data in `data-splits'.


3 Evaluation
============

  We will use the accuracy of your classifiers to evaluate them. Note
  that the examples are split randomly among the three splits (train,
  test and eval). So we expect that the cross-validation performance on
  the training set and the accuracies on the test set and the public and
  private splits of the evaluation set to be similar.


4 Submission format
===================

  Kaggle accepts a csv file with your predictions on the examples in the
  evaluation data. There should be a header line containing
  `example_id,label'. Each subsequent line should consist of two
  entries: The example id (from the file `data.eval.ids') and the
  prediction (0 or 1).

  We have provided two sample solutions for your reference:

  1. `sample-solutions/sample-solutions.all.positive.csv': Where all
     examples are labeled as positive

  2. `sample-solutions/sample-solutions.half-neg.csv': Where the first
     half of examples are labeled false and the second half are labeled
     true


5 References
============

  1. Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew
     Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for
     Sentiment Analysis. The 49th Annual Meeting of the Association for
     Computational Linguistics (ACL 2011).
