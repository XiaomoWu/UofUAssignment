#!/usr/bin/env python

__version__ = '1.5'
__author__  = "Avinash Kak (kak@purdue.edu)"
__date__    = '2011-May-16'
__url__     = 'http://RVL4.ecn.purdue.edu/~kak/distDT/DecisionTree-1.5.html'
__copyright__ = "(C) 2011 Avinash Kak. Python Software Foundation."

__doc__ = '''

DecisionTree.py

Version: ''' + __version__ + '''
   
Author: Avinash Kak (kak@purdue.edu)

Date: ''' + __date__ + '''


@title
CHANGES:

  Version 1.5:

    This is a Python 3.x compliant version of the DecisionTree module.
    This version should work with both Python 2.x and Python 3.x.

  Version 1.0:

    This is a Python implementation of the author's Perl module
    Algorithm::DecisionTree, Version 1.41.  The Python version should work
    faster for large decision trees since it uses probability and entropy
    caching much more extensively than Version 1.41 of the Perl module.
    (Note: I expect my next release of the Perl module to catch up with
    this Python version in terms of performance.)


@title
USAGE:

    The DecisionTree module is a pure-Python implementation for
    constructing a decision tree from multidimensional training data and
    for using the decision tree thus constructed for classifying unlabeled
    data.

    For constructing a decision tree and for classifying a sample:

        dt = DecisionTree( training_datafile = "training.dat", debug1 = 1 )
        dt.get_training_data()
        dt.show_training_data()
        root_node = dt.construct_decision_tree_classifier()
        root_node.display_decision_tree("   ")
        test_sample = ['exercising=>never', 'smoking=>heavy', 
                       'fatIntake=>heavy', 'videoAddiction=>heavy']
        classification = dt.classify(root_node, test_sample)
        print "Classification: ", classification
        print "Number of nodes created: ", root_node.how_many_nodes()

    For the above calls to work, the format in which the training data is
    made available to the decision-tree constructor must meet certain
    assumptions.  (See the 'training.dat' file in the Examples directory
    for how to format the training data in a file.)  The training datafile
    must declare the class names, the feature names and the names of the
    different possible values for the features.  The rest of the training
    datafile is expected to contain the training samples in the form of a
    multi-column table.

    HIGHLY RECOMMENDED: Always use the "debug1" option in the constructor
    call above when working with a training dataset for the first time.

    If your training file specifies a large number of features or a large
    number of values for the features, the above constructor call could
    result in a decision tree that is simply much too large (and much too
    slow to construct).  For such cases, consider using following
    additional options in the constructor call shown above:

        dt = DecisionTree( training_datafile = "training.dat",  
                           max_depth_desired = some_number,
                           entropy_threshold = some_value,
                           debug1 = 1,
                         )

    where for max_depth_desired you should choose a number that is less
    than the number of features in your training file. This will set the
    depth of your decision tree to max_depth_desired. The algorithm will
    automatically use the BEST max_depth_desired features --- best in the
    sense of being the most discriminative --- for constructing the
    decision tree.  The parameter entropy_threshold sets the granularity
    with which the entropies will be sampled.  Its default value is 0.001.
    The larger the value you choose for entropy_threshold, the smaller the
    tree.


@title
INTRODUCTION:

    DecisionTree is a pure Python module for constructing a decision tree
    from a training datafile containing multidimensional data in the form
    of a table. In one form or another, decision trees have been around for
    the last fifty years. However, their popularity during the last decade
    is owning to the entropy-based method proposed by Ross Quinlan for
    their construction.  Fundamental to Quinlan's approach is the notion
    that a decision node in a tree should be split only if the entropy at
    the ensuing child nodes will be less than the entropy at the node in
    question.  The implementation presented here is based on the same idea.

    For those not familiar with decision tree ideas, the traditional way to
    classify multidimensional data is to start with a feature space whose
    dimensionality is the same as that of the data.  Each feature in this
    space would correspond to the attribute that each dimension of the data
    measures.  You then use the training data to carve up the feature space
    into different regions, each corresponding to a different class.
    Subsequently, when you are trying to classify a new data sample, you
    locate it in the feature space and find the class label of the region
    to which it belongs.  One can also give the data point the same class
    label as that of the nearest training sample.  (This is referred to as
    the nearest neighbor classification.)

    A decision tree classifier works differently.  When you construct a
    decision tree, you select for the root node a feature test that can be
    expected to maximally disambiguate the class labels that could be
    associated with the data you are trying to classify.  You then attach
    to the root node a set of child nodes, one for each value of the
    feature you chose at the root node. Now at each child node you pose the
    same question that you posed when you found the best feature to use at
    the root node: What feature at the child node in question would
    maximally disambiguate the class labels to be associated with a given
    data vector assuming that the data vector passed the root node on the
    branch that corresponds to the child node in question.  The feature
    that is best at each node is the one that causes the maximal reduction
    in class entropy at that node.

    As the reader would expect, the two key steps in any approach to
    decision-tree based classification are the construction of the decision
    tree itself from a file containing the training data, and then using
    the decision tree thus obtained for classifying the data.

    What is cool about decision tree classification is that it gives you
    soft classification, meaning it may associate more than one class label
    with a given data vector.  When this happens, it may mean that your
    classes are indeed overlapping in the underlying feature space.  It
    could also mean that you simply have not supplied sufficient training
    data to the decision tree classifier.  For a tutorial introduction to
    how a decision tree is constructed and used, visit
    http://cobweb.ecn.purdue.edu/~kak/DecisionTreeClassifiers.pdf


@title
WHAT PRACTICAL PROBLEM IS SOLVED BY THIS MODULE?

    Consider the following scenario: Let's say you are running a small
    investment company that employs a team of stockbrokers who make
    buy/sell decisions for the customers of your company.  Assume that your
    company has asked the traders to make each investment decision on the
    basis of the following four criteria:

            price_to_earnings_ratio   (P_to_E)

            price_to_sales_ratio      (P_to_S)

            return_on_equity          (R_on_E)

            market_share              (M_S)

            sentiment                 (S)

    Since you are the boss, you keep track of the buy/sell decisions made
    by the individual traders.  But one unfortunate day, all of your
    traders decide to quit because you did not pay them enough.  So what do
    you do?  If you had a module like the one here, you could still run
    your company and do so in such a way that, on the average, would do
    better than any of the individual traders who worked for your company.
    This is what you do: You pool together the individual trader buy/sell
    decisions you have accumulated during the last one year.  This pooled
    information is likely to look like:


      example      buy/sell     P_to_E     P_to_S     R_on_E     M_S     S
      ====================================================================

      example_1     buy          high       low        medium    low    high
      example_2     buy          medium     medium     low       low    medium
      example_3     sell         low        medium     low       high   low
      ....
      ....

    This data would constitute your training file. You could feed this file
    into the module by calling:

        dt = DecisionTree( training_datafile = "training.dat" )

        dt.get_training_data()

    and then construct a decision tree by calling:

        root_node = dt.construct_decision_tree_classifier()

    Now you and your company (with practically no employees) are ready to
    service the customers again. Suppose your computer needs to make a
    buy/sell decision about an investment prospect that is best described
    by:

        price_to_earnings_ratio   =>  low
        price_to_sales_ratio      =>  very_low
        return_on_equity          =>  none
        market_share              =>  medium    
        sentiment                 =>  low

    All that your computer would need to do would be to construct a data
    vector like

        test_case = [ 'P_to_E=>low', 
                      'P_to_S=>very_low', 
                      'R_on_E=>none',
                      'M_S=>medium',
                      'S=>low'  ]

    and call the decision tree classifier you just constructed by

        classification = dt.classify(root_node, test_case)

        print "Classification: ", classification

    The answer returned will be 'buy' and 'sell', along with the associated
    probabilities.  So if the probability of 'buy' is considerably greater
    than the probability of 'sell', that's what you should instruct your
    computer to do.

    The chances are that, on the average, this approach would beat the
    performance of any of your individual traders who worked for you
    previously since the buy/sell decisions made by the computer would be
    based on the collective wisdom of all your previous traders.
    DISCLAIMER: There is obviously a lot more to good investing than what
    is captured by the silly little example here. However, it does the
    convey the sense in which the current module could be used.


@title
WHAT HAPPENS IF THE NUMBER OF FEATURES AND/OR VALUES IS LARGE?

    If n is the number of features and m the largest number for the
    possible values for any of the features, then, in only the worst case,
    the algorithm would want to construct m**n nodes.  In other words, in
    the worst case, the size of the decision tree grows exponentially as
    you increase either the number of features or the number of possible
    values for the features.  That is the bad news.  The good news is that
    you have two constructor parameters at your disposal for controlling
    the size of the decision tree: The parameter max_depth_desired controls
    the depth of the constructed tree from its root node, and the parameter
    entropy_threshold controls the granularity with which the entropy space
    will be sampled.  The smaller the max_depth_desired and the larger the
    entropy_threshold, the smaller the size of the decision tree.  The
    default value for max_depth_desired is the number of features specified
    in the training datafile, and the the default value for
    entropy_threshold is 0.001.

    The users of this module with a need to create very large decision
    trees should also consider storing the tree once constructed in a
    diskfile and then using the stored tree for classification work.  The
    scripts store_dt_on_disk.py and classify_from_diskstored_dt.py in the
    Examples directory show you how you can do that with the help of
    Python's pickle module. (NOTE: At this time, this feature works only
    with Python 2.x.)

    Also note that it is always a good idea to keep the debug1 option set
    anytime you are experimenting with a new datafile --- especially if
    your training datafile is likely to create an inordinately large
    decision tree.


@title
WHAT HAPPENS WHEN THE FEATURE VALUES ARE NUMERIC?

    The current module will treat a numeric value for a feature as just a
    string.  In that sense, there is no difference between a string value
    for a feature and a numeric value.  This would obviously make the
    module unsuitable for applications in which a feature may take on a
    numeric value from a very large set of such values and you want feature
    values to be compared using numeric comparison predicates as opposed to
    string comparison predicates.  (Consider, for example, using color as
    an object feature in a computer vision application.)  The decision
    trees for applications in which the feature values are primarily
    numeric in nature are constructed differently, as explained in the
    tutorial at
    http://cobweb.ecn.purdue.edu/~kak/DecisionTreeClassifiers.pdf

    
@title
METHODS:

    The module provides the following methods for decision-tree induction
    from training data in a diskfile, and for data classification with the
    decision tree.


@title
Constructing a decision tree:

        dt = DecisionTree( training_datafile = "training.dat", debug1 = 1 )

    This yields a new instance of the DecisionTree class.  For this call to
    make sense, the training data in the training datafile must be
    according to a certain format that is shown below.  (Also see the file
    training.dat in the Examples directory.)

    If your training file specifies a large number of features or a large
    number of values for the features, the above constructor call could
    result in a decision tree that is simply much too large (and much too
    slow to construct).  For such cases, consider using following
    additional options in the constructor call shown above:

        dt = DecisionTree( training_datafile = "training.dat",  
                           max_depth_desired = some_number,
                           entropy_threshold = some_value,
                           debug1 = 1,
                         )

    where for max_depth_desired you should choose a number that is less
    than the number of features in your training file. This will set the
    depth of your decision tree to max_depth_desired. The algorithm will
    automatically use the BEST max_depth_desired features --- best in the
    sense of being the most discriminative --- for constructing the
    decision tree.  The parameter entropy_threshold sets the granularity
    with which the entropies will be sampled.  Its default value is 0.001.
    The larger the value you choose for entropy_threshold, the smaller the
    tree.

@title
Reading in the training data:

    After you have constructed a new instance of the DecisionTree class,
    you must now read in the training data that is contained in the file
    named above.  This you do by:

        dt.get_training_data()

    IMPORTANT: The training data file must be in a format that makes sense
    to the decision tree constructor.  The information in this file should
    look like

        Class names: malignant benign
   
        Feature names and their values:
            videoAddiction => none low medium heavy
            exercising => never occasionally regularly
            smoking => heavy medium light never
            fatIntake => low medium heavy


        Training Data:
    
        sample     class      videoAddiction   exercising    smoking   fatIntake
        ==========================================================================
    
        sample_0   benign     medium           occasionally  heavy     low
        sample_1   malignant  none             occasionally  heavy     medium
        sample_2   benign     low              occasionally  light     heavy
        sample_3   malignant  medium           occasionally  heavy     heavy
        ....
        ....

    IMPORTANT: Note that the class names, the number of classes, the
    feature names, and the possible values for the features can be anything
    that your data requires them to be.


@title
Displaying the training data:

    If you wish to see the training data that was just digested by the
    module, call

        dt.show_training_data() 


@title
Constructing a decision-tree classifier:

    After the training data is ingested, it is time to construct a decision
    tree classifier.  This you do by

        root_node = dt.construct_decision_tree_classifier()

    This call returns an instance of type Node.  The Node class is defined
    within the main package file, at its end.  So, don't forget, that
    root_node in the above example call will be instantiated to an instance
    of type Node.


@title
Displaying the decision tree:

    You display a decision tree by calling

        root_node.display_decision_tree("   ")

    This will display the decision tree in your terminal window by using a
    recursively determined offset for each node as the display routine
    descends down the tree.

    I have intentionally left the syntax fragment root_node in the above
    call to remind the reader that display_decision_tree() is NOT called on
    the instance of the DecisionTree we constructed earlier, but on the
    Node instance returned by the call to
    construct_decision_tree_classifier().


@title
Classifying new data:

    You classify new data by first constructing a new data vector:

        test_sample = ['exercising=>never', 'smoking=>heavy', 
                       'fatIntake=>heavy', 'videoAddiction=>heavy']

    and calling the classify() method as follows:
 
        classification = dt.classify(root_node, test_sample)

    where, again, root_node is an instance of type Node that was returned
    by calling construct_decision_tree_classifier().  The variable
    classification is a dictionary whose keys are the class labels and
    whose values the associated probabilities.  You can print it out by

        print "Classification: ", classification


@title
Displaying the number of nodes created:

    You can print out the number of nodes in a decision tree by calling

        root_node.how_many_nodes()


@title
Determining the condition of the training data:

    The following method, automatically invoked when debug1 option is set
    in the call to the decision-tree constructor, displays useful
    information regarding your training data file.  This method also warns
    you if you are trying to construct a decision tree that may simply be
    much too large.

        dt.determine_data_condition()


@title
HOW THE CLASSIFICATION RESULTS ARE DISPLAYED

    It depends on whether you apply the classifier at once to all the data
    samples in a file, or whether you feed one data vector at a time into
    the classifier.

    In general, the classifier returns soft classification for a data
    vector.  What that means is that, in general, the classifier will list
    all the classes to which a given data vector could belong and the
    probability of each such class label for the data vector.  For the
    sort of data that is in the 'training.dat' file in the Examples 
    directory, the result of classification for a single data vector
    would look like:

        malignant with probability 0.744186046511628
        benign with probability 0.255813953488372

    For large test datasets, you would obviously want to process an entire
    file of test data at a time.  The best way to do this is to follow my
    script

        classify_test_data_in_a_file.py

    in the Examples directory.  This script requires three command-line
    arguments, the first argument names the training datafile, the second
    the test datafile, and the third in which the classification results
    will be deposited.  The test datafile must mention the order in which
    the features values are presented.  For an example, see the file
    'testdata.dat' in the Examples directory.

    With regard to the soft classifications returned by this classifier, if
    the probability distributions for the different classes overlap in the
    underlying feature space, you would want the classifier to return all
    of the applicable class labels for a data vector along with the
    corresponding class probabilities.  Another reason for why the decision
    tree classifier may associate significant probabilities with multiple
    class labels is that you used inadequate number of training samples to
    induce the decision tree.  The good thing is that the classifier does
    not lie to you (unlike, say, a hard classification rule that would
    return a single class label corresponding to the partitioning of the
    underlying feature space).  The decision tree classifier give you the
    best classification that can be made given the training data you fed
    into it.


@title
THE EXAMPLES DIRECTORY:

    See the 'Examples' directory in the distribution for how to induce a
    decision tree, and how to then classify new data using the decision
    tree.  To become more familiar with the module, run the script

        construct_dt_and_classify_one_sample.py

    to classify the data vector that is in the script.  Next run the script
    as it is

        classify_test_data_in_a_file.py   training.dat   testdata.dat   out.txt

    This call will first construct a decision tree using the training data
    in the file 'training.dat'.  It will then calculate the class label for
    each data vector in the file 'testdata.dat'.  The estimated class
    labels will be written out to the file 'out.txt'.

    The 'Examples' directory also contains the script 'store_dt_on_disk.py'
    that shows how you can use Python's pickle module to store a decision
    tree in a disk file.  The file 'classify_from_diskstored_dt.py' in the
    same directory shows how you can classify new data vectors with the
    stored decision tree.  This is expected to be extremely useful for
    situations that involve tens of thousands or millions of decision
    nodes. (NOTE: At this time, this feature only works with Python 2.x)


@title  
INSTALLATION:

    The DecisionTree class was packaged using Distutils.  For installation,
    execute the following command-line in the source directory (this is the
    directory that contains the setup.py file after you have downloaded and
    uncompressed the package):
 
            python setup.py install

    You have to have root privileges for this to work.  On Linux
    distributions, this will install the module file at a location that
    looks like

             /usr/lib/python2.6/site-packages/

    If you do not have root access, you have the option of working directly
    off the directory in which you downloaded the software by simply
    placing the following statements at the top of your scripts that use
    the DecisionTree class:

            import sys
            sys.path.append( "pathname_to_DecisionTree_directory" )

    To uninstall the module, simply delete the source directory, locate
    where the DecisionTree module was installed with "locate DecisionTree"
    and delete those files.  As mentioned above, the full pathname to the
    installed version is likely to look like
    /usr/lib/python2.6/site-packages/DecisionTree*

    If you want to carry out a non-standard install of the DecisionTree
    module, look up the on-line information on Disutils by pointing your
    browser to

              http://docs.python.org/dist/dist.html


@title
BUGS:

    Please notify the author if you encounter any bugs.  When sending
    email, please place the string 'DecisionTree' in the subject line.


@title
ACKNOWLEDGMENTS:

    The importance of the 'sentiment' feature in the "What Practical Problem
    is Solved by this Module" section was mentioned to the author by John
    Gorup.  Thanks John.

@title
AUTHOR:

    Avinash Kak, kak@purdue.edu

    If you send email, please place the string "DecisionTree" in your
    subject line to get past my spam filter.

@title
COPYRIGHT:

    Python Software Foundation License

    Copyright 2011 Avinash Kak

'''

import math
import re
import sys
import functools 


#-------------------------  Utility Functions ---------------------------

def sample_index(sample_name):
    '''
    We assume that every record in the training datafile begins with
    the name of the record. The only requirement is that these names
    end in a suffix like '_23', as in 'sample_23' for the 23rd training
    record.  This function returns the integer in the suffix.
    '''
    m = re.search('_(.+)$', sample_name)
    return int(m.group(1))

# Meant only for an array of strings (no nesting):
def deep_copy_array(array_in):
    array_out = []
    for i in range(0, len(array_in)):
        array_out.append( array_in[i] )
    return array_out

# Returns simultaneously the minimum value and its positional index in an
# array. [Could also have used min() and index() defined for Python's
# sequence types.]
def minimum(arr):
    min,index = None,None
    for i in range(0, len(arr)):  
        if min is None or arr[i] < min:
            index = i
            min = arr[i]
    return min,index


#---------------------- DecisionTree Class Definition ---------------------

class DecisionTree(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''DecisionTree constructor can only be called
                      with keyword arguments for the following
                      keywords: training_datafile, entropy_threshold,
                      max_depth_desired, debug1, and debug2''') 

        allowed_keys = 'training_datafile','entropy_threshold', \
                       'max_depth_desired','debug1','debug2'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError("Wrong keyword used --- check spelling") 

        training_datafile = entropy_threshold = max_depth_desired = None
        debug1 = debug2 = None

        if 'training_datafile' in kwargs : \
                           training_datafile = kwargs.pop('training_datafile')
        if 'entropy_threshold' in kwargs : \
                           entropy_threshold = kwargs.pop('entropy_threshold')
        if 'max_depth_desired' in kwargs : \
                           max_depth_desired = kwargs.pop('max_depth_desired')
        if 'debug1' in kwargs  :  debug1 = kwargs.pop('debug1')
        if 'debug2' in kwargs  :  debug2 = kwargs.pop('debug2')

        if training_datafile:
            self._training_datafile = training_datafile
        else:
            raise ValueError('''You must specify a training datafile''')
        if entropy_threshold: 
            self._entropy_threshold =  entropy_threshold
        else:
            self._entropy_threshold =  0.001        
        if max_depth_desired:
            self._max_depth_desired = max_depth_desired 
        else:
            self._max_depth_desired = None
        if debug1:
            self._debug1 = debug1
        else:
            self._debug1 = 0
        if debug2:
            self._debug2 = debug2
        else:
            self._debug2 = 0
        self._root_node = None
        self._probability_cache           = {}
        self._entropy_cache               = {}
        self._training_data_dict          = {}
        self._features_and_values_dict    = {}
        self._samples_class_label_dict    = {}
        self._class_names                 = []
        self._class_priors                = []
        self._feature_names               = []

    def get_training_data(self):
        recording_features_flag = 0
        recording_training_data = 0
        table_header = None
        column_labels_dict = {}
        FILE = None
        try:
            FILE = open( self._training_datafile )
        except IOError:
            print("unable to open %s" % self._training_datafile)
        for line in FILE:
            line = line.rstrip()
            lineskip = r'^[\s=#]*$'
            if re.search(lineskip, line): 
                continue
            elif re.search(r'\s*class', line, re.IGNORECASE) \
                       and not recording_training_data \
                       and not recording_features_flag:
                classpattern = r'^\s*class names:\s*([ \S]+)\s*'
                m = re.search(classpattern, line, re.IGNORECASE)
                if not m: 
                    raise ValueError('''No class names in training file''')
                self._class_names = m.group(1).split()
                continue
            elif re.search(r'\s*feature names and their values', \
                               line, re.IGNORECASE):
                recording_features_flag = 1
                continue
            elif re.search(r'training data', line, re.IGNORECASE):
                recording_training_data = 1
                recording_features_flag = 0
                continue
            elif not recording_training_data and recording_features_flag:
                feature_name_value_pattern = r'^\s*(\S+)\s*=>\s*(.+)'
                m = re.search(feature_name_value_pattern, line, re.IGNORECASE)
                feature_name = m.group(1)
                feature_values = m.group(2).split()
                self._features_and_values_dict[feature_name]  = feature_values
            elif recording_training_data:
                if not table_header:
                    table_header = line.split()
                    for i in range(2, len(table_header)):
                        column_labels_dict[i] = table_header[i]
                    continue
                record = line.split()
                self._samples_class_label_dict[record[0]] = record[1]
                self._training_data_dict[record[0]] = []
                for i in range(2, len(record)):
                    self._training_data_dict[record[0]].append(
                          column_labels_dict[i] + "=>" + record[i] )
        FILE.close()                        
        self._feature_names = self._features_and_values_dict.keys()
        if self._debug2:
            print("Class names: ", self._class_names)
            print( "Feature names: ", self._feature_names)
            print("Features and values: ", self._features_and_values_dict.items())
        for feature in self._feature_names:
            values_for_feature = self._features_and_values_dict[feature]
            for value in values_for_feature:
                feature_and_value = "".join([feature, "=>", value])
                self._probability_cache[feature_and_value] = \
                       self.probability_for_feature_value(feature, value)

    def show_training_data(self):
        print("Class names: ", self._class_names)
        print("\n\nFeatures and Their Possible Values:\n\n")
        features = self._features_and_values_dict.keys()
        for feature in sorted(features):
            print("%s ---> %s" \
                  % (feature, self._features_and_values_dict[feature]))
        print("\n\nSamples vs. Class Labels:\n\n")
        for item in sorted(self._samples_class_label_dict.items(), \
                key = lambda x: sample_index(x[0]) ):
            print(item)
        print("\n\nTraining Samples:\n\n")
        for item in sorted(self._training_data_dict.items(), \
                key = lambda x: sample_index(x[0]) ):
            print(item)


    #------------------    Classify with Decision Tree  -----------------------

    def classify(self, root_node, features_and_values):
        if not self.check_names_used(features_and_values):
            raise ValueError("Error in the names you have used for features and/or values") 
        classification = self.recursive_descent_for_classification( \
                                    root_node, features_and_values )
        if self._debug1:
            print("\nThe classification:")
            for class_name in self._class_names:
                print("    " + class_name + " with probability " + \
                                        str(classification[class_name]))
        return classification

    def recursive_descent_for_classification(self, node, feature_and_values):
        feature_test_at_node = node.get_feature()
        value_for_feature = None
        remaining_features_and_values = []
        for feature_and_value in feature_and_values:
            pattern = r'(.+)=>(.+)'
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            if feature == feature_test_at_node:
                value_for_feature = value
            else:
                remaining_features_and_values.append(feature_and_value)
        if feature_test_at_node:
            feature_value_combo = \
                    "".join([feature_test_at_node,"=>",value_for_feature])
        children = node.get_children()
        answer = {}
        if len(children) == 0:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(0, len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            return answer
        for child in children:
            branch_features_and_values = child.get_branch_features_and_values()
            last_feature_and_value_on_branch = branch_features_and_values[-1] 
            if last_feature_and_value_on_branch == feature_value_combo:
                answer = self.recursive_descent_for_classification(child, \
                                        remaining_features_and_values)
                break
        return answer


    #----------------------  Construct Decision Tree  -------------------------- 

    def construct_decision_tree_classifier(self):
        if self._debug1:        
            self.determine_data_condition() 
            print("\nStarting construction of the decision tree:\n") 
        class_probabilities = \
          list(map(lambda x: self.prior_probability_for_class(x), \
                                                   self._class_names))
        entropy = self.class_entropy_on_priors()
        root_node = Node(None, entropy, class_probabilities, [])
        self._root_node = root_node
        self.recursive_descent(root_node)
        return root_node        

    def recursive_descent(self, node):
        features_and_values_on_branch = node.get_branch_features_and_values()
        best_feature, best_feature_entropy =  \
         self.best_feature_calculator(features_and_values_on_branch)
        node.set_feature(best_feature)
        if self._debug1: node.display_node() 
        if self._max_depth_desired is not None and \
         len(features_and_values_on_branch) >= self._max_depth_desired:
            return
        if best_feature is None: return
        if best_feature_entropy \
                   < node.get_entropy() - self._entropy_threshold:
            values_for_feature = \
                  self._features_and_values_dict[best_feature]
            feature_value_combos = \
              map(lambda x: "".join([best_feature,"=>",x]), values_for_feature)
            for feature_and_value in feature_value_combos:
                extended_branch_features_and_values = None
                if features_and_values_on_branch is None:
                    extended_branch_features_and_values = feature_and_value
                else:
                    extended_branch_features_and_values = \
                        deep_copy_array( features_and_values_on_branch )
                    extended_branch_features_and_values.append(\
                                                      feature_and_value)
                class_probabilities = list(map(lambda x: \
         self.probability_for_a_class_given_sequence_of_features_and_values(\
                x, extended_branch_features_and_values), self._class_names))
                child_node = Node(None, best_feature_entropy, \
                     class_probabilities, extended_branch_features_and_values)
                node.add_child_link( child_node )
                self.recursive_descent(child_node)

    def best_feature_calculator(self, features_and_values_on_branch):
        features_already_used = []
        for feature_and_value in features_and_values_on_branch:
            pattern = r'(.+)=>(.+)'
            m = re.search(pattern, feature_and_value)
            feature = m.group(1)
            features_already_used.append(feature)
        feature_tests_not_yet_used = []
        for feature in self._feature_names:
            if (feature not in features_already_used):
                feature_tests_not_yet_used.append(feature)
        if len(feature_tests_not_yet_used) == 0: return None, None
        array_of_entropy_values_for_different_features = []
        for i in range(0, len(feature_tests_not_yet_used)):
            values = \
             self._features_and_values_dict[feature_tests_not_yet_used[i]]
            entropy_for_new_feature = None
            for value in values:
                feature_and_value_string = \
                   "".join([feature_tests_not_yet_used[i], "=>", value]) 
                extended_features_and_values_on_branch = None
                if len(features_and_values_on_branch) > 0:
                    extended_features_and_values_on_branch =  \
                          deep_copy_array(features_and_values_on_branch)
                    extended_features_and_values_on_branch.append(  \
                                              feature_and_value_string) 
                else:
                    extended_features_and_values_on_branch  =    \
                        [feature_and_value_string]
                if entropy_for_new_feature is None:
                    entropy_for_new_feature =  \
                   self.class_entropy_for_a_given_sequence_of_features_values(\
                             extended_features_and_values_on_branch) \
                     * \
                     self.probability_of_a_sequence_of_features_and_values( \
                         extended_features_and_values_on_branch)
                    continue
                else:
                    entropy_for_new_feature += \
                  self.class_entropy_for_a_given_sequence_of_features_values(\
                         extended_features_and_values_on_branch) \
                     *  \
                     self.probability_of_a_sequence_of_features_and_values( \
                         extended_features_and_values_on_branch)
            array_of_entropy_values_for_different_features.append(\
                                         entropy_for_new_feature)
        min,index = minimum(array_of_entropy_values_for_different_features)
        return feature_tests_not_yet_used[index], min


    #--------------------------  Entropy Calculators  --------------------------

    def class_entropy_on_priors(self):
        if 'priors' in self._entropy_cache:
            return self._entropy_cache['priors']
        entropy = None
        for class_name in self._class_names:
            prob = self.prior_probability_for_class(class_name)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        self._entropy_cache['priors'] = entropy
        return entropy

    def class_entropy_for_a_given_feature_and_given_value(self, \
                                feature_name, feature_value):
        feature_and_value = "".join([feature_name, "=>", feature_value])
        if feature_and_value in self._entropy_cache:
            return self._entropy_cache[feature_and_value]
        entropy = None
        for class_name in self._class_names:
            prob = self.probability_for_a_class_given_feature_value( \
                                class_name, feature_name, feature_value)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:  
                log_prob = 0 
            if prob > 0.999:   
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += - (prob * log_prob)
        self._entropy_cache[feature_and_value] = entropy 
        return entropy

    def class_entropy_for_a_given_feature(self, feature_name):
        if feature_name in self._entropy_cache:
            return self._entropy_cache[feature_name]
        entropy = None
        for feature_value in self._features_and_values_dict[feature_name]:
            if entropy is None:
                entropy = \
                    self.class_entropy_for_a_given_feature_and_given_value(\
                                              feature_name, feature_value) \
                    * \
                    self.probability_for_feature_value( \
                                       feature_name,feature_value)
                continue
            entropy += \
                self.class_entropy_for_a_given_feature_and_given_value( \
                                              feature_name, feature_value) \
                *  \
                self.probability_for_feature_value(feature_name,feature_value)
        self._entropy_cache[feature_name] = entropy
        return entropy

    def class_entropy_for_a_given_sequence_of_features_values(self, \
                                       array_of_features_and_values):
        sequence = ":".join(array_of_features_and_values)
        if sequence in self._entropy_cache:
            return self._entropy_cache[sequence]
        entropy = None    
        for class_name in self._class_names:
            prob = \
           self.probability_for_a_class_given_sequence_of_features_and_values(\
                 class_name, array_of_features_and_values)
            if prob == 0:
                prob = 1.0/len(self._class_names)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        self._entropy_cache[sequence] = entropy
        return entropy


    #-------------------------  Probability Calculators ------------------------

    def prior_probability_for_class(self, class_name):
        class_name_in_cache = "".join(["prior::", class_name])
        if class_name_in_cache in self._probability_cache:
            return self._probability_cache[class_name_in_cache]
        total_num_of_samples = len( self._samples_class_label_dict )
        all_values = self._samples_class_label_dict.values()
        for this_class_name in self._class_names:
            trues = list(filter( lambda x: x == this_class_name, all_values ))
            prior_for_this_class = (1.0 * len(trues)) / total_num_of_samples
            this_class_name_in_cache = "".join(["prior::", this_class_name])
            self._probability_cache[this_class_name_in_cache] = \
                                                    prior_for_this_class
        return self._probability_cache[class_name_in_cache]

    def probability_for_feature_value(self, feature, value):
        feature_and_value = "".join([feature, "=>", value])
        if feature_and_value in self._probability_cache:
            return self._probability_cache[feature_and_value]
        values_for_feature = self._features_and_values_dict[feature]
        values_for_feature = list(map(lambda x: feature + "=>" + x, \
                                                   values_for_feature))
        value_counts = [0] * len(values_for_feature)
        for sample in sorted(self._training_data_dict.keys(), \
                key = lambda x: sample_index(x) ):
            features_and_values = self._training_data_dict[sample]
            for i in range(0, len(values_for_feature)):
                for current_value in features_and_values:
                    if values_for_feature[i] == current_value:
                        value_counts[i] += 1 
        for i in range(0, len(values_for_feature)):
            self._probability_cache[values_for_feature[i]] = \
                      value_counts[i] / (1.0 * len(self._training_data_dict))
        if feature_and_value in self._probability_cache:
            return self._probability_cache[feature_and_value]
        else:
            return 0

    def probability_for_feature_value_given_class(self, feature_name, \
                                        feature_value, class_name):
        feature_value_class = \
             "".join([feature_name,"=>",feature_value,"::",class_name])
        if feature_value_class in self._probability_cache:
            return self._probability_cache[feature_value_class]
        samples_for_class = []
        for sample_name in self._samples_class_label_dict.keys():
            if self._samples_class_label_dict[sample_name] == class_name:
                samples_for_class.append(sample_name) 
        values_for_feature = self._features_and_values_dict[feature_name]
        values_for_feature = \
        list(map(lambda x: "".join([feature_name,"=>",x]), values_for_feature))
        value_counts = [0] * len(values_for_feature)
        for sample in samples_for_class:
            features_and_values = self._training_data_dict[sample]
            for i in range(0, len(values_for_feature)):
                for current_value in (features_and_values):
                    if values_for_feature[i] == current_value:
                        value_counts[i] += 1 
        total_count = functools.reduce(lambda x,y:x+y, value_counts)
        for i in range(0, len(values_for_feature)):
            feature_and_value_for_class = \
                     "".join([values_for_feature[i],"::",class_name])
            self._probability_cache[feature_and_value_for_class] = \
                                       value_counts[i] / (1.0 * total_count)
        feature_and_value_and_class = \
              "".join([feature_name, "=>", feature_value,"::",class_name])
        if feature_and_value_and_class in self._probability_cache:
            return self._probability_cache[feature_and_value_and_class]
        else:
            return 0

    def probability_of_a_sequence_of_features_and_values(self, \
                                        array_of_features_and_values):
        sequence = ":".join(array_of_features_and_values)
        if sequence in self._probability_cache:
            return self._probability_cache[sequence]
        probability = None
        for feature_and_value in array_of_features_and_values:
            pattern = r'(.+)=>(.+)'
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            if probability is None:
                probability = \
                   self.probability_for_feature_value(feature, value)
                continue
            else:
                probability *= \
                  self.probability_for_feature_value(feature, value)
        self._probability_cache[sequence] = probability
        return probability

    def probability_for_sequence_of_features_and_values_given_class(self, \
                            array_of_features_and_values, class_name):
        sequence = ":".join(array_of_features_and_values)
        sequence_with_class = "".join([sequence, "::", class_name])
        if sequence_with_class in self._probability_cache:
            return self._probability_cache[sequence_with_class]
        probability = None
        for feature_and_value in array_of_features_and_values:
            pattern = r'(.+)=>(.+)'
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            if probability is None:
                probability = self.probability_for_feature_value_given_class(\
                                                 feature, value, class_name)
                continue
            else:
                probability *= self.probability_for_feature_value_given_class(\
                                           feature, value, class_name)
        self._probability_cache[sequence_with_class] = probability
        return probability 

    def probability_for_a_class_given_feature_value(self, class_name, \
                                              feature_name, feature_value):
        prob = self.probability_for_feature_value_given_class( \
                                 feature_name, feature_value, class_name)
        answer = (prob * self.prior_probability_for_class(class_name)) \
                 /                                                     \
                 self.probability_for_feature_value(feature_name,feature_value)
        return answer

    def probability_for_a_class_given_sequence_of_features_and_values(self, \
                    class_name, array_of_features_and_values):
        sequence = ":".join(array_of_features_and_values)
        class_and_sequence = "".join([class_name, "::", sequence])
        if class_and_sequence in self._probability_cache:
            return self._probability_cache[class_and_sequence]
        array_of_class_probabilities = [0] * len(self._class_names)
        for i in range(0, len(self._class_names)):
            prob = \
            self.probability_for_sequence_of_features_and_values_given_class(\
                   array_of_features_and_values, self._class_names[i]) 
            if prob == 0:
                array_of_class_probabilities[i] = 0 
                continue
            prob_of_feature_sequence = \
                self.probability_of_a_sequence_of_features_and_values(  \
                                              array_of_features_and_values)
            prior = self.prior_probability_for_class(self._class_names[i])
            array_of_class_probabilities[i] =   \
               prob * self.prior_probability_for_class(self._class_names[i]) \
                 / prob_of_feature_sequence
        sum_probability = \
          functools.reduce(lambda x,y:x+y, array_of_class_probabilities)
        if sum_probability == 0:
            array_of_class_probabilities = [1.0/len(self._class_names)] \
                                              * len(self._class_names)
        else:
            array_of_class_probabilities = \
                     list(map(lambda x: x / sum_probability,\
                               array_of_class_probabilities))
        for i in range(0, len(self._class_names)):
            this_class_and_sequence = \
                     "".join([self._class_names[i], "::", sequence])
            self._probability_cache[this_class_and_sequence] = \
                                     array_of_class_probabilities[i]
        return self._probability_cache[class_and_sequence]


    #---------------------  Class Based Utilities  ---------------------

    def determine_data_condition(self):
        num_of_features = len(self._feature_names)
        values = list(self._features_and_values_dict.values())
        print("Number of features: ", num_of_features)
        max_num_values = 0
        for i in range(0, len(values)):
            if ((not max_num_values) or (len(values[i]) > max_num_values)):
                max_num_values = len(values[i])
        print("Largest number of feature values is: ", max_num_values)
        estimated_number_of_nodes = max_num_values ** num_of_features
        print("\nWORST CASE SCENARIO: The decision tree COULD have as many \
as   \n   %d nodes. The exact number of nodes created depends\n   critically \
on the entropy_threshold used for node expansion.\n   (The default for this \
threshold is 0.001.)\n" % (estimated_number_of_nodes))
        if estimated_number_of_nodes > 10000:
            print("\nTHIS IS WAY TOO MANY NODES. Consider using a relatively \
large\n   value for entropy_threshold to reduce the number of nodes created.\n")
            ans = raw_input("\nDo you wish to continue? Enter 'y' if yes:  ")
            if ans != 'y':
                sys.exit(0)
        print("\nHere are the probabilities of feature-value pairs in your data:\n\n")
        for feature in self._feature_names:
            values_for_feature = self._features_and_values_dict[feature]
            for value in values_for_feature:
                prob = self.probability_for_feature_value(feature,value) 
                print("Probability of feature-value pair (%s,%s): %.3f" % \
                                                (feature,value,prob)) 

    def check_names_used(self, features_and_values_test_data):
        for feature_and_value in features_and_values_test_data:
            pattern = r'(.+)=>(.+)'
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            if feature is None or value is None:
                raise ValueError("Your test data has formatting error")
            if feature not in self._feature_names:
                return 0
            if value not in self._features_and_values_dict[feature]:
                return 0
        return 1

    def get_class_names(self):
        return self._class_names


#-----------------------------   Class Node  ----------------------------

# The nodes of the decision tree are instances of this class:
class Node(object):

    nodes_created = -1

    def __init__(self, feature, entropy, class_probabilities, \
                                                branch_features_and_values):
        self._serial_number = self.get_next_serial_num()
        self._feature       = feature
        self._entropy       = entropy
        self._class_probabilities = class_probabilities
        self._branch_features_and_values = branch_features_and_values
        self._linked_to = []

    def get_next_serial_num(self):
        Node.nodes_created += 1
        return Node.nodes_created

    def get_serial_num(self):
        return self._serial_number

    # This is a class method:
    @staticmethod
    def how_many_nodes():
        return Node.nodes_created

    # this returns the feature test at the current node
    def get_feature(self):
        return self._feature

    def set_feature(self, feature):
        self._feature = feature

    def get_entropy(self):
        return self._entropy

    def get_class_probabilities(self):
        return self._class_probabilities

    def get_branch_features_and_values(self):
        return self._branch_features_and_values

    def add_to_branch_features_and_values(self, feature_and_value):
        self._branch_features_and_values.append(feature_and_value)

    def get_children(self):
        return self._linked_to

    def add_child_link(self, new_node):
        self._linked_to.append(new_node)                  

    def delete_all_links(self):
        self._linked_to = None

    def display_node(self):
        feature_at_node = self.get_feature() or " "
        entropy_at_node = self.get_entropy()
        class_probabilities = self.get_class_probabilities()
        serial_num = self.get_serial_num()
        branch_features_and_values = self.get_branch_features_and_values()
        print("\n\nNODE " + str(serial_num) + ":\n   Branch features and values to this \
node: " + str(branch_features_and_values) + "\n   Class probabilities at \
current node: " + str(class_probabilities) + "\n   Entropy at current \
node: " + str(entropy_at_node) + "\n   Best feature test at current \
node: " + feature_at_node + "\n\n")

    def display_decision_tree(self, offset):
        serial_num = self.get_serial_num()
        if len(self.get_children()) > 0:
            feature_at_node = self.get_feature() or " "
            entropy_at_node = self.get_entropy()
            class_probabilities = self.get_class_probabilities()
            print("NODE " + str(serial_num) + ":  " + offset +  "feature: " + feature_at_node \
+ "   entropy: " + str(entropy_at_node) + "   class probs: " + str(class_probabilities) + "\n")
            offset += "   "
            for child in self.get_children():
                child.display_decision_tree(offset)
        else:
            entropy_at_node = self.get_entropy()
            class_probabilities = self.get_class_probabilities()
            print("NODE " + str(serial_num) + ": " + offset + "   entropy: " \
+ str(entropy_at_node) + "    class probs: " + str(class_probabilities) + "\n")

