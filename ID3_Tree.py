import pandas as pd
import argparse
import decision_tree # this is the moduole where we write the node class and implement the algorithm

# Here we initialize a parser to read the command line arguments to the script
parser = argparse.ArgumentParser(description='Takes L, K, training set, validation set, test set, to_print\n' + \
                                'Constructs 2 Decision Trees based on ID3 algorithm')
parser.add_argument('L',type=int,help='L value for tree pruning',nargs=1)
parser.add_argument('K',type=int,help='K value for tree pruning',nargs=1)
parser.add_argument('training_set',type=str,help='Address to training set',nargs=1)
parser.add_argument('validation_set',type=str,help='Address to validation set',nargs=1)
parser.add_argument('test_set',type=str,help='Address to test set',nargs=1)
parser.add_argument('to_print',type=bool,help='Whether or not to print the decision trees',nargs=1)

# Here we parse the arguments
args = parser.parse_args()

# Load the data into pandas dataframes
training_set = pd.read_csv(args.training_set[0])
validation_set = pd.read_csv(args.validation_set[0])
test_set = pd.read_csv(args.test_set[0])

# Initialize the two trees
col_list = training_set.columns.to_list()
attributes = col_list.remove('Class')

tree1 = decision_tree.Node(attributes, [], training_set, decision_tree.info_gain)
tree2 = decision_tree.Node(attributes, [], training_set, decision_tree.var_impurity)

# Hopefully once all of the TODOs are filled in these lines should train the
# trees correctly
# tree1.train()
# tree2.train()
