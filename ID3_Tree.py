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
parser.add_argument('to_print',type=str,help='Whether or not to print the decision trees',nargs=1)

# EX command line input (replace 0 and 0 with L and K)
# python id3_tree.py 0 0 data_sets1/training_set.csv data_sets1/validation_set.csv data_sets1/test_set.csv no

# Here we parse the arguments
args = parser.parse_args()
printing = args.to_print[0] == 'yes'
print('Arguments Parsed')


# Load the data into pandas dataframes
training_set = pd.read_csv(args.training_set[0])
validation_set = pd.read_csv(args.validation_set[0])
test_set = pd.read_csv(args.test_set[0])
print('CSVs Read')

# Initialize the two trees
attributes = training_set.columns.to_list()
attributes.remove('Class')

tree1 = decision_tree.Node(attributes, [], training_set, decision_tree.entropy_gain)
tree2 = decision_tree.Node(attributes, [], training_set, decision_tree.impurity_gain)
print('Trees Initialized, trees will train now this may take up to 5 minutes')

tree1.train()
tree2.train()
print('Trees Trained')

print('Trees Evaluated',
      f'Entropy Based Tree Accuracy: {decision_tree.accuracy(tree1, test_set)}',
      f'Variance Impurity Based Tree Accuracy: {decision_tree.accuracy(tree2, test_set)}',  
      sep='\n')

tree1 = tree1.post_pruning(args.L[0], args.K[0], validation_set)
tree2 = tree2.post_pruning(args.L[0], args.K[0], validation_set)

print('Trees Pruned',
      f'Entropy Based Tree Accuracy Post Pruning: {decision_tree.accuracy(tree1, test_set)}',
      f'Variance Impurity Based Tree Accuracy Post Pruning: {decision_tree.accuracy(tree2, test_set)}',  
      sep='\n')

if printing:
    print('Printing trees post pruning',
          'Entropy Based Tree',
          tree1,
          'Variance Impurity Based Tree',
          tree2, sep='\n')