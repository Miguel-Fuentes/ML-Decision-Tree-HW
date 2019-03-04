import pandas as pd
from collections import namedtuple

# These tuples will store associated pairs of data
gain_tuple = namedtuple('gain_tuples',['node','gain'])
attribute_value = namedtuple('attribute_value',['attribute','value'])

class Node:
    def __init__(self, attributes, ancestors, data, heuristic):
        self.ancestors = ancestors
        self.data = data
        
        self.result = self.purity_test() # This should be None unless the node is pure
        self.attributes = [] if self.result else attributes # If the node is pure this should be empty
        
        self.val0 = None
        self.val1 = None
        
    def train(self):
        # This should get the node which will have the maximum heuristic gain when split
        max_tuple = self.max_gain()
        # If that gain is 0 then every node should be a pure leaf (hopefully) and you can stop
        while max_tuple.gain != 0:
            max_tuple.node.split()
            max_tuple = self.max_gain()
            
    def max_gain(self):
        # TODO this should traverse the tree and return a gain_tuple with the node with the maximum
        # gain, and the value for maximum gain at that node
        return gain_tuple(None,0)
    
    def split(self, attribute):
        # TODO this should split this node according to the arrtibute with highest gain, i.e. set val0
        # and val1 to point to new nodes
        return 0
        
    def filter_data(self):
        # At first initialize every row to True
        final_filter = pd.Series(np.array([True] * data.shape[0]))
        # And with a filter for every attribute, value pair in ancestors
        for attribute, value in self.ancestors:
            final_filter &= data[attribute] == value
        # This will return a subset of the data containing only the rows with the values matching ancestors
        return self.data[final_filter]
    
    def purity_test(self):
        # If the sample is pure, returns class, else returns None
        mean = data['Class'].mean()
        if mean == 0:
            return 0
        elif mean == 1:
            return 1
        return None

def info_gain(data, attributes):
    # TODO write information gain
    return 0

def var_impurity(data, attributes):
    # TODO write variance impurity
    return 0