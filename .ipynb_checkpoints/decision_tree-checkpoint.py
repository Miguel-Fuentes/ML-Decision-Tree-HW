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
    
    def split(self):
        # TODO this should split this node according to the arrtibute with highest gain, i.e. set val0
        # and val1 to point to new nodes
        return 0
        
    def purity_test(self):
        # TODO if the node is pure this should return the class of the node
        # else it should return None
        return None

def info_gain(data, attributes):
    # TODO write information gain
    return 0

def var_impurity(data, attributes):
    # TODO write variance impurity
    return 0