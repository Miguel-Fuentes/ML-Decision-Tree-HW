import pandas as pd
import numpy as np

from collections import namedtuple

from scipy.stats import entropy

# These tuples will store associated pairs of data
gain_tuple = namedtuple('gain_tuples',['node','attribute','gain'])
attribute_value = namedtuple('filter',['attribute','value'])

class Node:
    def __init__(self, attributes, ancestors, data, heuristic):
        self.heuristic = heuristic
        self.ancestors = ancestors
        self.data = data
        
        self.result = self.purity_test() # This should be None unless the node is pure
        self.attributes = [] if self.result else attributes # If the node is pure this should be empty
        
        self.val0 = None
        self.val1 = None
        
    def __repr__(self):
        """ This is used for debugging and will not return the tree in a pretty way,
        only the info about this node will be returned
        """
        return (f'Heuristic: {self.heuristic}\n'\
                f'Ancestors: {self.ancestors}\n'\
                f'Result: {self.result}\n'\
                f'Attributes: {self.attributes}\n')
                        
    def train(self):
        """ This will continue splitting the tree until every leaf node is pure and the training data
        is perfectly characterized by the decision tree
        """
        max_tuple = self.max_gain()
        # If that gain is 0 then every node should be a pure leaf (hopefully) and you can stop
        while max_tuple.gain != 0:
            max_tuple.node.split(max_tuple.attribute)
            max_tuple = self.max_gain()
            
    def max_gain(self):
        """ 
        If the node has children it will return the (node, gain) tuple of the child with the
        highest gain
        If the node does not have children and is not pure it will return the (node, gain) tuple
        with itself as the node and the highest heuristic score of splitting on any of its attributes
        as the gain
        If the node is pure it will return (None, 0) as it can no longer be split
        """
        if self.val1:
            val1_gain_tuple, val0_gain_tuple = self.val1.max_gain(), self.val0.max_gain()
            if val1_gain_tuple.gain > val0_gain_tuple.gain:
                return val1_gain_tuple
            else:
                return val0_gain_tuple
        if self.attributes:
            filtered_data = filter_data(self.data,self.ancestors)
            max_attribute, max_gain = max([(attribute,
                                            self.heuristic(self,attribute)) for attribute in self.attributes],
                                         key = lambda x: x[1])
            return gain_tuple(self, max_attribute, max_gain)
        return gain_tuple(None, '', 0)
    
    def split(self, attribute):
        """ This splits a node on the attribute "attribute"
        """
        if attribute not in self.attributes:
            raise KeyError('Attribute not present in node')
    
        child_attributes = list(self.attributes)
        child_attributes.remove(attribute)
    
        child1_ancestors = list(self.ancestors)
        child0_ancestors = list(self.ancestors)
        child1_ancestors.append(attribute_value(attribute, 1))
        child0_ancestors.append(attribute_value(attribute, 0))
    
        self.val1 = Node(child_attributes, child1_ancestors, self.data, self.heuristic)
        self.val0 = Node(child_attributes, child0_ancestors, self.data, self.heuristic)
        
    def purity_test(self):
        """If the sample is pure, returns class, else returns None"""
        mean = filter_data(self.data,self.ancestors)['Class'].mean()
        if mean == 0:
            return 0
        elif mean == 1:
            return 1
        return None

def info_gain(node,attribute):
    """This will return the information gain that you get from splitting a node
    with that has the data "data" on the attribute "attribute"
    """
    _, data_counts = np.unique(node.data['Class'], return_counts=True)
    base_entropy = entropy(data_counts,base=2)
    num_values = len(node.data)
    entropy_sum = 0
    
    for value in [0,1]:
        data_subset = filter_data(node.data,node.ancestors + [(attribute,value)])
        _, subset_counts = np.unique(data_subset['Class'], return_counts=True)
        entropy_sum += (len(data_subset)/num_values) * entropy(subset_counts,base=2)
    
    return base_entropy - entropy_sum

def var_impurity(data, attributes):
    # TODO write variance impurity
    return 0

def filter_data(data,filters):
    """ This filters the training data according to the ancestors of this node so that only a
    subset of the training data is considered for calculations like entropy
    """
    final_filter = pd.Series(np.array([True] * data.shape[0]))
    for attribute, value in filters:
        final_filter &= data[attribute] == value
    return data[final_filter]
			
			
			