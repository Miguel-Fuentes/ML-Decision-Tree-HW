import pandas as pd
import numpy as np

from collections import namedtuple
from collections import Counter

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
        self.attributes = [] if self.result != None else attributes # If the node is pure this should be empty
        self.split_attr = None
        
        self.val0 = None
        self.val1 = None
        
    def __str__(self):
        if self.result != None:
            return str(self.result)
        else:
            depth = len(self.ancestors)
            output = '\n' + '| ' * depth
            output += f'{self.split_attr} = 0: '
            output += str(self.val0)
            output += '\n' + '| ' * depth
            output += f'{self.split_attr} = 1: '
            output += str(self.val1)
            return output
        
    def __repr__(self):
        """ This is used for debugging and will not return the tree in a pretty way,
        only the info about this node will be returned
        """
        return (f'Heuristic: {self.heuristic}\n'\
                f'Ancestors: {self.ancestors}\n'\
                f'Result: {self.result}\n'\
                f'Attributes: {self.attributes}\n'\
                f'Split Attribute: {self.split_attr}\n'\
                f'Has children: {self.val0 != None}\n')
                        
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
        If the node has children it will return the (node, attribute, gain) tuple of the child with the
        highest gain
        If the node does not have children and is not pure it will return the (node, attribute, gain) tuple
        with itself as the node and the highest heuristic score of splitting on any of its attributes
        as the gain
        If the node is pure it will return (None, '', 0) as it can no longer be split
        """
        if self.val1:
            val1_gain_tuple, val0_gain_tuple = self.val1.max_gain(), self.val0.max_gain()
            if val1_gain_tuple.gain > val0_gain_tuple.gain:
                return val1_gain_tuple
            else:
                return val0_gain_tuple
        elif self.attributes:
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
        
        self.split_attr = attribute
        
        # list() is used to make a copy of the list instead of pointing to the same list
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
    
    def predict_row(self, row):
        current_node = self
        result = self.result
        while result == None and current_node.val0 != None:
            if row[current_node.split_attr] == 0:
                current_node = current_node.val0
            else:
                current_node = current_node.val1
            result = current_node.result
        return current_node.result
    
    def predict(self, df):
        return df.apply(self.predict_row, axis=1)

def entropy_gain(node,attribute):
    """This will return the information gain that you get from splitting a node
    with that has the data "data" on the attribute "attribute"
    """
    data_subset1 = filter_data(node.data,node.ancestors)
    data_counts = list(Counter(data_subset1['Class']).values())
    base_entropy = entropy(data_counts,base=2)
    num_values = len(data_subset1)
    entropy_sum = 0
    
    for value in [0,1]:
        data_subset2 = filter_data(node.data, node.ancestors + [(attribute,value)])
        subset_counts = list(Counter(data_subset2['Class']).values())
        entropy_sum += (len(data_subset2)/num_values) * entropy(subset_counts,base=2)
    
    return base_entropy - entropy_sum

def impurity(count_list):
    ''' This function takes a list of counts i.e. the number of samples in each class
    and returns the impurity of that set of samples
    '''
    if len(count_list) < 2:
        return 0
    product = 1
    total = 0
    values = 0
    for count in count_list:
        product *= count
        total += count
        values += 1
    return (product / (total**values))

def impurity_gain(node, attribute):  
    """This will return the information gain that you get from splitting a node
    with that has the data "data" on the attribute "attribute"
    """
    data_subset1 = filter_data(node.data,node.ancestors)
    data_counts = list(Counter(data_subset1['Class']).values())
    base_impurity = impurity(data_counts)
    num_values = len(data_subset1)
    impurity_sum = 0
    
    for value in [0,1]:
        data_subset2 = filter_data(node.data, node.ancestors + [(attribute,value)])
        subset_counts = list(Counter(data_subset2['Class']).values())
        impurity_sum += (len(data_subset2)/num_values) * impurity(subset_counts)
    
    return base_impurity - impurity_sum

def filter_data(data,filters):
    """ This filters the training data according to the ancestors of this node so that only a
    subset of the training data is considered for calculations like entropy
    """
    final_filter = pd.Series(np.array([True] * data.shape[0]))
    for attribute, value in filters:
        final_filter &= data[attribute] == value
    return data[final_filter]		
			