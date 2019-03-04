#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:15:06 2019

@author: praveena
"""

import numpy as np
from scipy.stats import entropy

import pandas as pd

import timeit
from collections import Counter



#         
def infogain(S,A):
    
    sum_v = 0
    for v in set(A):
        Ex_a_v = [x for x, t in zip(S,A) if t == v]
   
        sum_v += (len(Ex_a_v) / len(S)) *\
                 (entropy(list(Counter(Ex_a_v).values()),base=2))
                 

# Varience Impurity 
 #   Let K denote the number of examples in the training set. Let K0 denote the 
 #number of training examples that have class = 0 and K1 denote the number of 
 #training examples that have class = 1. The variance impurity of the training 
 #set S is defined as:
#VI(S) = K0 K1 / KK

#Gain(S, X) = VI(S) − ∑ Pr(x)VI(Sx) x∈Values(X)

    
    KK = len(S)*len(S)
    K0K1 = list(Counter(S).values())[0] *  list(Counter(S).values())[1] 
  
    VIofS = K0K1/KK
    
    IGEntropy = entropy(list(Counter(S).values()),base=2) - sum_v  
    IGVI = K0K1/KK - sum_v  
    #print(  entropy(list(Counter(S).values()),base=2) )
    #return    IGEntropy,  IGVI         
    return    IGEntropy   



testset=pd.read_csv('training_set.csv')
trainingset=pd.read_csv('training_set.csv') 
validationset = pd.read_csv('validation_set.csv') 

#print(trainingset)

testdf = pd.DataFrame(testset)
trainingdf = pd.DataFrame(trainingset)
validationdf = pd.DataFrame(validationset)