# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:37:45 2018

@author: Lewis
"""

from qc import *
from functions import *
import numpy as np
from scipy.spatial import distance
from sympy.utilities.iterables import multiset_permutations
from itertools import permutations


def grover_search(oracle):
    n = oracle.n_qubits
    Hn = Hadamard(n)
    H = Hadamard()
    reg1 = QuantumRegister()
    reg0 = QuantumRegister()
    #set registers here
    #
    #
    #gate 1
    reg0 = Hn*reg0
    reg1 = H*reg1
    
    #reflection gate > I|w>-2(<e|w>)|e>
    
    R = Operator(n,-1*np.eye(n-1))
    
    #grover diffusion gate
    
    grov_diff = Hn*R*Hn
    
    for i in range(np.round(np.sqrt(n))):
        reg = grov_diff*oracle*reg
        
    k = reg.measure()
    return k

class TSPOracle():
    def __init__(self, locations,threshold):
        self.locations = locations
        self.locationPerms = tuple(permutations(self.locations))
        self.threshold = threshold
        
    def calcDistance(self, locationArray):
        dist = 0
        for i in range(len(locationArray)-1):
            dist += distance.euclidean(locationArray[i],locationArray[i+1])
            
        return dist
    
    def __getitem__(self,key):
        dist = self.calcDistance(self.locationPerms[key])
        print(dist)
        if dist < self.threshold:
            return 1
        else:
            return 0

if __name__=='__main__':
    #oracle = Oracle(x=3,n_qubits = 10)
    #k = grover_search(oracle)
    #print(k)
    locations = np.array([[0,1],[3,4],[4,3]])
    
    
    o = TSPOracle(locations, 5)
    for i in range(6):
        o[i]
    
    
    
    
    