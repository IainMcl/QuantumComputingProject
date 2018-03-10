# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:37:45 2018

@author: Lewis
"""

from qc_simulator.qc import *
from qc_simulator.functions import *
import numpy as np
from scipy.spatial import distance
from sympy.utilities.iterables import multiset_permutations
from itertools import permutations
import math
import matplotlib.pyplot as plt

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
        self.locations = locations[1:locations.size-1]
        self.start = locations[0]
        self.locationPerms = tuple(permutations(self.locations))
        self.threshold = threshold
        
    def calcDistance(self, locationArray):
        self.dist = distance.euclidean(self.start,locationArray[0]) + distance.euclidean(self.start,locationArray[-1])
        for i in range(len(locationArray)-1):
            self.dist += distance.euclidean(locationArray[i],locationArray[i+1])
            
        return self.dist
    
    def __getitem__(self,key):
        dist = self.calcDistance(self.locationPerms[key])
        
        if dist < self.threshold:
            return 1
        else:
            return 0
        
    def plot_locations(self, key):
        locs = np.swapaxes(self.locationPerms[key],0,1)
        plt.scatter(locs[0],locs[1])
        plt.scatter(self.start[0],self.start[1])
        labels = np.arange(1, locs[0].size+3)
        for i in range(labels.size-2):
            plt.text(locs[0][i],locs[1][i],str(i+1))
if __name__=='__main__':
    #oracle = Oracle(x=3,n_qubits = 10)
    #k = grover_search(oracle)
    #print(k)
    locations = np.array([[0,1],[3,4],[4,3],[2,1],[5,2],[1,2],[2,3]])
    
    n=locations.size-1
    o = TSPOracle(locations, 14.5)
    dists = []
    for i in range(720):
        o[i]
        dists.append(o.dist)
        
    print(min(dists))
    o.plot_locations(np.argmin(dists))
    
        
    
    
    
    
    