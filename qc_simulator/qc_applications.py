# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:37:45 2018

@author: Lewis
"""

#from qc import *
#from functions import *
#import numpy as np
#from scipy.spatial import distance
#from sympy.utilities.iterables import multiset_permutations
#from itertools import permutations
#import math
#import matplotlib.pyplot as plt
#from grover import *
#from FunctionalOracle import *

class TSPOracle():
    def __init__(self, locations,threshold):
        self.locations = locations[1:locations.size-1]
        self.start = locations[0]
        self.locationPerms = tuple(permutations(self.locations))
        self.locationPerms += self.locationPerms[:math.ceil(math.log(len(self.locationPerms),2))-len(self.locations)]
        print(len(self.locationPerms))
        self.threshold = threshold
        
    def calcDistance(self, locationArray):
        self.dist = distance.euclidean(self.start,locationArray[0]) + distance.euclidean(self.start,locationArray[-1])
        for i in range(len(locationArray)-1):
            self.dist += distance.euclidean(locationArray[i],locationArray[i+1])
            
        return self.dist
    
    def __getitem__(self,key):
        if key >= len(self.locationPerms):
            return 0
        dist = self.calcDistance(self.locationPerms[key])
        
        if dist < self.threshold:
            return 1
        else:
            return 0
        
    def plot_locations(self, key):
        locs = np.swapaxes(self.locationPerms[key],0,1)
        plt.scatter(locs[0],locs[1])
        plt.scatter(self.start[0],self.start[1])
        plt.text(self.start[0]+0.06,self.start[1]-0.05, "1")
    
        for i in range(locs.shape[1]):
            plt.text(locs[0][i]+0.06,locs[1][i]-0.05,str(i+2))
            
        plt.show()
if __name__=='__main__':
    #oracle = Oracle(x=3,n_qubits = 10)
    #k = grover_search(oracle)
    #print(k)
    locations = np.array([[2,1],[0,5],[2,3],[1,4],[3,2]])
    
    n=locations.size-1
    o = TSPOracle(locations, 13)
    dists = []
    for i in range(1):
        o[i]
        dists.append(o.dist)
        
    print(min(dists))
    #o = OracleFunction(4)
    
    o.plot_locations(np.argmin(dists))
    
    Of = FunctionalOracle(o, 7)
    i = grover(Of,k=3)[1]
    print(i)
    print(o.calcDistance(o.locationPerms[i]))
    print(min(dists))
    o.plot_locations(i)
    
    
        
    
    
    
    
    