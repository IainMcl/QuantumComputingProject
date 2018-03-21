# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:37:45 2018

@author: Lewis
"""

#from qc import *

import numpy as np
from scipy.spatial import distance
from sympy.utilities.iterables import multiset_permutations
from itertools import permutations
import math
import matplotlib.pyplot as plt
from qc_simulator.grover import *
from qc_simulator.FunctionalOracle import *


class TSPOracle():
    """
    This class implements a series of functions used to create a 
    functional oracle object for the Travelling saleman problem
    It stores permutations of different locations

    """
    def __init__(self, locations,threshold):
       
        """
        Class constructor
        :param locations: numpy array of 2-d coordinates
        :param threshold: Threshold distance
        """
        self.locations = locations[1:locations.size-1]#list of all stops excluding the first one

        self.start = locations[0]#fixes the starting stop
        self.locationPerms = tuple(permutations(self.locations))#creates a list of all permuations of locations
        self.locationPerms += self.locationPerms[:2**math.ceil(math.log(len(self.locationPerms),2))-len(self.locationPerms)]
        self.threshold = threshold#threshold distance
        self.size = int(math.log(len(self.locationPerms),2)) #size of oracle

    def calcDistance(self, locationArray):
        """
        calculates the distance needed to complete a route specified by the input array
        :params locationArray: Array of locations in the order of the route you want to calculate the distance for
        :return self.dist: The Euclidian distance of the route specified
        """
        self.dist = distance.euclidean(self.start,locationArray[0]) + distance.euclidean(self.start,locationArray[-1])#distance from start to first stop and last to first stop

        for i in range(len(locationArray)-1):#all other distances
            self.dist += distance.euclidean(locationArray[i],locationArray[i+1])

        return self.dist

    def __getitem__(self,key):
        """
        Overrides the getitem method to act as the function f(x)
        :params key: key specifying the permutation 
        :return:  0 or 1 if the specific permuation is under the distance threshold
        """
        if key >= len(self.locationPerms):
            return 0
        dist = self.calcDistance(self.locationPerms[key])



        if dist < self.threshold:
            return 1
        else:
            return 0

    def plot_locations(self, key):
               
        """
        plots the specified route permutation
        :params key: specifies the permuation to be plotted
        """
        try:
            locs = np.swapaxes(self.locationPerms[key],0,1)
            print(locs)
        except IndexError:
            print("Undefined route, try running Grover again :(")
            return 0
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
    locations = np.array([[2,3],[3,6],[9,2],[3,2],[5,4]])
    #locations = 10*np.random.rand(6,2)


    d=18#threshold
    o = TSPOracle(locations, d)#create oracle object
    dists = np.zeros(len(o.locationPerms))#run it classically
    for i in range(len(o.locationPerms)):
        o[i]
        dists[i] = o.dist

    k = (dists < d).sum()#get number of tagged states/number of distances below the threshold classically
    #print(min(dists))
    #o = OracleFunction(4)

    #o.plot_locations(np.argmin(dists))
    Of = FunctionalOracle(o, o.size)#creates the functional oracle
    outputs = np.zeros(100,dtype=int)
    for i in range(10):#runs grover for the oracle 10 times

        outputs[i] = grover(Of,k=k)[1]
        print(outputs[i])
        print
    i = np.argmax(np.bincount(outputs))#best result
    o.plot_locations(i)
    o.plot_locations(np.argmin(dists))
    print(np.bincount(outputs))
    print("\n")
    print("N_qubits = {}".format(o.size))
    print("k = {}".format(k))
    print("Correct base state: {}".format(np.argmin(dists)))
    print("Grover base state: {}".format(i))
    print("Average distance: {}".format(np.average(dists)))
    print("Grover minimum distance: {}".format(o.calcDistance(o.locationPerms[i])))
    print("Classical minimum distance: {} ".format(min(dists)))
