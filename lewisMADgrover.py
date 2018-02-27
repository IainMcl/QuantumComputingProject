# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:37:45 2018

@author: Lewis
"""

from qc import *
from functions import *
import numpy as np

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

class Oracle():
    def __init__(self, x):
        self.x = x
    def __getitem__(self, key):
        if key == x:
            return 1
        else:
            return 0


if __name__=='__main__':
    oracle = Oracle(x=3,n_qubits = 10)
    k = grover_search(oracle)
    print(k)