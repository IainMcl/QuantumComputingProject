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
    
    #entangle registers
    reg = reg1*reg0
    
    #gate 1
    gate1 = H%Hn
    print(Hn.n_qubits)
    print(H.n_qubits)
    print(H.n_qubits+Hn.n_qubits)
    print(gate1.matrix.toarray())
    print(gate1.n_qubits)
    gate1.n_qubits = H.n_qubits + Hn.n_qubits
    reg = gate1*reg
    
    #reflection gate > I|w>-2(<e|w>)|e>
    
    R = Operator(n,-1*np.eye(n-1))
    
    #grover diffusion gate
    
    grov_diff = Hn*R*Hn
    
    for i in range(np.round(np.sqrt(n))):
        reg = grov_diff*oracle*reg
        
    k = reg.measure()
    return k

if __name__=='__main__':
    oracle = Oracle(x=3,n_qubits = 10)
    k = grover_search(oracle)
    print(k)