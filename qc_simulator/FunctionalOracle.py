# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:49:33 2018

@author: Lewis
"""
from qc import *
class FunctionalOracle(Operator):
    def __init__(self, oracle_function, n_qubits):
        self.function = oracle_function
        self.n_qubits = n_qubits
        
    def __mul__(self,rhs):
        """
        if False == True: #not isinstance(rhs, QuantumRegister):
            raise TypeError("Cannont Multiply a functional operator by {}".format(type(rhs)))
        else:
        """
        output = np.zeros(rhs.base_states.size,dtype=complex)
        for i in range(rhs.base_states.size):
            if self.function[i]==1:
                output[i]=-rhs.base_states[i]
            if self.function[i]==0:
                output[i]=rhs.base_states[i]
            '''
            output[i] = -1*rhs.base_states[i]*self.function[i]-1*self.function[i]
            output[i] +=1
            '''
                
        outReg = QuantumRegister(self.n_qubits)
        outReg.base_states = output
        outReg.normalise()
        return outReg
                
                
class OracleFunction():
    def __init__(self, x):
        self.x = x
    def __getitem__(self, key):
        if key == self.x:
            return 1
        else:
            return 0


if __name__ == "__main__":
    
    q = QuantumRegister(5)
    H = Hadamard(5)
    q = H*q
    o = OracleFunction(10)
    f = FunctionalOracle(o,5)
    
    print("\n")
    out = f*q
    print(out) 
    print("\n")



    '''
    n=4
    o = OracleFunction(10)
    f = FunctionalOracle(o,5)
    #oracle2=oracle_single_tag(n,5)
    #oracle3=oracle_single_tag(n,10)
    #oracle4=oracle_single_tag(n,15)
    #oracle=oracle1*oracle2*oracle3*oracle4
    #oracle=oracle1*oracle2

    reg = grover(f)
    print(reg[0].measure())
    '''