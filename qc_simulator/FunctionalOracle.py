# -*- coding: utf-8 -*-
"""
Demonstation of the implementation of an Oracle using functions rather than
matrices.
"""
from qc import *
import math
class FunctionalOracle(Operator):
    def __init__(self, oracle_function, n_qubits):
        self.function = oracle_function
        self.n_qubits = n_qubits

    def __mul__(self,rhs):
        """
        Overrides the multiplication method to define a functional multiplication without a matrix
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
        #if key == self.x:
        #if key/2 in [self.x, self.x+0.5]:
        if self.x in [key/2, (key-1)/2]:
            return 1
        else:
            return 0

class Oracle4k1():
    # If a prime is of form 4k+1, it can be expressed as a^2 + b^2 for only one set of a and b
    # Oracle 4k1 finds those two values
    def __init__(self, x):
        self.x = x
    def __getitem__(self, key):
        if (key/2)**2 >self.x:
            return 0
        else:
            if math.sqrt(self.x-(key/2)**2) == round(math.sqrt(self.x-(key/2)**2)):
                #print (key)
                return 1
            elif math.sqrt(self.x-((key-1)/2)**2) == round(math.sqrt(self.x-((key-1)/2)**2)):
                return 1
            else:
                return 0



if __name__ == "__main__":
    '''
    q = QuantumRegister(5)
    H = Hadamard(5)
    q = H*q

    o = OracleFunction(10)
    f = FunctionalOracle(o,5)

    o = Oracle4k1(41)
    f = FunctionalOracle(o,n)
    print("\n")
    out = f*q
    print(out)
    print("\n")
    '''



    n=5
    o = Oracle4k1(41)
    #o = OracleFunction(4)
    f = FunctionalOracle(o,n)
    #oracle2=oracle_single_tag(n,5)
    #oracle3=oracle_single_tag(n,10)
    #oracle4=oracle_single_tag(n,15)
    #oracle=oracle1*oracle2*oracle3*oracle4
    #oracle=oracle1*oracle2
    for i in range(40):
        reg = grover(f, k=2)
        print(reg[0].measure())
