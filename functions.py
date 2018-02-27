# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:22:28 2018

@author: Lewis
"""
from qc import *
import numpy as np
def init_qubit(theta,phi):
    """
    Initialises a qubit to the following state:
    |psi> = cos(theta) * |0> + exp(i*phi) * |1>
    """
    h_gate = Hadamard()
    r_theta = PhaseShift(2 * theta)
    r_phi = PhaseShift(pi/2 + phi )
    initial_state = QuantumRegister()

    u_gate = r_phi * h_gate * r_theta * h_gate

    result = u_gate*initial_state

    return result

def deutsch(oracle):
    """
    Implements deutsch's algortithm to determine whether the oracle passed is
    balanced or not.

    Extension of deutsch algorithm to n qubits
    """
    #Extract number of qubits
    n = oracle.n_qubits

    #Initialise both quantum registers
    register1 = QuantumRegister(n)

    not_gate = Not()
    h_gate = Hadamard()
    register2 = h_gate * not_gate  * QuantumRegister()

    #Pass register 1 through hadamard gate
    register1 = h_gate * register1

    #Apply oracle to register 2

    return k

def grover_search(oracle):
    """
    implements grover's search algorithm at the given quantum register and
    oracle, for just one element. Will later be extedned to multiple elements.
    Inputs:
    quant_register:
    oracle: oracle function that "tags" a qubit in the quantum register

    Outputs:
    measured_state: the state which the oracle tagged
    """

    #Initialise quantum state and set it in superposition
    n = oracle.n_qubits
    register1 = QuantumRegister(n)
    h_gate = Hadamard(n)
    psi = h_gate * psi
    oracle_0 = Oracle()
    I = Operator(1, np.eye(2))
    resgiter2 = QuantumRegister()
    register.qubits = 1/np.sqrt(2)*np.array( [ 1, -1]) #will be done through unitary gate in the future: 1/sqrt(2) * (|0> - |1>)
    #Initialise grover's itearate
    c_fk = CUGate( oracle, n)
    c_f0 = CUGate( oracle_0, n)

    term1 = c_fk.dot(h_gate*I)
    term2 = c_f0.dot(h_gate*I)
    grover_iterate = term1.dot(term2)
    #based on first paper for grover's iterate


    #for loop for grover search. At each iteration, apply grover's diffucion operator

    #After for loop, perform measurement

    k = psi.measure()
    return k

def quantumAdder(a,b):
    """
    Implements a quantum addition circuit using a C-Not and Toffili Gates
    Takes in 2 single qubit quantum registers as input
    returns a quantum register containing the sum of the values of the 2 input registers
    
    currently implements the circuit but fails to return the right thing
    """
    x0 = QuantumRegister(a.n_qubits)
    CN = CUGate(Not(),1)
    I = Operator(3, np.eye(3))
    CNN = CUGate(Not(),2)
    reg1 = a*b*x0
    reg1 = CNN*reg1
    I = Operator(reg1.n_qubits, np.eye(3))
    CN = I%CN
    reg2 = CN*reg1
    k = reg2.measure()
    #untensorfy here
    return k

def GetQuBitofValue(theta, phi):
    """
    Implements a setter circuit, setting the value of 
    """
    register = QuantumRegister()
    H = Hadamard()
    P1 = PhaseShift(2*theta,register.n_qubits)
    P2 = PhaseShift((np.pi*0.5)+phi,register.n_qubits)
    result = H*P1*H*P2*register
    return result

def invert_average(quant_register):
    """
    Applies the inversion about the average operation to the current state, given
    the initial state. Meant to be used as a helper function for Grover's algorithm.
    """
    #Extract risze of register and define hadamard gate
    register_size = quant_register.size
    h_gate = Hadamard(register_size)

    #Set the quantum register in |0> basis
    psi0 = h_gate * quant_register

    #Eigenvalue kickback by applying control not gate to this register

    pass




if __name__ == '__main__':

    #Create function
    f = Operator(10, np.array( [ [1,0], [0,-1] ]))

    k = deutsch(f)

    #ITS ALIVE WOOOOOOOO
    print(k)
