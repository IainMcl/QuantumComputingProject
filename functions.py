# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:22:28 2018

@author: Lewis
"""
from qc_testing import *
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
    """
    #Initialise both quantum registers
    register1 = QuantumRegister()

    not_gate = Not()
    h_gate = Hadamard()
    register2 = h_gate * not_gate  * QuantumRegister()

    #Pass register 1 through hadamard
    register1 = h_gate * register1
    register1 = oracle * register1

    #Pass again through hadamard
    final = h_gate * register1

    #Perform measurement
    k = final.measure()
    return k


# def grover_search(oracle):
#     """
#     implements grover's search algorithm at the given quantum register and
#     oracle, for just one element. Will later be extedned to multiple elements.
#     Inputs:
#     quant_register:
#     oracle: oracle function that "tags" a qubit in the quantum register
#
#     Outputs:
#     measured_state: the state which the oracle tagged
#     """
#
#     #Initialise quantum state and set it in superposition
#     n = oracle.n_qubits
#     register1 = QuantumRegister(n)
#     h_gate = Hadamard(n)
#     psi = h_gate * psi
#     oracle_0 = Oracle()
#     I = Operator(1, np.eye(2))
#     resgiter2 = QuantumRegister()
#     register.qubits = 1/np.sqrt(2)*np.array( [ 1, -1]) #will be done through unitary gate in the future: 1/sqrt(2) * (|0> - |1>)
#     #Initialise grover's itearate
#     c_fk = CUGate( oracle, n)
#     c_f0 = CUGate( oracle_0, n)
#
#     term1 = c_fk.dot(h_gate*I)
#     term2 = c_f0.dot(h_gate*I)
#     grover_iterate = term1.dot(term2)
#     #based on first paper for grover's iterate
#
#
#     #for loop for grover search. At each iteration, apply grover's diffucion operator
#
#     #After for loop, perform measurement
#
#     k = psi.measure()
#
#     pass

if __name__ == '__main__':

    #Create function
    f = Operator(1, np.array( [ [1,0], [0,-1] ]))

    k = deutsch(f)

    #ITS ALIVE WOOOOOOOO
    print(k)
