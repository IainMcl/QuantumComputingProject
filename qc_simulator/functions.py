# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:22:28 2018

@author: Lewis
"""
#!/usr/bin/env python3

from qc import *
import numpy as np
import math


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

def oracle_single_tag(n, tag):
    n_qubits=n+1
    # Convert tag to binary string
    bina_str=np.binary_repr(tag, n)
    # Convert binary string to list of integers
    binalist=[int(s) for s in bina_str]
    # Reverse list as operators are applied in opposite order
    binalist=binalist[::-1]

    # Initiate base gates
    not_gate = Not()
    I = Operator(base=np.eye(2,2))

    # Initiate the prep gate with if statement
    '''
    if binalist[0]==1:
        prep_gate=I
    elif binalist[0]==0:
        prep_gate=not_gate
    '''
    prep_gate=I
    # For loop to create whole gate
    for i in binalist:
        if i==1:
            prep_gate= I % prep_gate
        if i==0:
            prep_gate= not_gate % prep_gate

    cn_not = CUGate(not_gate, n_qubits-1)

    oracle_gate=prep_gate*cn_not*prep_gate
    return oracle_gate

def build_nc_not(n):
    """
    Builds n controleld not gate
    :param n: number of control qubits
    :return: cn_gate, controlled not gate
    """

    # Iniate the three base gates
    not_gate = Not()
    c_c_not=build_c_c_not()
    c_not=CUGate(not_gate)

    # Initialise total number of quibts
    n_qubits = n+1


    # Two cases, n is even and n odd
    # Num will be the number of gates necessary
    # If odd
    if (n_qubits%2)==1:
        num_of_gates = n_qubits-2
    if (n_qubits%2)==0:
        num_of_gates = n_qubits-1

    # Initiate the gates list
    gates = np.empty(2 * num_of_gates - 1, dtype=Operator)

    # Define first column of gates
    gates[0] = c_c_not % IdentityGate(n_qubits-3)

    # Construct n_not gate in for Loop
    for i in range(1, num_of_gates-1):

        # Check if we are on an even or odd step
        if num_of_gates%2 == 0:
            num_of_i_above = 2 * (i-1)
            num_of_i_below = n_qubits - num_of_i_above - 3

            I_above = IdentityGate(num_of_i_above)
            I_below = IdentityGate(num_of_i_below)

            gates[i] = I_above % c_c_not % I_below
        else:
            num_of_i_above = 2 * i
            num_of_i_below = n_qubits - num_of_i_above - 1

            I_above = IdentityGate(num_of_i_above)
            I_below = IdentityGate(num_of_i_below)

            gates[i] = I_above % not_gate % I_below




    # Define middle column of gates
    gates[num_of_gates-1] = IdentityGate(n_qubits-3) % c_c_not

    # Fill out the rest of the array
    gates[num_of_gates: ] = np.flip(gates[:num_of_gates-1], axis=0)

    # Complete gate is the multiplication of everything inside the array
    cn_gate = np.prod(gates)

    return cn_gate

def build_nc_z(n):
    """
    Builds an n controlled not gate
    Total number of qubits is therefore n+1
    """
    # Iniate the three base gates
    z_gate = PhaseShift(np.pi)
    c_c_z = CUGate(z_gate, 2)
    not_gate = Not()
    c_c_not=build_c_c_not()
    c_not=CUGate(not_gate)
    #I = Operator(base=np.eye(2,2))

    # Initialise total number of quibts
    n_qubits = n+1


    # Two cases, n is even and n odd
    # Num will be the number of gates necessary
    # If odd
    if (n_qubits%2)==1:
        num_of_gates = n_qubits-2
    if (n_qubits%2)==0:
        num_of_gates = n_qubits-1

    # Initiate the gates list
    gates = np.empty(2 * num_of_gates - 1, dtype=Operator)

    # Define first column of gates
    gates[0] = c_c_not % IdentityGate(n_qubits-3)

    # Construct n_not gate in for Loop
    for i in range(1, num_of_gates-1):

        # Check if we are on an even or odd step
        if num_of_gates%2 == 0:
            num_of_i_above = 2 * (i-1)
            num_of_i_below = n_qubits - num_of_i_above - 3

            I_above = IdentityGate(num_of_i_above)
            I_below = IdentityGate(num_of_i_below)

            gates[i] = I_above % c_c_not % I_below
        else:
            num_of_i_above = 2 * i
            num_of_i_below = n_qubits - num_of_i_above - 1

            I_above = IdentityGate(num_of_i_above)
            I_below = IdentityGate(num_of_i_below)

            gates[i] = I_above % not_gate % I_below




    # Define middle column of gates
    gates[num_of_gates-1] = IdentityGate(n_qubits-3) % c_c_z

    # Fill out the rest of the array
    gates[num_of_gates: ] = np.flip(gates[:num_of_gates-1], axis=0)

    # Complete gate is the multiplication of everything inside the array
    cnz_gate = np.prod(gates)

    return cnz_gate



if __name__ == '__main__':
    n = 3
    not_gate = Not()

    hacky_cn = CUGate(not_gate, n)
    proper_cn = build_nc_not(n)

    print(hacky_cn)
    print(proper_cn)

