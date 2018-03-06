"""
Testing grover. Parts are lazily implemented (such as controlled - n gates.
Based on Quantum Computer Science book, Chapter 4
"""

# Test implementation of Grover's algorithm for 3 qubits with a 3 qubit oracle.
from qc_simulator.qc import *
import numpy as np



def grover5():

    # Define oracle for 5 qubits, where a = 10010 is the "magic number"
    not_gate = Not()
    h_gate = Hadamard()
    I = Operator(base=np.eye(2,2))

    # Define 5-controlled-Not gate (the crude way for now)
    c5_not = CUGate(not_gate, 5)

    oracle_gate = (I % not_gate % not_gate % I % not_gate % I) *\
        c5_not * ( I % not_gate % not_gate % I % not_gate % I)

    # Define the inversion about average operator
    # W = H**n % X**n % (cn^-1 Z) * X%%n * H%%

    #define z and control z_gates
    z = PhaseShift(np.pi)
    control_z = CUGate(z, 5)
    h_n_gate = Hadamard(6)
    not_n_gate = Not(6)

    W = h_n_gate * not_n_gate * control_z * not_n_gate * h_n_gate
    test = W * QuantumRegister(6)
    test.remove_aux(1/np.sqrt(2))

    G = W * oracle_gate

    # Define the input and ancillary quantum registers
    input_register = Hadamard(5) *  QuantumRegister(5)
    aux = h_gate * not_gate * QuantumRegister()
    register = input_register

    # Loop and apply grover operator iteratively
    for i in range(10):
        # Add auxilary qubit to register
        register = register * aux

        # Apply grover iteration
        register = G * register

        #Extrac input register and reset auxillary qubit (hacky way)
        register.remove_aux(1/np.sqrt(2))

        aux =  h_gate * not_gate * QuantumRegister()


    n = register.measure()

    return n



for i in range(10):
    test = grover5()
    print(test)



