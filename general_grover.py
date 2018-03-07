from qc_simulator.qc import *
from qc_simulator.functions import *
import numpy as np
import math


def grover_gen(oracle):
    '''
    Implements a generic grover search, with the oracle as input
    '''

    # Save the oracle gate and n_qubits
    oracle_gate = oracle

    # The oracle has one more qubit than the number of qubits in the register
    n_qubits=oracle.n_qubits-1

    # Define basis gates
    not_gate = Not()
    h_gate = Hadamard()
    z = PhaseShift(np.pi)
    control_z = CUGate(z, n_qubits-1)
    h_n_gate = Hadamard(n_qubits+1)
    not_n_gate = Not(n_qubits+1)
    I=IdentityGate()

    # Create the reflection round average gate
    W = h_n_gate * not_n_gate * (control_z % I) * not_n_gate * h_n_gate

    # Define the input and ancillary quantum registers
    input_register = Hadamard(n_qubits) * QuantumRegister(n_qubits)
    aux = h_gate * not_gate * QuantumRegister()
    register = input_register

    # Loop and apply grover operator iteratively
    #n = math.ceil( math.sqrt(n_qubits) )*3
    n= math.ceil(math.sqrt(2**n_qubits)/2)
    for i in range(n):
        #register.plot_register()
        # Add auxilary qubit to register
        register = register * aux

        # Apply grover iteration
        #register = oracle_gate * register
        #register = oracle_gate2 * register
        register=oracle * register
        register.remove_aux(1/np.sqrt(2))
        register = register* aux
        register = W * register
        #register.remove_aux(1/np.sqrt(2))

        # Extract input register and reset auxillary qubit (hacky way)
        register.remove_aux(1/np.sqrt(2))

        aux = h_gate * not_gate * QuantumRegister()

    #register.plot_register()
    measurement = register.measure()

    return measurement


def main():

    n=5
    oracle1=oracle_single_tag(n,1)
    oracle2=oracle_single_tag(n,5)
    #oracle3=oracle_single_tag(n,10)
    #oracle4=oracle_single_tag(n,15)
    #oracle=oracle1*oracle2*oracle3*oracle4
    oracle=oracle1*oracle2
    for i in range(10000):
        measurement=grover_gen(oracle)
        #print(measurement)
        #print("Grover!")


main()
