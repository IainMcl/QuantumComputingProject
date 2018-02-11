"""
07/02/2018
Andreas Malekos - School of Physics and Astronomy, The University of Edinburgh

Quantum Computing Project
Definition of abstract classes for qubits and quantum registers

Some notes from Andreas:
-Not yet sure whih python data type to use for ket and bra vectors (probably numpy array)
-This file need to contain:
    -Abstract clss definition for qubit (ket vector)
        -Define both bra and ket (maybe add class parameter)
        -Operations between qubits
            -Tensor add
            -Tensor product
            -Measurement
            -Some sort of interface for matrix operations
            -Visualization?

            Loosely based on the QObj object found in QuTip (Quantum Toolbox in Python)
            http://qutip.org/tutorials.html

            -Does this need to be an abstract class? Don't think so, since there's only
            one type of qubit (as opposed to many different types of matrix operators )
                -Special class for fock states?
                -Special qubit class that implements general

"""

import numpy as np
from numpy.linalg import norm
from abc import ABC, abstract method

class QuBit():

    def __init__(self, n_states, state ):
        """ Potential qubit parameters:
            type: either bra or ket
            state: in which state is the qubit in
            dimensions: number of possible states the qubit can be in
        """

        #Definiton of variables that hold the total number of possible states
        #and the current state
        self.n_states = n_states
        self.state = state




    def __mul__(self, q2):
        """
        Overrides multiplication operator in the following way:
        -Case 1: Two kets -> Tensor product
        Outputs a new ket
        -Case 2: Braket -> Dot product
        Outputs a number
        -Case 3: KetBra -> Operator
        Output a matrix object (to be defined)
        (is this feasible for large qubits)
        """
        pass


class QuantumRegister():

    def __init__(self, n_qubits):
        """"
        Quantum register class. The quantum register is saved as a complex
        numpy array. Each element of the array is the amplitude of the
        corresponding state, eg. the first element is the first state, the second
        element is the second state.
        """"
        self.n_states = 2 ** n_qubits
        self.qubits = np.zeros(self.n_state, dtype = complex)
        qubits[0] = 1.0


    def set_qubit(self, a ,b):
        """

        """

    def measure(self):
        """
        Make a measurement. Square all the amplitudes and choose a random state

        """
        #Square the amplitudes
        #use np.random.choice and set probablilty to the probabiliy amplitudes calculated before


    def __mult__(self, other):
        """
        Tensor prodcut between the two quantum registers. Outputs a new quantum
        register.
        """
        #Result is tensor product of the qubits in each state
        temp_result = np.kron(self.qubits, other.qubits)

        #Result has to be normalized
        result_normalized = temp_result/norm(temp_result)

        #Creaete quantum register object for result
        qmr_result = QuantumRegister(self.n_qubits*other.n_qubits )
        qmr_result.qubits = result_normalized

        return qmr_result





class Operator():
    """
    Class that defines a quantum mechanical operator. The operator is
    a matrix. Only non zero elements are saved in a list of triplets. For each
    element in the list (i,j,Mij):
        -i,j: row nad column number
        -Mij: of the operator at that row and column

    """

    def __init__(self, size):

        pass

    def apply(self, quant_register):
    """
    applies the operator to the given quantum register. Returns another quantum
    register.
    """


class Hadamard(Operator):

    def __init__(self, size):
