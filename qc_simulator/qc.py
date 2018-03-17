"""
07/02/2018
Andreas Malekos, Gabriel Hoogervorst, John Harbottle, Lewis Lappin, Huw Haigh
School of Physics and Astronomy, The University of Edinburgh

Classes for quantum register and operators. Implement the abstract classes
defined in qc_abastract. The Quantum register is implemented as a 1D numpy array
and the matrix gates as sparse matrices, using the Compressed Sparse Column
matrix format (csc).

"""

import numpy as np
from numpy.linalg import norm
from scipy.sparse import identity as sparse_identity
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix, kron
from math import pi
import matplotlib.pyplot as plt
from copy import deepcopy
#from qutip import *

#import abstract classes
from qc_simulator.qc_abstract import *
#from qc import

class QuantumRegister(AbstractQuantumRegister):
    """
    Quantum register class. The quantum register is saved as a complex
    numpy array. Each element of the array is the amplitude of the
    corresponding state, eg. the first element is the first state, the second
    element is the second state.
    """


    def __init__(self, n_qubits=1, isempty=False):
        """
        Class constructor
        :param n_qubits: number of qubits in quantum register
        :param isempty: parameter that tells whether the quantum register should be empty.
        set to False by default.
        """
        self.n_states = int(2 ** n_qubits)
        self.n_qubits = n_qubits
        self.base_states = np.zeros(self.n_states, dtype=complex)
        # If isempty = False, initialise in ground state
        if not isempty:
            self.base_states[0] = 1.0

    def measure(self):
        """
        Make a measurement. Square all the amplitudes and choose a random state.
        Outputs an integer representing the state measured (decimal system).
        :return: state, integer that corresponds to the number of the state measured, in decimal format.
        """
        # Calculate probabilities
        probabilities = np.zeros(self.n_states)
        for i in range(self.n_states):
            probabilities[i] = norm(self.base_states[i]) ** 2

        # Choose a random state
        n = int(self.n_states)
        state =  int (np.random.choice(n, p=probabilities) )

        return state

    def __mul__(self, other):
        """
        Overrides multiplication operator to define tensor product between two quantum registers.
        :param other: right hand side quantum register
        :return: qmr_result, resulting quantum register.
        """

        # Check if other is of the right tyoe
        if isinstance(other, QuantumRegister):
            # Result is tensor product of the qubits in each state
            temp_result = np.kron(self.base_states, other.base_states)

            # Result has to be normalized
            result_normalized = temp_result / norm(temp_result)

            # Create quantum register object for result
            qmr_result = QuantumRegister(self.n_qubits + other.n_qubits)
            qmr_result.base_states = result_normalized

            return qmr_result
        else:
            raise TypeError('Multiplication not defined between quantum register and {}.'.format(type(other)))

    def __str__(self):
        """
        Overrides str method to print out quantum register in braket notation
        :return: rep : string reply
        """
        base_states = self.base_states
        l = len(base_states)
        n_qubits = self.n_qubits
        if base_states[0] != 0:
            rep = '({0:+.2f})'.format(base_states[0]) + "*|" + np.binary_repr(0, n_qubits) + "> "
        else:
            rep = ''

        for i in range(1, l):
            if base_states[i] == 0:
                continue
            rep = rep + '({0:+.2f})'.format(base_states[i]) + "*|" + np.binary_repr(i, n_qubits) + "> "

        return rep


    def remove_aux(self, a=1/np.sqrt(2)):
        """
        Removes auxillary qubit from quantum register.
        Requires previous knowledge of auxilary qubit. Usage is meant for phase
        kickback operations.
        :param a:
        """

        # Remove every second element of the base state array. Then divide every element by a
        base_states = self.base_states
        new_base_state = base_states[::2] / a
        self.base_states = new_base_state
        self.n_qubits = int( self.n_qubits - 1)
        self.n_states = int( self.n_states / 2)

    def normalise(self):
        """
        Normalise coefficients of qubits array
        """
        # Add tolerance to remove extremely small floating point calculation errors
        tol = 10 ** (-8)
        filter = abs(self.base_states) >= tol
        self.base_states = self.base_states * filter

        base_states_normalised = self.base_states / norm(self.base_states)

        self.base_states = base_states_normalised

    def plot_register(self, show=True):
        """
        Produce bar graph of quantum register.
        :param show: Boolean flag that if set to true, shows the bar graph.
        :return: ax, axis handle object.
        """
        ax = plt.bar(np.arange(2**self.n_qubits),np.absolute(self.base_states))
        if show:
            plt.show()
        return ax

    def plot_bloch(self, is3D=False):
        """
        Creates a bloch sphere of the quantum register.
        :param is3D: lewis has to add comments for this
        """
        if is3D:
            b = Bloch3d()
        else:
            b = Bloch()
        objs = []
        for i in range(self.n_qubits):
            obj = Qobj(self.base_states[2*i:2*i+2])
            b.add_states(obj)
            objs.append(obj)
        #b.add_states(objs)
        b.show()

class Operator(AbstractOperator):
    """
    Class that defines a quantum mechanical operator. Implments abstract class
    OperatorAbstract. The operator is stored as a square sparse matrix.
    """

    def __init__(self, n_qubits=1, base=np.zeros((2, 2))):
        """
         Class constructor
         :param n_qubits: number of qubits operator operates on
         :param base: base matrix
        """
        self.n_qubits = n_qubits
        self.size = 2 ** n_qubits
        self.matrix = self.__create_sparse_matrix(self.n_qubits, base)

    def __create_sparse_matrix(self, n_qubits, base):
        """
        Create matrix by taking successive tensor producs between for the total
        number of qubits.
        :param n_qubits: number of qubits operator operates on
        :param base: base matrix
        :return: sparse matrix (csc format)
        """
        base_complex = np.array(base, dtype=complex)
        result = lil_matrix(base_complex)

        if n_qubits == 1:
            result = csc_matrix(result)

            return result
        else:
            for i in range(n_qubits - 1):
                result = kron(result, base)

            result = csc_matrix(result)
            return result

    def __mul__(self, rhs):
        """
        Overrides multiplication operator and defined the multiplication between
        two operators and an operator and a quantum register.
        :param rhs: right hand side, can be either operator or quantum register
        :return: Operator if rhs is of type Operator then return Operator. If it's of
        type QuantumRegister, then return a quantum register object.
        """
        if isinstance(rhs, QuantumRegister):
            # Apply operator to quantum register
            # check if number of states is the same
            if rhs.n_qubits != self.n_qubits:
                raise ValueError(
                    'Number of states do not correspnd: rhs.n_qubits = {}, lhs.n_qubits = {}'.format(rhs.n_qubits,
                                                                                                     self.n_qubits))

            # Otherwise return a new quantum register
            result = QuantumRegister(rhs.n_qubits)

            # Calculate result. Check if matrix is sparse or not first. If sparse
            # use special sparse dot product csc_matrix.dot
            result.base_states = self.matrix.dot(rhs.base_states.transpose())

            # Normalise result
            result.normalise()
            return result

        if isinstance(rhs, Operator):
            """
            Matrix multiplication between the two operators
            """
            if rhs.size != self.size:
                raise ValueError(
                    'Operators must of of the same size: rhs.size = {} lhs.size = {} '.format(rhs.size, self.size))

            # Otherwise take dot product of
            result = Operator(self.n_qubits)
            result.matrix = self.matrix.dot(rhs.matrix)
            return result

        else :
            " Raise type error if the right type isn't provided"
            raise TypeError(
                'Multiplication not defined for Operator and {}.'.format(type(rhs))
            )



    def __mod__(self, other):
        """
        Overrides "%" operator to define tensor product between two operators.
        :param other: Operator object, right hand side
        :return: Operator object
        """

        # Tensor product between the two operators
        if isinstance(other, Operator):
            result = Operator(self.n_qubits + other.n_qubits)
            result.matrix = csc_matrix(kron(self.matrix, other.matrix))
            return result
        else:
            raise TypeError(
                'Operation not defined between operator and {}.'.format(type(other))
            )

    def __str__(self):
        """
        Provides method to pring out operator.
        :return: rep: String that corresponds the __str__() method of the
        numpy array.
        """
        return self.matrix.toarray().__str__()

    def dag(self):
        """
        Computes hermitian transpose of operator.
        :return: herm_transpse: Hermitian transpose of operator
        """

        herm_transpose = Operator(self.n_qubits)
        herm_transpose.matrix = self.matrix.getH()

        return herm_transpose

class Hadamard(Operator):
    """
    Class that defines hadamard gate. This class extends the Operator class.
    """

    def __init__(self, n_qubits=1):
        # Define "base" hadamard matrix for one qubit and correponding sparse matrix
        self.base = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        super(Hadamard, self).__init__(n_qubits, self.base)

class PhaseShift(Operator):
    """
    Class that implements phase shift gate
    """

    def __init__(self, phi, n_qubits=1):
        self.base = np.array([[1, 0], [0, np.exp(1j * phi)]])
        super(PhaseShift, self).__init__(n_qubits, self.base)

class Not(Operator):
    """
    Class that implements not gate.
    """

    def __init__(self, n_qubits=1):
        self.base = np.array([[0, 1], [1, 0]])
        super(Not, self).__init__(n_qubits, self.base)

class CUGate(Operator):
    """
    Class that implements a controlled U gate.
    """

    def __init__(self, base, n_control=1, num_of_i=0):
        """
        Class constructor.
        :param base: base Operator U
        :param n_control: number of control qubits
        :param num_of_i: number of empty lines between control and target qubit.
        """
        self.n_control = n_control
        self.n_qubits = 1 + self.n_control + num_of_i
        self.size = 2 ** (self.n_control + 1 + num_of_i)
        self.num_of_i = num_of_i
        self.matrix = self.__create_sparse_matrix(base)

    def __create_sparse_matrix(self, base):
        """
        Creates spasrse matrix according to how many target qubits we have.
        Matrix is constructed using the 'lil' format, which is better for
        incremental construction of sparse matrices and is then converted
        to 'csc' format, which is better for operations between matrices
        :param base: base operator U
        :return: sparse matrix
        """

        # Create sparse hadamard matrix
        base_matrix = lil_matrix(base.matrix)
        print('Size of base sparse matrix is {}.'.format(base_matrix.toarray().shape))

        # Create full sparse identity matrix
        sparse_matrix = sparse_identity(self.size, dtype=complex, format='lil')
        print('Size of full sparse matrix is {}.'.format(sparse_matrix.toarray().shape))
        # print(self.size)
        # print(base.matrix.toarray().size)


        if self.num_of_i == 0:
            # "Put" dense hadamard matrix in sparse matrix
            target_states = 2
            sub_matrix_index = self.size - target_states
            sparse_matrix[sub_matrix_index:, sub_matrix_index:] = base_matrix

            # Convert to csc format
            c_gate = csc_matrix(sparse_matrix)

            return c_gate
        else:
            # Put sub matrix in corner of big matrix
            i_sparse = sparse_identity(2 ** (self.num_of_i), format='lil')
            bottom_right_quarter = kron(i_sparse, base.matrix, format='lil')

            n = int(bottom_right_quarter.shape[0])
            sparse_matrix[-n:, -n:] = bottom_right_quarter
            c_gate = csc_matrix(sparse_matrix)

            return c_gate

# class CNotGate(CUGate):
#     """
#     Class that implements controlled-not gate
#     """
#
#     def __init__(self, num_of_i):
#         self.base = Not()
#         self.n_control = 1
#         super(CNotGate, self).__init__(self.base, self.n_control, num_of_i)

class IdentityGate(Operator):
    """
    Class that implements identity operator.
    """
    def __init__(self, n_qubits = 1):
        super(IdentityGate, self).__init__(n_qubits, base=np.eye(2,2))

class fGate(Operator):
    """
    Class that implements an f-Gate, Uf. The action of Uf is defined as follows
    Uf*|x>*|y> = |x>*|(y+f(x))%2>
    """
    def __init__(self, f, n):
        """
        Class constructor:
        :param f: callable function f defined from {0,1}^n -> {0,1}
        :param n: number of states n acts on
        """
        self.f = f
        self.n_qubits = n + 1
        self.size = 2**(self.n_qubits)
        self.matrix = self.__f_matrix()

    def __f_matrix(self):
        """
        Constructs a numpy matrix that corresponds to the function
        evaluation. The matrix is then converted to a sparse array.
        """
        matrix_full = np.eye(self.size, self.size)
        n = int(self.size/2)
        f = self.f

        for i in range(n):
            # Loop over the rows 2 at a time and exchange only if f(x)
            # returns 1.
            if f(i) == 1:
                print('Switching rows {} and {}.'.format(2*i, 2*i+1))
                temp = deepcopy(matrix_full[2*i,:])
                temp2 = deepcopy(matrix_full[2*i + 1,:])
                matrix_full[2*i,:] = temp2
                matrix_full[2*i + 1, : ] = temp

        return csc_matrix(matrix_full)
