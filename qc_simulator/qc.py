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
from qutip import *

#import abstract classes
from qc_abstract import *

class QuantumRegister(QuantumRegisterAbstract):
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

    def shift(self, n):
        """
        Implements + n (mod2^n_states) by rolling the qubit array by n.
        :param n: number of shifts
        """
        self.base_states = np.roll(self.base_states, n)


    def remove_aux(self, a):
        """
        Removes auxillary qubit from quantum register. Requires previous knowledge of auxilary qubit.
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

class Operator(OperatorAbstract):
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
        self.matrix = self.__create_full_matrix(self.n_qubits, base)
        # self.sparce_matrix = coo_matrix(np.zeros( ( self.size, self.size) ) )  not sure that we need thsi right now

    def __create_full_matrix(self, n_qubits, base):
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

    def __init__(self, base, n_control=1, n_target=1, num_of_i=0):
        """
        Class constructor.
        :param base: base Operator U
        :param n_control: number of control qubits
        :param n_target: number of target qubits
        :param num_of_i: number of empty lines between control and target qubit.
        """
        self.n_control = n_control
        self.n_target = n_target
        self.n_qubits = self.n_target + self.n_control
        self.size = 2 ** (self.n_control + self.n_target + num_of_i)
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

        # Create full sparse identity matrix
        sparse_matrix = sparse_identity(self.size, dtype=complex, format='lil')

        if self.num_of_i == 0:
            # "Put" dense hadamard matrix in sparse matrix
            target_states = 2 ** self.n_target
            sub_matrix_index = self.size - target_states
            sparse_matrix[sub_matrix_index:, sub_matrix_index:] = base_matrix

            # Convert to csc format
            c_gate = csc_matrix(sparse_matrix)

            return c_gate
        else:
            # Put sub matrix in corner of big matrix
            i_sparse = sparse_identity(2 ** self.num_of_i, format='lil')
            bottom_right_quarter = kron(i_sparse, base.matrix)

            sparse_matrix[int(self.size / 2):, int(self.size / 2):] = bottom_right_quarter
            c_gate = csc_matrix(sparse_matrix)

            return c_gate

    def apply(self, control, target):
        """
        Applies the "V" gate to the target register according to the values in the control register
        :param control: control register
        :param target: target register
        :return: result -> resulting quantum register
        """
        result = QuantumRegister(target.n_qubits)

        # Base is applied only if the last element of the qubits np array is non zero.
        if control.qubits[-1] != 0:
            result = self.base * target
        else:
            result = target

        return result

class IdentityGate(Operator):
    """
    Class that implements identity operator.
    """
    def __init__(self, n_qubits = 1):
        super(IdentityGate, self).__init__(n_qubits, base=np.eye(2,2))

class fGate(Operator):
    """
    Class that implements the Uf operator, where f is a black box function f : {0,1}^n -> {0,1}, such that
    Uf|x>|y> -> |x>|(y + f(x))(mod2)>
    """

    def __init__(self, n_control, f):
        """
        Class constructor
        :param n_control: number of control qubits
        :param f: callable black box function
        """
        self.f = f
        # Not sure what to do about base matrix yet
        super(fGate, self).__init__(n_control + 1)

    def apply(self, control, target):
        """
        Returns two new quantum registers one with the altered qubits and one
        without?
        :param control:
        :param target:
        :return:
        """

        if control.n_qubits != self.n_qubits - 1:
            raise ValueError('Number of qubits for control qubit do not match')

        control_qubits = control.qubits
        target_qubits = target.qubits
        result = QuantumRegister(target.n_qubits, isempty=True)
        result_qubits = np.zeros(2 ** result.n_qubits, dtype=complex)

        # Initialise not_gate
        n_gate = Not()

        for i in range(control.n_states):
            if self.f(i) == 1 and control_qubits[i] != 0:
                result = result + n_gate * target
            elif self.f(i) == 0 and control_qubits[i] != 0:
                result = result + target

        # normalise at the end
        result.normalise()

        return result



def build_c_c_not(num_control_i=0, num_target_i=0):
    """
    Builds a toffoli gate, given the number of I operators between the second control and the target qubit from the
    first control. By default these distances are set to 1 and 2 respectively.
    :param num_control_i:
    :param num_target_i:
    :return: toffoli, toffoli gate (Operator Object)
    """

   # add statement that checks whether control2 <= target

    h_gate = Hadamard()
    I = IdentityGate()
    v_gate = PhaseShift(np.pi / 2)
    control_v = CUGate(v_gate, num_of_i=num_target_i - num_control_i)
    control_not = CUGate(Not(), num_of_i=num_control_i)
    v3 = v_gate * v_gate * v_gate
    control_v3 = CUGate(v3, num_of_i=num_target_i - num_control_i)
    c_I_v_gate = CUGate(v_gate, num_of_i=num_control_i + num_target_i + 1)

    # Build circuit
    toffoli = (I % I % h_gate) * c_I_v_gate * (control_not % I) * (I % control_v3) * \
              (control_not % I) * (I % control_v) * (I % I % h_gate)

    return toffoli
