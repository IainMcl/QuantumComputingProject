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

            And on this more basic implementation of a quantum register
            https://github.com/thmp/quantum/blob/master/register.py

            Scipy sparse matrix docs:
            https://docs.scipy.org/doc/scipy/reference/sparse.html

            -Does this need to be an abstract class? Don't think so, since there's only
            one type of qubit (as opposed to many different types of matrix operators )
                -Special class for fock states?
                -Special qubit class that implements general

            -Do we need something to convert integers to binary

    -Probably need a class for grover's iterate defined as child class of more
    general operator class?

    -Add method to check if state is normalised, if not, normalise

"""

import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix, identity

#this class might not be necessary
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
        """
        Quantum register class. The quantum register is saved as a complex
        numpy array. Each element of the array is the amplitude of the
        corresponding state, eg. the first element is the first state, the second
        element is the second state.
        """
        self.n_states = 2 ** n_qubits
        self.n_qubits = n_qubits
        self.qubits = np.zeros(self.n_states, dtype = complex)
        #Initialse quantum state at ground state.
        self.qubits[0] = 1.0


    def set_qubit(self, a ,b):
        """
        Set qubit of state a to value b
        """
        pass



    def measure(self):
        """
        Make a measurement. Square all the amplitudes and choose a random state.
        Outputs an integer representing the state measured (decimal system).
        """
        #Calculate probabilites
        probabilities = np.zeros(n_states)
        for i in range(1, n_states):
                probabilites[i] = norm( self.qubits[i] )**2

        #Choose a random state
        state = np.random.choice(self.n_states, probabilities)

        return state



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


    def normalise(self):
        """
        Normalise coefficients of qubits array
        """
        qubits_normalised = self.qubits/norm(self.qubits)
        self.qubits = qubits_normalised



class Operator():
    """
    Class that defines a quantum mechanical operator. The operator is
    a matrix. Only non zero elements are saved in a list of triplets. For each
    element in the list (i,j,Mij):
        -i,j: row nad column number
        -Mij: of the operator at that row and column

    How to define the application of an operator to a quantum state? Method or override
    multiplication.

    Should there be an is_sparce property passed to the opeartor, with some default
    value set?
    """

    def __init__(self, size):
        #Define number of columns and number of rows
        self.size = size
        self.sparce_matrix = coo_matrix(np.zeros( ( size,size) ) )


    def __mul__(self, rhs):
        """
        Overides multiplication operator so that the product between two operators
        (assuming they have the same size) gives the correct result.
        """
        if isinstance(rhs, QuantumRegister):
            #Apply operator to quantum register
            #check if number of states is the same
            if rhs.n_qubits != self.size:
                print('Number of states do not correspnd!')
                return 0

            #Otherwise return a new quantum register
            result = QuantumRegister(rhs.n_qubits)

            #Calculate result
            result.qubits = np.dot(self.matrix, rhs.qubits )

            #Normalise result
            result.normalise()
            return result

        if isinstance(rhs, Operator):
            #matrix multiplication between the two operators. Return another operator

            if rhs.size != self.size:
                print('Number of states does not correspond')
                return 0

            #Otherwise take dot product of matrices
            result = Operator(self.size)
            result.matrix = np.dot(self.matrix, rhs.matrix)

            return result

    def dag(self):
        """
        Returns the hermitian transpose of the operator
        """

        herm_transpose = Operator(self.size)
        herm_tranpose.matrix = self.matrix.getH()

        return herm_tranpose




class Hadamard(Operator):
    """
    Class that defines hadamard gate.
    """

    def __init__(self, size=1):
        super(Hadamard, self).__init__(size)

        #Define "base" hadamard matrix for one qubit and correponding sparse matrix
        self.base = 1/np.sqrt(2)*np.array( [ [1 , 1], [1 ,-1] ] )
        self.base_sparse = coo_matrix(self.base)
        self.matrix = self.__create_full_matrix(size)

    def __create_full_matrix(self,size):
        """
        Create sparse matrix by calculating kronecker product of base matrix with
        itself
        """
        result = self.base

        if size == 1 :
            return result
        else:
            for i in range(size-1):
                result = np.kron(result,self.base)

            return result


    # def apply(self, quant_register):
    #     """
    #     Apply hadamard gate to given quantum register
    #
    #     Vary number of inputs? If two inputs are submitted, then the first one
    #     automatically becomes a control qubit?
    #     """
    #
    #     #Initialize resulting quantum register
    #     result = QuantumRegister( quant_register.n_qubits )
    #
    #     #Calculate result
    #     result.qubits = np.dot(self.matrix, quant_register.qubits )
    #
    #     #Normalise result
    #     result.normalise()
    #
    #
    #     return result

class CHadamard():
    """
    Class that defines controlled hadamard gate. Takes as inputs number of control
    qubits and number of target qubits. And builds a sparse matrix
    """

    def __init__(self, n_control, n_target):
        self.n_control = n_control
        self.n_target = n_target
        self.n_states = 2**(n_control+n_target)#######
        self.matrix = self.__create_sparse_matrix()


    def __create_sparse_matrix(self):
        """
        Creates spasrse matrix according to how many target qubits we have.
        Matrix is constructed using the 'lil' format, which is better for
        incremental construction of sparse matrices and is then converted
        to 'csc' format, which is better for operations between matrices
        """

        #Create sparse hadamard matrix
        hadamard_matrix = lil_matrix( Hadamard(self.n_target).matrix )

        #Create full sparse identity matrix
        sparse_matrix = identity(self.n_states, format='lil')

        #"Put" dense hadamard matrix in sparse matrix
        target_states = 2**self.n_target
        sub_matrix_index = self.n_states-target_states
        sparse_matrix[sub_matrix_index: , sub_matrix_index: ] = hadamard_matrix

        #Convert to csc format
        controlled_hadamard = csc_matrix(sparse_matrix)

        return controlled_hadamard

    def apply(self, quant_register):
        """
        Applies controlled-Hadamard gate to given quantum register and returns
        the result
        """

        #Initialise result qunatum register
        result = QuantumRegister(quant_register.n_qubits)

        #Calculate result
        result.qubits = np.dot(self.matrix, quant_register.qubits)

        #Normalise result
        result.normalise()

        return result




########testing stuff

#Create 2-qubit hadamard gate
H_2 = Hadamard(1)
print(H_2.matrix.shape)

#Create a register with 2 qubits at the ground state
ground_2 = QuantumRegister(1)
print(ground_2.qubits)


#matrix multiplication between
test = H_2.apply(ground_2)
print(test.qubits)

test2 = H_2*ground_2
print(test2.qubits)


c_H = CHadamard(1,3)
print(c_H.matrix.toarray())

print(np.zeros( (5,5) ))
