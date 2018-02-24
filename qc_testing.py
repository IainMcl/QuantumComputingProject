"""
07/02/2018
Andreas Malekos - School of Physics and Astronomy, The University of Edinburgh

Quantum Computing Project
Definition of abstract classes for qubits and quantum registers

Some notes from Andreas:
-This file need to contain:
        -Operations between qubits
            -Bitwise operations between quantum registers?
            -Visualization?

            Loosely based on the QObj object found in QuTip (Quantum Toolbox in Python)
            http://qutip.org/tutorials.html

            And on this more basic implementation of a quantum register
            https://github.com/thmp/quantum/blob/master/register.py

            Scipy sparse matrix docs:
            https://docs.scipy.org/doc/scipy/reference/sparse.html


            -Do we need something to convert integers to binary

    -Probably need a class for grover's iterate defined as child class of more
    general operator class?
##################################################
    -Add method to check if state is normalised, if not, normalise
    -Add special class for any type of control gate, that accepts a gate, Number
     of control qubits and number of taget qubits as inputs
    -Add get methods for matrices of operators and qubits
    -Add class for unitary operator (H*R*H*R)
    -

##################################################
The way the implementation works right now is that the gates are implemented
as Operator class objects. Quantum circuits can be defined as functions that
"string" different gates together
"""

import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix, identity,kron
from math import pi

class QuantumRegister():
    """
    Quantum register class. The quantum register is saved as a complex
    numpy array. Each element of the array is the amplitude of the
    corresponding state, eg. the first element is the first state, the second
    element is the second state.
    """

    def __init__(self, n_qubits=1):
        self.n_states = int(2 ** n_qubits)
        self.n_qubits = n_qubits
        self.qubits = np.zeros(self.n_states, dtype = complex)
        #Initialse quantum state at ground state.
        self.qubits[0] = 1.0


    def set_qubit(self, a ,b):
        """
        Set qubit of state a to value b. Not sure how to do this right now.
        Maybe instead of having a method, an extra argument can be passed to
        the class constructor telling it in which states we wish our state
        function to be...
        """
        pass



    def measure(self):
        """
        Make a measurement. Square all the amplitudes and choose a random state.
        Outputs an integer representing the state measured (decimal system).
        """
        #Calculate probabilites
        probabilities = np.zeros(self.n_states)
        for i in range(self.n_states):
                probabilities[i] = norm( self.qubits[i] )**2

        #Choose a random state
        n = int(self.n_states)
        state = np.random.choice(n, p=probabilities)

        return state

    def __mul__(self, other):
        """
        Tensor prodcut between the two quantum registers. Outputs a new quantum
        register.
        """
        #Result is tensor product of the qubits in each state
        temp_result = np.kron(self.qubits, other.qubits)

        #Result has to be normalized
        result_normalized = temp_result/norm(temp_result)

        #Creaete quantum register object for result
        qmr_result = QuantumRegister(self.n_qubits+other.n_qubits )
        qmr_result.qubits = result_normalized

        return qmr_result


    def normalise(self):
        """
        Normalise coefficients of qubits array
        """
       
        qubits_normalised = self.qubits/norm(self.qubits)
        self.qubits = qubits_normalised

    def select(self,a,b):
        """
        Return quantum register with comprising of the ath to bth qubits
        """

class Operator():
    """
    Class that defines a quantum mechanical operator. The operator is
    a matrix. Only non zero elements are saved in a list of triplets. For each
    element in the list (i,j,Mij):
        -i,j: row nad column number
        -Mij: of the operator at that row and column


    For now the operator is saved as a dense matrix. Special cases like control
    gates will be saved as sparse matrices.
    """

    def __init__(self, n_qubits, base=np.zeros( (2,2) ) ):
        """
        Class initialiser. The class accepts two inputs:
            -n_qubits: Number of qubits on which this operator will operate.
            -base: The "base" 2*2 matrix of the operator.

        [Note that fow now we assume that every operator has a unitary "base"
        and that there is no need "native" binary operators (such as SWAP gate)]
        """
        self.n_qubits = n_qubits
        self.size = 2**n_qubits
        self.matrix = self.__create_full_matrix(self.n_qubits, base)
        #self.sparce_matrix = coo_matrix(np.zeros( ( self.size, self.size) ) )  not sure that we need thsi right now


    def __create_full_matrix(self,n_qubits, base):
        """
        Create matrix by taking successive tensor producs between for the total
        number of qubits.
        """
        result = lil_matrix(base)

        if n_qubits == 1 :
            result = csc_matrix(result)

            return result
        else:
            for i in range(n_qubits-1):
                result = kron(result, base)

            result = csc_matrix(result)
            return result


    def __mul__(self, rhs):
        """
        Overides multiplication operator so that the product between two operators
        (assuming they have the same size) gives the correct result.
        """
        if isinstance(rhs, QuantumRegister):
            #Apply operator to quantum register
            #check if number of states is the same
            if rhs.n_qubits != self.n_qubits:
                raise ValueError('Number of states do not correspnd: rhs.n_qubits = {}, lhs.n_qubits = {}'.format(rhs.n_qubits,self.n_qubits))

            #Otherwise return a new quantum register
            result = QuantumRegister(rhs.n_qubits)

            #Calculate result. Check if matrix is sparse or not first. If sparse
            #use special sparse dot product csc_matrix.dot
            if isinstance(self.matrix, np.ndarray):
                result.qubits = np.dot(self.matrix, rhs.qubits )

            elif isinstance(self.matrix, csc_matrix):
                result.qubits = self.matrix.dot(rhs.qubits)

            #Normalise result
            result.normalise()
            return result

        if isinstance(rhs, Operator):
            """
            Matrix multiplication between the two operators
            """
            if rhs.size != self.size:
                raise ValueError('Operators must of of the same size: rhs.size = {} lhs.size = {} '.format(rhs.size,self.size))

            #Otherwise take dot product of
            result = Operator(self.n_qubits)
            result.matrix = self.matrix.dot(rhs.matrix)

            return result


    def __mod__(self, other):
        """
        Override mod operator to defint tensor product between operators
        """
        #Tensor product between the two operators
        result = Operator(self.n_qubits, other.n_qubits)
        result.matrix = kron(self.matrix, other.matrix)
        return result


    def dag(self):
        """
        Returns the hermitian transpose of the operator
        """

        herm_transpose = Operator(self.n_qubits)
        herm_transpose.matrix = self.matrix.getH()

        return herm_transpose

    def __getitem__(self, key):
        """
        Override of the [] operator to return a "subregister" of the current quantum reigster. The method checks first
        to see whether the subregister desired contains entangled qubits or not, and raises an error if it does. The
        condition to check whether a subregister is entangled or not is that non zero elelents of the corresponding
        array must all be either in odd or even indexes.
        :param slice: qubit number
        :return: subregister as defined by slice
        """

        #Length of new quantum register:
        if key.start == None:
            l = 1
        else :
            l = key.stop - key.start

        # Extract the qubits
        qubits = self.qubits


        # Check to see if the sliced qubits are enatngled or not.


        pass


class Hadamard(Operator):
    """
    Class that defines hadamard gate. This class extends the Operator class. For
    now it simply calls the parent classes and passes to it the base argument.
    """
    def __init__(self, n_qubits=1):
        #Define "base" hadamard matrix for one qubit and correponding sparse matrix
        self.base = 1/np.sqrt(2)*np.array( [ [1 , 1], [1 ,-1] ] )
        super(Hadamard, self).__init__(n_qubits,self.base)

class PhaseShift(Operator):
    """
    Implementation of phase shift gate.
    """
    def __init__(self, phi, n_qubits=1):
        self.base = np.array( [ [ 1, 0], [ 0, np.exp( 1j*phi ) ] ])
        super(PhaseShift, self).__init__(n_qubits, self.base)

class Not(Operator):
    """Implements NOT gate
    """
    def __init__(self, n_qubits=1):
        self.base = np.array( [ [0,1], [1,0] ])
        super(Not, self).__init__(n_qubits, self.base)

class Unitary(Operator):
    """
    unitary gate
    """
    def __init__(self, theta, phi, n_qubits = 1):
        self.base = np.array([[np.cos(theta), np.sin(theta)*-1j*np.exp( -1j*phi )], [np.sin(theta)*-1j*np.exp( -1j*phi ), np.cos(theta)]])
        super(Unitary, self).__init__(n_qubits, self.base)

class CUGate(Operator):
    """
    Class that implements a controlled U gate
    """

    def __init__(self, base, n_control, n_target=1):
        """
        Class accepts the base matrix U, number of control qubits and number of
        target qubits.
        Inputs:
        base: matrix/operator
        n_control: number of control qubits
        n_target: number of target qubits (has been set to 1 as default)
        """
        self.n_control = n_control
        self.n_target = n_target
        self.n_qubits = self.n_target + self.n_control
        self.size = 2**(self.n_control + self.n_target)
        self.matrix = self.__create_sparse_matrix(base)

    def __create_sparse_matrix(self,base):
        """
        Creates spasrse matrix according to how many target qubits we have.
        Matrix is constructed using the 'lil' format, which is better for
        incremental construction of sparse matrices and is then converted
        to 'csc' format, which is better for operations between matrices
        """

        #Create sparse hadamard matrix
        base_matrix = lil_matrix(base.matrix)

        #Create full sparse identity matrix
        sparse_matrix = identity(self.size, format='lil')

        #"Put" dense hadamard matrix in sparse matrix
        target_states = 2**self.n_target
        sub_matrix_index = self.size-target_states
        sparse_matrix[sub_matrix_index: , sub_matrix_index: ] = base_matrix

        #Convert to csc format
        c_gate = csc_matrix(sparse_matrix)

        return c_gate

class CHadamard(Operator):
    """
    Class that defines controlled hadamard gate. Takes as inputs number of control
    qubits and number of target qubits. And builds a sparse matrix.
    ############################
    This class may not be necessary anymore
    ###########################
    """

    def __init__(self, n_control, n_target):

        self.n_control = n_control
        self.n_target = n_target
        self.n_qubits= self.n_target + self.n_control
        self.size = 2**(n_control+n_target)
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
        sparse_matrix = identity(self.size, format='lil')

        #"Put" dense hadamard matrix in sparse matrix
        target_states = 2**self.n_target
        sub_matrix_index = self.size-target_states
        sparse_matrix[sub_matrix_index: , sub_matrix_index: ] = hadamard_matrix

        #Convert to csc format
        controlled_hadamard = csc_matrix(sparse_matrix)

        return controlled_hadamard
    
class ControlV(Operator):
    def __init__(self,n_qubits=1):
        self.base = np.array([[1,0],[0,1j]])
        super().__init__(n_qubits-1,self.base)
        
class ControlNot(Operator):
    def __init__(self):
        self = Hadamard()*ControlV()*ControlV()*Hadamard()
        
class Toffoli(Operator):
    def __init__(self):
        pass
    
        
class Oracle(Operator):
    """
    Class that implements the oracle. This gate takes an n qubits as
    input and flips it if f(x) = 1, otherwise it leaves it as the same.
    """

    def __init__(self, n_qubits=1, x=0):
        """
        Class constructor.
        Inputs:
        n_states: Total number of n_states
        x: state that is fliped.
        """
        self.n_states = 2**n_qubits
        self.n_qubits = n_qubits
        #Operator matrix will be identity with a -1 if the xth,xth element
        self.matrix = identity(self.n_states, format='csc')
        self.matrix[x,x] = -1


########testing stuff##############
if __name__ == '__main__':
    #Create 2 qubit hadamard gate
    H2 = Hadamard(2)

    #Create 2 qubit quantum register in ground state
    target2 = QuantumRegister(2)

    #Apply hadamard gate to target state
    H2 = Hadamard(2)
    result_1 = H2*target2
    #Print result
    print(result_1.qubits)

    #Define control qubit and apply hadamard to it
    control1 = QuantumRegister(1)
    control1_superposition = Hadamard(1)*control1

    #Print result
    print(control1_superposition.qubits)

    #Define controlled hadamard gate with 1 control and 2 targets
    c_H = CHadamard(1,2)
    h1 = Hadamard()
    print(h1.matrix)
    c_u = CUGate(h1,2)

    print(c_u.matrix.toarray())

    #Create new quantum register with control and target qubits
    control_target = control1_superposition*target2

    #Print new quantum register
    print(control_target.qubits)

    #Apply controlled hadamard gate to this quantum register
    print(control_target.n_states )
    result = c_H*control_target

    #Print result
    print(result.qubits)

    I = Operator(2, np.eye(2))
    print(I.matrix.toarray())

    #Testing oracle operator
    qubit1 = Hadamard(3) * QuantumRegister(3)
    oracle = Oracle(3,2)
    
    #wooo it works
    print((oracle*qubit1).qubits)
    
    #check unitarty
    print(Unitary(pi,0).base)
