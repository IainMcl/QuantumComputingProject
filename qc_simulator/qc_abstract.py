"""
Abstract classes for quantum register and quantum operator. The compulsory
methods to implement are restricted to:
    QuantumRegisterAbstract:
        -Measurement method
        -Override of the multiplication operator so that interaction between
        quantum registers is defined
        -Print Method
    OperatorAbstract
        -Override of multiplication operator so that interaction between
        operators as well as operators and quantum registers is defined.
"""

from abc import ABC, abstractclassmethod

class AbstractQuantumRegister(ABC):


    @abstractclassmethod
    def measure(self):
        """
        Measure that defined measurement of quantum register
        """
        pass

    @abstractclassmethod
    def __mul__(self, other: 'AbstractQuantumRegister')-> 'AbstractQuantumRegister' :
        """
        Define multiplication between quantum registers
        """
        pass

    @abstractclassmethod
    def __str__(self):
        """
        Define print method for the quantum register
        """

class AbstractOperator(ABC):

    @abstractclassmethod
    def __mul__(self, rhs):
        """
        Define multiplication between operators, and optionally, between operators and
        quantum registers.
        """
        pass

    @abstractclassmethod
    def __mod__(self, other: 'AbstractOperator') -> 'AbstractOperator' :
        """
        Define tensor product between operators
        """
        pass

    @abstractclassmethod
    def __str__(self):
        """
        Define print method for operator objectself.
        """
        pass
