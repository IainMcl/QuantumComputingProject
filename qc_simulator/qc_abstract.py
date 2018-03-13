"""
Abstract classes for quantum register and quantum operator
"""

from abc import ABC, abstractclassmethod

class QuantumRegisterAbstract(ABC):


    @abstractclassmethod
    def measure(self):
        """
        Measure that defined measurement of quantum register
        """
        pass

    @abstractclassmethod
    def __mul__(self, other):
        """
        Define multiplication between quantum registers
        """
        pass

    @abstractclassmethod
    def __str__(self):
        """
        Define print method for the quantum register
        """

class OperatorAbstract(ABC):

    @abstractclassmethod
    def __mul__(self, rhs):
        """
        Define multiplication between operators, and optionally, between operators and
        quantum registers.
        """
        pass

    @abstractclassmethod
    def __mod__(self, other):
        """
        Define tensor product between operators
        """
        pass

