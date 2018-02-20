# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:29:09 2018

@author: Lewis
"""
import unittest
import numpy as np
import math
from qc import QuantumRegister, Hadamard,CHadamard, Oracle
class QCTesting(unittest.TestCase):
    def test_mul_Hadamard(self):
        reg1 = QuantumRegister(1)
        H = Hadamard()
        applied = np.asmatrix((H*reg1).qubits)
        H_test = 1/math.sqrt(2)*np.matrix([[1,1],[1,-1]])
        expected = np.dot(H_test,reg1.qubits)
        np.testing.assert_almost_equal(expected.tolist(),applied.tolist())


    def test_mul_CHadamard(self):
        reg1 = QuantumRegister(2)
        CH = CHadamard(1,1)
        result = (CH*reg1).qubits
        expected= np.array((1,0,0,0))
        self.assertEqual(result.tolist(),expected.tolist())
        
        
    def test_oracle(self):
        O = Oracle(5)
        reg = QuantumRegister(10)
        result = O*reg
        self.assertEqual(result.qubits[5],-1*reg.qubits[5])
        for i in range(reg.qubits.size):
            if i != 5:
                self.assertEqual(result.qubits[i], reg.qubits[i])
        
        
unittest.main()