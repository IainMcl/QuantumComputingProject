# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:29:09 2018

@author: Lewis
"""
import unittest
import numpy as np
import math
from qc_testing import *
from functions import *

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
        O = Oracle(10,x=5)
        reg = QuantumRegister(10)
        result = O*reg
        self.assertEqual(result.qubits[5],-1*reg.qubits[5])
        for i in range(reg.qubits.size):
            if i != 5:
                self.assertEqual(result.qubits[i], reg.qubits[i])
                
                
    def test_phase_shift(self):
        P = PhaseShift(math.pi, n_qubits = 1)
        H = Hadamard(1)
        reg = H*QuantumRegister(1)
        regQuBits = reg.qubits
        result = P*reg
        
        gate = np.array([[1,0],[0,-1]])
        expected = np.dot(gate,regQuBits)  
        np.testing.assert_almost_equal(expected.tolist(),result.qubits.tolist())
        
    def test_deutsch(self):
        o1 = Operator(1, np.array([[1,0],[0,1]]))
        self.assertEqual(deutsch(o1),0)
        o2 = Operator(1,np.array([[1,0],[0,-1]]))
        self.assertEqual(deutsch(o2),1)
       
    def test_grover(self):
        o = Oracle(10,x=5)
        result = grover_search(o)
        self.assertEqual(5,result)
     
    def test_adder(self):
        a = QuantumRegister(1)
        b = QuantumRegister(1)
        result = quantumAdder(a,b)
        print(result.qubits)
        self.assertEqual(np.array([1,0]),result.qubits)
        
    def testControlV(self):
        reg = QuantumRegister(1)
        reg.qubits = np.array([1,1])
        result = ControlV(1)*reg
#        expected = reg
        expected.qubits[-1] *= 1j
        np.testing.assert_almost_equal(expected.qubits.tolist(),result.qubits.tolist())
        
    def test_operator_tensor(self):
        H = Hadamard()
        result = H%H
        expected = 0.5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
        np.testing.assert_almost_equal(result.matrix.toarray().tolist(),expected.tolist())
    
    def test_settingQubits(self):
        a = GetQuBitofValue(np.pi/4,np.pi/4)
        expected = (1/np.sqrt(2))*np.array([1,1])
        result = np.absolute(a.qubits)
        np.testing.assert_almost_equal(expected,result)
        
        
        
        
unittest.main()