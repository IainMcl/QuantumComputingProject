import numpy as np
from qc_simulator.qc import *

class shors:
    """
    only 1 regester 1 qbit number as
    regester 2 based on first
    all based on thing on git
    """
    def __init__(self, n_qbits, m_qbits):
        print("shor")
        QR1 = QuantumRegister(n_qbits)
        QR2 = QuantumRegister(m_qbits)
        out = self.shors(QR1, QR2)
        return out
        
        

    def QFT(self, QR):
        """
        maybe to be edited, if statment changed maybe
        """
        n = QR.n_qubits
        H = Hardamard()
        I = IdentityGate()
        P = PhaseShift()
        M = I
        for j in range(n):
            if j == 0:
                M1 = IdentityGate(n-1)%H
            elif j == n-1:
                M1 = H%IdentityGate(n-1)
            else:
                M1 = IdentityGate(j)%H%IdentityGate(n-1-j)
            M2 = H
            for i in range(j):
                M2 = M2%I
            for i in range(n-j-1):
                M2 = P%M2
            M = M2*M1*M
        return M

    def SumGate(self, n=3):
        """
        unlikely that it can be applied to n =\=3
        but will keep just in case for now, only used as n = 3
        """
        I = IdentityGate()
        M = I%CUGate(not())
        M = CUGate(Not(),1,1,1)*M
        return M
    
    def CarryGate(self, n = 4):
        """
        like sum, likely only usable with 4
        """
        I = IdentityGate()
        M = I%build_c_c_not()
        M = (I%CUGate(Not())%I)*M
        M = build_c_c_not(1,0)*M #CQubit I CQubit TQubit
        return M

    def AdderGate(self):
        print("addder")


    def Shors(self, QR1, QR2):
        """
        aplication
        """
        print("shor")
