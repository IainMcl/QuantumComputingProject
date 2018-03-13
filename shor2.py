import numpy as np
from qc_simulator.qc import *

class shors:
    """
    only 1 regester 1 qbit number as
    all based on thing on git
    """
    def __init__(self, n_qbits):
        print("shor")
        QR1 = QuantumRegister(n_qbits)
        QR2 = QuantumRegister(1)
        N_states = QR1.n_states
        n_qubits = QR1.n_qubits
        out = self.run(QR1, QR2, N_states, n_qubits)
        return out
        
        

    def QFT(self, N_states=1):
        """
        maybe to be edited, if statment changed maybe
        """
        H = Hardamard()
        I = IdentityGate()
        P = PhaseShift()
        M = I
        for j in range(N_states):
            if j == 0:
                M1 = IdentityGate(N_states-1)%H
            elif j == n-1:
                M1 = H%IdentityGate(N_states-1)
            else:
                M1 = IdentityGate(j)%H%IdentityGate(N_states-1-j)
            M2 = H
            for i in range(j):
                M2 = M2%I
            for i in range(N_states-j-1):
                M2 = P%M2
            M = M2*M1*M
        return M

    def Sum(self, n=3):
        """
        unlikely that it can be applied to n =\=3
        but will keep just in case for now, only used as n = 3
        """
        I = IdentityGate()
        M = I%CUGate(not())
        M = CUGate(Not(),1,1,1)*M
        return M
    
    def Carry(self, n = 4):
        """
        like sum, likely only usable with 4
        """
        I = IdentityGate()
        M = I%CUGate(Not(),2,1)
        M = (I%CUGate(Not())%I)*M
        M = M1*M #CQubit I CQubit TQubit

    def f(self, QR1, QR2, N_states, n_qubits):
        """
        x mod N
        """
        QR2 = QuantumRegister(n_qubits)
        for i in range(N_states):
            QR2[i] = np.mod(i, n_qubits)
        QR = QR1*QR2
        return QR


    def run(self, QR1, QR2, N_states, n_qubits):
        """
        aplication
        """
        QR1 = Hadamard(n_qubits)*QR1
        QR = self.f(QR1, QR2, N_states, n_qubits)
        QTF = self.QTF(N_states)
        M = QTF % np.identity(N_states)
        QR = M*QR
        result = measure(QR)
        print(result)
        return result

a = shors(2)

