import numpy as np
from qc_simulator.qc import *
from fractions import gcd

class shors:
    """
    only 1 regester 1 qbit number as
    all based on thing on git
    """
    def __init__(self, N):
        print("shor")
        out = self.classical
        return out

    def classical(self, N):
        m = np.random.randint(N-1)
        if gcd(m,N)!=1:
            return m
        else:
            p = self.find_period(N,m)
            if p%2 != 0:
                return self.classical(N)
            elif (m**p)%N ==0:
                return self.classical(N)
            else:
                return gcd(m**(p/2)-1, N)

    def find_period(self, N, m):
        n_qubits = len(format(N+1),'b')
        QR1 = QuantumRegister(n_qubits)
        QR1 = Hadamard(n_qubits)*QR1
        QR2 = QuantumRegister()
        QR = fmapping_lazy(QR1,N, m)
        QFT = self.QFT(2**n_qubits)
        QR = QFT*QR
        c = np.array([QR.measure() for x in range(100)])
        return p

    def QFT_cuircuts(self, N_states=1, inverse=False):
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

    def fmapping_lazy(self, QR1, N, m, QR2=QuantumRegister()):
        """
        x mod N
        """
        n_qubits = QR1.n_qubits
        n_states = 2**n_qubits
        QR2 = QuantumRegister(n_qubits)
        states = np.zeros(n_states)
        for i in range(N_states):
            x = np.mod(m**i, N)
            states[x] = states[x] +1
        QR2.base_states = states
        QR = QR1*QR2
        return QR
# not being used from here downwards

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

    def Addition(self, n, m, bar_right = True):
        if n == m:
            if bar_right == true:
                M = CUgate(PhaseShift(phi),1,1,n-1)%IdentityGate(n-1)
                for i in range(n-1):
                    for j in range(n-i-1):
                        phi = 2*np.pi/np.power(2, j)
                        M = (IdentityGate(j+i+1)%CUgate(PhaseShift(phi),1,1,n-2-j)%IdentityGate(n-(i+1)))*M
                M = (IdentityGate(n-1)%CUgate(PhaseShift(phi),1,1,n-1))*M
                return M
            else:
                M = IdentityGate(n-1)%CUgate(PhaseShift(phi),1,1,n-1)
                for i in range(n-1):
                    for j in range(n-i-1):
                        phi = 2*np.pi/np.power(2, j)
                        M = (IdentityGate(n-(i+1))%CUgate(PhaseShift(phi),1,1,n-2+j)%IdentityGate(j+i+1))*M
                M = (CUgate(PhaseShift(phi),1,1,n-1)%IdentityGate(n-1))*M
                return M
        else:
            print("quantum regesters should be equal in size")
        
        

    def AdderGate(self):
        print("addder")


    def run1(self, QR1, QR2, N_states, n_qubits):
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

