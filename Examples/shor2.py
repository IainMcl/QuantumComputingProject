import numpy as np
from qc_simulator.qc import *
#from qc_simulator.functions import *
from math import gcd

class Shors():
    """
    only 1 regester 1 qbit number as
    all based on thing on git
    """
    def __init__(self, N):
        self.cs = []
        print("shor's algorithum")
        if N%2 == 0:
            print("odd number please")
            self.out = 2
        else:
            print("input: ",N)
            cs = []
            for i in range(8):
                c = self.classical(N)
                print(c, cs)
                cs.extend(c)
            cs = np.array(cs)
            cs = cs.flatten()
            cs = np.unique(cs)
            cs = [c for c in cs if c!=1]
            self.out = set(cs)

    def classical(self, N):
        print("classical")
        m = np.random.randint(2,N-1)
        #m = 2
        d = gcd(m,N)
        if d!=1:#  
            print("easy: ")
            return [d]
        else:
            print("though",m," :")
            p = self.find_period(N,m)
            print("period(p): ",p)
            print("m**p: ",m**p)
            print("m**(p/2): ",m**(p/2))
            return self.period_check(N,m,p)

    def period_check(self, N, m, p):
        if p%2 != 0:
            print("oops-------1")
            return self.classical(N)
        elif (m**(p/2))%N == 1%N:
            print("oops-------2")
            return self.classical(N)
        elif (m**(p/2))%N == -1%N:
            print("oops------3")
            return self.classical(N)
        else:
            c = [gcd(N, int(m**(p/2)-1)), gcd(N, int(m**(p/2)+1))]
            return c


    def find_period(self, N, m):
        n_qubits = len(format((N+1),'b'))
        if n_qubits%2!=0:
            n_qubits = n_qubits+1
        print("find period with ", n_qubits, " qubits")
        QR1 = QuantumRegister(n_qubits)
        QR1 = Hadamard(n_qubits)*QR1
        QR2 = QuantumRegister()
        QR = self.fmapping_lazy(QR1, QR2, N, m, n_qubits)
        QFT = self.QFT(n_qubits)
        QR = (QFT%IdentityGate(n_qubits))*QR
        states = QR.base_states
        print(states)
        print("measureing")
        #c = self.get_p(QR)
        #print(c)
        #c = gcd(QR.measure(), QR.measure())
        #c = gcd(self.mes(QR), self.mes(QR))
        c = self.mes(QR)
        print("measured")
        #print(c)
        return c

    def get_p(self,QR):
        c = self.mes(QR)
        for i in range(2):
            c = gcd(c, self.mes(QR))
        return c

    def mes(self,QR):
        c=1
        while c == 1:
            c = QR.measure()
        print(c)
        return c


    def QFT(self, n):
        print("QFT")
        H = Hadamard()
        M = H%IdentityGate(n-1)
        for i in range(n-2):
            phi = 2*np.pi/np.power(2,i+2)
            M = (CUGate(PhaseShift(phi),1,i)%IdentityGate(n-2-i))*M
        M = (CUGate(PhaseShift(phi),1,n-2))*M
        for j in range(n-2):
            M = (IdentityGate(j+1)%H%IdentityGate(n-j-2))*M
            for i in range(n-3-j):
                #print(i)
                phi = 2*np.pi/np.power(2,i+2)
                #print(M.matrix.shape)
                M1 = (IdentityGate(j+1)%(CUGate(PhaseShift(phi),1,i))%IdentityGate(n-j-i-3))
                #print(j+1,i,n-j-i-3)
                #print(M1.matrix.shape)
                M = M1*M
            #print("pass")
            phi = 2*np.pi/np.power(2,n-j)
            M1 = (IdentityGate(j+1)%CUGate(PhaseShift(phi),1,n-3-j))
            M = M1*M
        M = (IdentityGate(n-1)%H)*M
        M = SWAPGate(n)*M
        return M


    def fmapping_lazy(self, QR1, QR2, N, m, n_qubits):
        """
        x mod N
        """
        print("lazy mapping")
        #n_qubits = QR1.n_qubits
        n_states = 2**n_qubits
        QR2 = QuantumRegister(n_qubits)
        states = np.zeros(n_states)
        for i in range(n_states):
            x = int(np.mod(m**i, N))
            states[x] = states[x] +1
        QR2.normalise()
        QR2.base_states = states
        QR = QR1*QR2
        return QR


# not being used from here downwards-------------------------

    def swap_gate(self, n=2):
        print("swap")
        if n < 2:
            return IdentityGate(n)
        else:
            M1 = CUGate(Not())
            print(M1.matrix.shape)
            M1 = self.flip_not_gate(M1)*M1
            M1 = CUGate(Not())*M1
            M = M1
            for i in range(int((n-1)/2)):
                M = M1%M
            print(M.n_qubits)
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

    def QFT_circuts(self, n_states=1, inverse=False):
        """
        maybe to be edited, if statment changed maybe
        """
        H = Hadamard()
        I = IdentityGate()
        for j in range(n_states-1):
            if j == 0:
                M = H%IdentityGate(N_states-1)
                for i in range(N_states-1):
                    phi = 2*np.pi/np.power(2,i)
                    M1 = CUGate(PhaseShift(phi),1, i)%IdentityGate()
            elif j == n_states:
                M = (IdentityGate(N_states-1)%H)*M
            else:
                M = H%IdentityGate(N_states-1-j)*M
                for i in range(N_states-j-1):
                    phi = 2*np.pi/np.power(2,i)
                    M = (IdentityGate(j)%CUGate(PhaseShift(phi),1, i)%IdentityGate())
        M = M2*M1*M
        return M

a = Shors(35)
a = a.out
print("we get: ",a)
