import numpy as np
from qc_simulator.qc import *
from math import gcd

class Shors:
    """
    shors factoring class returns set of factors
    """
    def __init__(self, N, tries=10, accuracy=1):
        """
        N number to be factored
        tries number of tries to find factors
        when creating QRs the calculation is
        bits required to store int((N+1)*accuracy)
        so accuracy can be a float
        """
        self.accuracy = accuracy
        cs = []
        print("shor's algorithum")
        if N%2 == 0:
            cs.extend([2])
            N = int(N/2)
        print("input: ",N)
        for i in range(tries):
            c = self.classical(N)
            if c == [0]:
                break
            print(c, cs)
            cs.extend(c)
        cs = np.array(cs)
        cs = cs.flatten()
        cs = np.unique(cs)
        cs = [c for c in cs if c!=1]
        cs = set(cs)
        if cs =={}:
            print("is N a prime")
            self.out = Shors(N,tries,accuracy)
        else:
            self.out = set(cs)

    def classical(self, N):
        """
        clasical component of the algorithum
        Inputs:
                N: <int> Number to be factorised.
        Outputs:
                <list> Factors of N or [0] if N is prime.
        """
        if self.check_prime(N):
            print("classical")
            m = np.random.randint(2,N-1)
            #m = 2
            d = gcd(m,N)
            if d!=1:#
                print("Quantum computing required: ")
                return [d]
            else:
                print("Quantum computing required",m," :")
                p = self.find_period(N,m)
                return self.period_check(N,m,p)
        else:
            print("its a prime")
            return [0]

    def check_prime(self,N):
        """
        Check if N is a prime or not
        return true if not prime.
        Inputs:
                N: <int> Number being factorised.
        Outputs:
                <bool> True if N prime, false if N not prime
        """
        for i in range(2, N):
            if N%i == 0:
                return True
        return False

    def period_check(self, N, m, p):
        """
        Checks to see if the period is aceptable
        returns factor if acceptable.
        Inputs:
                N: <int> Number being factorised.
                m: <int> Randomised integer.
                p: <int> Period of fmap function
        Outputs:
                c: <list> List containing greatest common divisors of m*
        """
        if p%2 != 0:
            print("period should be even")
            return self.classical(N)
        elif (m**(p/2))%N == 1%N:
            print("incorrect m**(p/2)")
            return self.classical(N)
        elif (m**(p/2))%N == -1%N:
            print("incorrect m**(p/2)")
            return self.classical(N)
        else:
            c = [gcd(N, int(m**(p/2)-1)), gcd(N, int(m**(p/2)+1))]
            return c


    def find_period(self, N, m):
        """
        Finds period in for N the number to be factored
        and m the base of the powers, f(x) = m**x mod N
        Inputs:
                N: <int> Number being factorised
                m: <int> Randomised integer
        Outputs:
                c: <int> Periof of fmap
        """
        n_qubits = len(format(int((N+1)*self.accuracy),'b'))
        if n_qubits%2!=0:
            n_qubits = n_qubits+1
        print("Find period with ", n_qubits, " qubits")
        QR1 = QuantumRegister(n_qubits)
        QR1 = Hadamard(n_qubits)*QR1
        QR2 = QuantumRegister()
        QR = self.fmapping_lazy(QR1, QR2, N, m, n_qubits)
        QFT = self.QFT(n_qubits)
        QR = (QFT%IdentityGate(n_qubits))*QR
        states = QR.base_states

        print("Measureing...")
        c = self.mes(QR)
        print("Measured.")
        #print(c)
        return c


    def mes(self,QR):
        """
        takes and returns measurment of quantum register QR
        only reurns when the measurment QR is not 1
        Inputs:
                QR: <QuantumRegister> Input quantum register
        Outputs:
                c: <int> Measured state
        """
        c=1
        while c == 1:
            c = QR.measure()
        print(c)
        return c


    def QFT(self, n):
        """
        Quantum fourier tansform
        Inputs:
                n: <int> Number of qubits
        Outputs:
                M: <Operator> QFT circuit for n qubits.
        """

        print("QFT cicrcuit calculation")


        H = Hadamard()
        M = H%IdentityGate(n-1)
        for i in range(n-2):
            phi = 2*np.pi/np.power(2,i+2)
            M = (CUGate(PhaseShift(phi),1,i)%IdentityGate(n-2-i))*M
        M = (CUGate(PhaseShift(phi),1,n-2))*M
        for j in range(n-2):
            M = (IdentityGate(j+1)%H%IdentityGate(n-j-2))*M
            for i in range(n-3-j):
                phi = 2*np.pi/np.power(2,i+2)
                M1 = (IdentityGate(j+1)%(CUGate(PhaseShift(phi),1,i))%IdentityGate(n-j-i-3))
                M = M1*M
            phi = 2*np.pi/np.power(2,n-j)
            M1 = (IdentityGate(j+1)%CUGate(PhaseShift(phi),1,n-3-j))
            M = M1*M
        M = (IdentityGate(n-1)%H)*M
        M = SWAPGate(n)*M
        return M


    def fmapping_lazy(self, QR1, QR2, N, m, n_qubits):
        """
        Lazy way to map|x> and |o> onto |x>|f(x)>
        f(x) = m^x mod N

        Inputs:
                QR1: <QuantumRegister> Quantum register 1
                QR2: <QuantumRegister> Quantum register 2
                N:
                m:
                n_qubits: <int> Number of qubits of QR1
        Outputs:
                QR: <QuantumRegister> Quantum register representing
                    |x>|f(x)>
        """
        print("lazy mapping")

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
