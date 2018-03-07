import numpy as np
from qc_simulator.qc import *

class shors:
    """
    """

    def QFT(self, QR, n):
        """
        maybe to be edited, if statment changed maybe
        """
        H = Hardamard()
        I = IdentityGate()
        P = PhaseShift()
        for j in range(n):
            if j == 0:
                M1 = IdentityGate(n-1)%H
            elif j == n-1:
                M1 = H%IdentityGate(n-1)
            else:
                M1 = IdentityGate(j)%H%IdentityGate(n-1-j)
                M2 = I
            for i in range(n-j):
                M2 = M2%I
            for i in range(j):
                M2 = P%M2
            if j==0: #check about identity
                M = M2*M1
            else:
                M = M2*M1*M
print("hel")
