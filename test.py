from qc_simulator.qc import *
from qc_simulator.functions import *
import numpy as np
import math
#from grover5_andy import grover5

def build_n_not(n):
    """
    Builds a c_not gate with n control bits
    """
    # Iniate the three base gates
    c_c_not=build_c_c_not()
    c_not=CUGate(Not())
    I = Operator(base=np.eye(2,2))



    # Two cases, n is even and n odd
    # Num is will be the number of gates necessary
    # If odd
    if (n&1)==1:
        num=n-2
    if (n&1)==0:
        num=n-1

    # Initiate the gates list
    gates=np.empty(num, dtype=Operator)


    # Construct n_not gate in for Loop
    for i in range(num-1):
        for j in range(i):
            









def main():
    a=11&1
    b=10&1
    print(a)
    print(b)



main()
