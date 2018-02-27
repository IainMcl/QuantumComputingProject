from qc import *
from functions import *
import numpy as np

def grovers():
    
    # Tag
    k=3

    # Number of qubits
    nqubits=5

    # Initiate the gates
    n_gate_1=Not(1)
    h_gate_n=Hadamard(nqubits)
    h_gate_1=Hadamard(1)
    i_gate_1=Operator(1,base=np.eye(2))

    # Initiate the registers
    reg1=QuantumRegister(nqubits)
    reg2=QuantumRegister(1)
    reg2=n_gate_1*reg2
    reg2=h_gate_1*reg2
    reg1=h_gate_n*reg1
    #print(reg1.qubits)

    # Define fk and f0
    fk=Oracle(nqubits, x=k)
    f0=Oracle(nqubits, x=0)
    
    m=20
    # Grovers algorithm
    for i in (range(m)):
        #print (i)
        reg1=h_gate_n*f0*h_gate_n*fk*reg1

    #print(reg1.qubits)
    print(reg1.measure())

    '''
    reg1=fk*reg1
    reg1=h_gate_n*reg1
    reg1=f0*reg1
    reg1=h_gate_n*reg1
    '''


def main():
    grovers()
    
    
class LewisOracle():
    def __init__(self, x):
        self.x = x
    def __getitem__(self, key):
        if key == x:
            return 1
        else:
            return 0

main()
