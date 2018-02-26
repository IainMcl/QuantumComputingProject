from qc import *
from functions import *

def grovers():
nqubits=10

register=QuantumRegister(nqubits)
H=Hadamard(nqubits-1)
