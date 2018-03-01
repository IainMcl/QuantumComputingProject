import numpy as np
from qc_testing import *


"""
Testing random quantum computing stuff 
"""
# control = QuantumRegister()
# target = QuantumRegister()
#
# h_gate = Hadamard()
# not_gate = Not()
# control = not_gate * control
# target = h_gate * not_gate * target
#
# c_not_gate = ControlNot()
#
# negated_target = c_not_gate.apply(control, target)
# print(negated_target.qubits)

# Testing addition of two quantum registers
# a = QuantumRegister(1)

#
# print(a.n_qubits)
#
# # set a to 1/sqrt(2) * (|0> - |1> )
# a = h_gate * not_gate * a
#
# c = a + b
#
# print(c.qubits)
#
# c.normalise()
# print(c.qubits)


# Testing the oracle function
def test_oracle(x):
    """
    Takes an integer as input and returns 1 only if x is 1. Otherwise returns 0
    :param x: integer x
    :return: 0 or 1
    """

    if x == 2 or x == 1 or x ==3 :
        return 1
    else:
        return 0

f_test_gate = fGate(2, test_oracle)
n_gate = Not()
h_gate1 = Hadamard()
target = h_gate1 * n_gate * QuantumRegister()

control = QuantumRegister(2)
h_gate = Hadamard(2)
control = h_gate * control

print(control)

f_applied = f_test_gate.apply(control, target)
print(f_applied)


"""
Testing random python stuff 
"""
# a = np.array([0+0.5j, 0.5+0.0j])
# b = np.roll(a, 2)
#
# c = np.array([1, 2, 3])
#
# d = a * c

# print(b)



