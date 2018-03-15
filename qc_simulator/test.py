from qc import *
from functions import *
import math

def build_rev_c_c_not(num_control_i=0, num_target_i=0):
    """
    Builds a toffoli gate, given the number of I operators between the second control and the target qubit from the
    first control. By default these distances are set to 0 and 0 respectively.
    :param num_control_i:
    :param num_target_i:
    :return: toffoli, toffoli gate (Operator Object)
    """

    # Initialise basis gates
    h_gate = Hadamard()
    I = IdentityGate()
    I_target = IdentityGate(num_target_i + 1)
    I_control = IdentityGate(num_control_i + 1)
    I_total = IdentityGate(num_target_i + num_control_i + 2)

    v_gate = PhaseShift(np.pi / 2)
    c_v_short = CUGate(v_gate, num_of_i=num_target_i)
    c_v_long = CUGate(v_gate, num_of_i=num_target_i+num_control_i + 1)

    c_not = CUGate(Not(), num_of_i=num_control_i)
    v3 = v_gate * v_gate * v_gate
    c_v3 = CUGate(v3, num_of_i=num_target_i)

    # Build circuit
    
    print((h_gate % I_total).size)  
    print((c_v_short % I_control).size)

     
    toffoli = (h_gate % I_total) * c_v_long * (I_target % h_gate % IdentityGate(num_control_i) % h_gate)\
    * (I_target % c_not) * (I_target % h_gate % IdentityGate(num_control_i) % h_gate) * (c_v3 % I_control)\
    * (I_target % h_gate % IdentityGate(num_control_i) % h_gate) * (I_target % c_not)\
    * (I_target % h_gate % IdentityGate(num_control_i) % h_gate) * (c_v_short % I_control) * (h_gate % I_total)
    

    
    return toffoli

H=Hadamard()
N=Not()
c_not=
CN2=CUGate(N, num_of_i=1)
I=IdentityGate()
CCNot=build_c_c_not(num_control_i=1, num_target_i=1)
rev= build_rev_c_c_not(num_control_i=1, num_target_i=1)


reg1=QuantumRegister(1)
reg2=QuantumRegister(2)
reg=reg1*reg2
reg=(c)

print(reg)




































