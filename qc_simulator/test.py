from qc import *
from functions import *
import math
import numpy as np

def build_rev_c_c_not(num_control_i=0, num_target_i=0):
    """
    Builds a reverse toffoli gate, given the number of I operators between the second control and the target qubit from the
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
    
    if num_control_i == 0:
        gate = (h_gate % I_total) * c_v_long  * (I_target % h_gate % h_gate)\
        * (I_target % c_not) * (I_target % h_gate % h_gate) * (c_v3 % I_control)\
        * (I_target % h_gate %  h_gate) * (I_target % c_not)\
        * (I_target % h_gate %  h_gate) * (c_v_short % I_control) * (h_gate % I_total)
    else:
    
        gate = (h_gate % I_total) * c_v_long  * (I_target % h_gate % IdentityGate(num_control_i) % h_gate)\
        * (I_target % c_not) * (I_target % h_gate % IdentityGate(num_control_i) % h_gate) * (c_v3 % I_control)\
        * (I_target % h_gate % IdentityGate(num_control_i) % h_gate) * (I_target % c_not)\
        * (I_target % h_gate % IdentityGate(num_control_i) % h_gate) * (c_v_short % I_control) * (h_gate % I_total)
    

    
    return gate

def build_rev_c_not(num_of_i=0):
    '''
    Builds a reverse c not gate
    num_of_i is the number of qubits between the control and target
    '''
    
    
    h_gate = Hadamard()
    c_not = CUGate(Not(), num_of_i=num_of_i)
    
    if num_of_i == 0:
        gate = (h_gate % h_gate) * c_not * (h_gate % h_gate)
    
    else:
        I = IdentityGate(n_qubits=num_of_i)
        gate = (h_gate % I % h_gate) * c_not * (h_gate % I % h_gate)
        
    return gate
    



    
    



def build_3qubit_encode_gate():
    I=IdentityGate()
    c_not=CUGate(Not())
    gate=(I % c_not) * (c_not % I)
    
    return gate

def build_3qubit_ancilla_gate():
    I=IdentityGate()
    c_not_1i=CUGate(Not(), num_of_i=1)
    c_not_2i=CUGate(Not(), num_of_i=2)
    c_not_3i=CUGate(Not(), num_of_i=3)
    
    gate=(c_not_2i % I)*(I % c_not_1i % I)*(c_not_3i)*(I % I % c_not_1i)
    
    return gate

def build_3qubit_correction_gate():
    rev_c_c_not = build_rev_c_c_not()
    rev_c_c_not_1i = build_rev_c_c_not(num_target_i = 1) 
    rev_c_c_not_2i = build_rev_c_c_not(num_target_i = 2) 
    rev_c_not_1i = build_rev_c_not(num_of_i = 1)
    I = IdentityGate()
    
    gate = rev_c_c_not_2i * (I % rev_c_not_1i % I) * (I % rev_c_c_not_1i)\
    * (I % I % rev_c_not_1i) * (I % I % rev_c_c_not)
    
    return gate

def build_9qubit_encode_gate():
    c_not_2i=CUGate(Not(), num_of_i=2)
    c_not_5i=CUGate(Not(), num_of_i=5)
    h_gate = Hadamard()
    I=IdentityGate()
    c_not=CUGate(Not())
    c_not_1i=CUGate(Not(), num_of_i=1)
    
    gate = (c_not_1i % c_not_1i % c_not_1i) * (c_not % I % c_not % I % c_not %I)\
    * (h_gate % I % I % h_gate % I % I % h_gate % I % I) * ( c_not_5i % I % I)\
    * (c_not_2i % I % I % I % I % I)
    
    return gate

#def build_9qubit_


gate = build_9qubit_encode_gate()
reg1 = Not()* QuantumRegister(1)
reg2 = QuantumRegister(8)
reg=reg1*reg2
reg= gate*reg
I=IdentityGate()
'''
H=Hadamard()
N=Not()
c_not=CUGate(Not())
CN2=CUGate(N, num_of_i=1)
I=IdentityGate()
CCNot=build_c_c_not(num_control_i=1, num_target_i=1)
rev= build_rev_c_c_not(num_control_i=1, num_target_i=1)
hel=(I % c_not)


reg1=(N%I%I) * QuantumRegister(3)
encode = build_3qubit_encode_gate()

reg1= encode * reg1

mess=(I%I%N)

reg1=mess*reg1

reg2=QuantumRegister(2)
reg=reg1*reg2

qubitanc = build_3qubit_ancilla_gate()


reg = qubitanc * reg

correction = build_3qubit_correction_gate()

reg = correction * reg 

'''
print(reg)






































