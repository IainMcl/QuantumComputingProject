from qc_simulator.qc import *
from qc_simulator.functions import *
import numpy as np
import math
from matplotlib import pyplot as plt

def grover(oracle, k=1, plot = False):
    '''
    Grover search.
    Inputs: Oracle -> Oracle gate that tags one state
            k -> number of states tagged
    Outputs: n -> measured tagged state.

    Still need to update in order to add a general control_n_gate method,
    instead of the "hacky way"
    '''

    # Save the oracle gate and n_qubits
    oracle_gate = oracle

    # The oracle has one more qubit than the number of qubits in the register
    n_qubits=oracle.n_qubits-1
    # Define basis gates
    not_gate = Not()
    h_gate = Hadamard()
    control_n=CUGate(not_gate, n_qubits)
    h_n_gate = Hadamard(n_qubits)
    not_n_gate = Not(n_qubits)
    I=IdentityGate()

    # Create the reflection round average gate
    W = (h_n_gate % I) * (not_n_gate % I) * (control_n) * (not_n_gate % I) * (h_n_gate % I)

    # Define the input and ancillary quantum registers
    input_register = Hadamard(n_qubits) * QuantumRegister(n_qubits)

    aux = h_gate * not_gate * QuantumRegister()
    register = input_register

    # Loop and apply grover operator iteratively
    #n_runs = round( math.pi * math.sqrt(n_qubits/k)/4)

    n_runs = round( ((math.pi/4)/math.sqrt(k))*2**(n_qubits/2))

    # Add auxilary qubit to register
    register = register * aux

    for i in range(n_runs):

        #register.plot_register()
        # Apply grover iteration
        register=oracle * register
        register = W * register
        if plot==True:
            ax = register.plot_register(False)
            ax.set_title("Amplitude of Base States at iteration {} of Grover's Algorithm".format(i))
            ax.set_xlabel("Base State")
            ax.set_ylabel("Amplitude")
            plt.show()



    #register.plot_register()
    # Extract input register and reset auxillary qubit (hacky way)
    # register.remove_aux(1/np.sqrt(2))
    input, aux = register.split(n_qubits, 1)
    # Normalise, measure and return results
    register.normalise()



    measurement = input.measure()

    result = (input, measurement)

    return result



## Main and testing###
if __name__=='__main__':

    print("Grover's algorithm demonstration")
    print("\n")
    
    
    print("Oracle_single_tag initiated with 7 qubits, targeted state #4")
    print("\n")
    n=7
    oracle1=oracle_single_tag(n,4)

    n_runs = 50
    results = np.zeros(n_runs, dtype=int)
    for i in range(n_runs):
        measurement=grover(oracle1,k=1, plot=False)
        results[i] = measurement[1]

    # Return number measured most often together with the accuracy
    num_of_occurences = np.bincount(results)
    target_state = np.argmax((num_of_occurences) )
    accuracy = num_of_occurences[target_state]/n_runs * 100

    print('Grover search ran {} times.'.format(n_runs))
    print('Most likely state being tagged is {} with {}/100 confidence.'.format(target_state, accuracy))
