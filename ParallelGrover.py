"""
08/03/2018
Andreas Malekos - Quantum Computing Project

File that runs Grover's algorithm in parallel and returns the most likely
state that is tagged.

For now, this can only work in single target mode.
"""

from qc_simulator.qc import *
from qc_simulator.grover import grover
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def run_grover_parallel(oracle, n_runs):
    """
    Function that runs grover many times in parallel

    Inputs:
        oracle: oracle gate
        n_runs: number of runs
    Outputs:
        results: list of results
    """
    pass



if __name__=='__main__':  
