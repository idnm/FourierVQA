from collections import namedtuple
from random import sample, seed

import numpy as np
import jax.numpy as jnp
from jax import random
from matplotlib import pyplot as plt
from qiskit.circuit import Parameter
from qiskit.quantum_info import pauli_basis, SparsePauliOp

from jax_utils import jax_loss
from wave_expansion import Loss, CliffordPhi, CliffordPhiVQA
from mynimize import mynimize, OptOptions


def linear_ansatz_circuit(num_qubits, depth):
    qc = CliffordPhi(num_qubits)
    for i in range(num_qubits):
        # qc.rz(Parameter(f'z_{i}'), i)
        qc.rx(Parameter(f'x_0{i}'), i)
        qc.rz(Parameter(f'z_0{i}'), i)

    i = 0
    for d in range(1, depth + 1):
        i = i % num_qubits
        j = (i + 1) % num_qubits
        qc.cz(i, j)
        for k in (i, j):
            qc.rx(Parameter(f'x_{d}{k}'), k)
            # qc.ry(Parameter(f'y_{d}{k}'), k)
            qc.rz(Parameter(f'z_{d}{k}'), k)
        i += 1

    return qc


def random_pauli_loss(num_qubits, num_terms):
    np.random.seed(0)
    seed(0)
    paulis = sample([p.to_label() for p in pauli_basis(num_qubits)], num_terms)
    coeffs = np.random.rand(num_terms)/np.sqrt(num_terms)

    hamiltonian = SparsePauliOp.from_list(zip(paulis, coeffs))
    construct_circuit = lambda qc: qc
    return Loss(construct_circuit, hamiltonian)


class Experiment:
    def __init__(self, num_qubits, depth, num_pauli_terms):
        self.num_qubits = num_qubits
        self.depth = depth
        self.num_pauli_terms = num_pauli_terms

    @property
    def loss(self):
        return random_pauli_loss(self.num_qubits, self.num_pauli_terms)

    @property
    def vqa(self):
        qc = linear_ansatz_circuit(self.num_qubits, self.depth)
        return CliffordPhiVQA(qc, self.loss)

    @staticmethod
    def random_parameter_batch(num_samples, num_parameters):
        values_batch = 2 * jnp.pi * random.uniform(random.PRNGKey(0), (num_samples, num_parameters))
        return values_batch

    def loss_func(self):
        return jax_loss(self.vqa.circuit, self.vqa.loss.hamiltonian)

    def run(self, num_samples=100, opt_options=None):
        if not opt_options:
            opt_options = OptOptions(learning_rate=0.01, num_iterations=2000)

        angles_batch = self.random_parameter_batch(num_samples, self.vqa.circuit.num_parameters)
        loss_func = self.loss_func()
        res = mynimize(loss_func, angles_batch, opt_options)
        plt.hist(np.array(res.all_best_losses))
        return res