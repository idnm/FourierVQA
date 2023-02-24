from collections import namedtuple
from functools import cached_property
from random import sample, seed

import numpy as np
import jax.numpy as jnp
from jax import random
from matplotlib import pyplot as plt
from qiskit.circuit import Parameter
from qiskit.quantum_info import pauli_basis, SparsePauliOp
from scipy.special import binom

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


def random_node_distribution(M):
    return [binom(M, m) * 2 ** (m - M) / (3 / 2) ** M for m in range(M + 1)]


def random_norm_distribution(M):
    return [binom(M, m) * 2 ** (-M) for m in range(M + 1)]


class RandomLocalCircuit:
    def __init__(self, num_qubits, paulis):
        self.num_qubits = num_qubits
        self.paulis = paulis

    def estimate_node_count(self, coarse_observable):
        node_distribution = NodeDistribution.initialize_with_observable(coarse_observable)
        for pauli in self.paulis[::-1]:
            node_distribution.update(pauli)
        return node_distribution

    @staticmethod
    def from_pauli_circuit(pauli_circuit):
        paulis = [CoarsePauli.from_pauli(pauli) for pauli in pauli_circuit.paulis]
        return RandomLocalCircuit(pauli_circuit.num_qubits, paulis)


class NodeDistribution:
    def __init__(self, counts_array):
        self.counts_array = counts_array
        self.num_qubits = self.counts_array.shape[0]-1

    @staticmethod
    def initialize_with_observable(observable):
        counts_array = np.zeros((observable.num_qubits+1, observable.num_qubits+1))
        counts_array[observable.z, observable.x] = 1
        return NodeDistribution(counts_array)

    def update(self, pauli):
        new_counts = np.zeros((self.num_qubits+1, self.num_qubits+1))
        for z in range(self.num_qubits+1):
            for x in range(self.num_qubits+1):
                current_count = self.counts_array[z, x]
                if current_count != 0:
                    observable = CoarsePauli(self.num_qubits, z, x)
                    node_distribution = NodeDistribution.from_observable_and_pauli(observable, pauli)
                    new_counts += node_distribution.counts_array*current_count
                    
        self.counts_array = new_counts

    @staticmethod
    def from_observable_and_pauli(observable, pauli):

        # Initialize zero counts array.
        node_distribution = NodeDistribution.initialize_with_observable(observable)
        commuting_probability = observable.commuting_probability(pauli)
        product_distribution = observable.product_distribution(pauli)

        node_distribution.counts_array += (1-commuting_probability)*product_distribution.counts_array

        return node_distribution

    def sorted(self):
        xx, yy = np.unravel_index(np.argsort(self.counts_array, axis=None), self.counts_array.shape)
        return [(x, y, self.counts_array[x, y]) for x, y in zip(xx, yy)][::-1]

    def __repr__(self, n=10):
        return f'{self.sorted()[:n]}'


class CoarsePauli:
    def __init__(self, num_qubits, z, x):
        self.num_qubits = num_qubits
        self.z = z
        self.x = x

    def commuting_probability(self, pauli):
        c_z = self.zx_commuting_probability(self.num_qubits, self.z, pauli.x)
        c_x = self.zx_commuting_probability(self.num_qubits, pauli.z, self.x)
        return c_z*c_x+(1-c_z)*(1-c_x)

    @staticmethod
    def zx_commuting_probability(num_qubits, z, x):
        kmin, kmax = CoarsePauli.limits_intersections(num_qubits, z, x)
        anti_commute_probability = 0
        for k in range(kmin, kmax+1):
            if k % 2 == 1:
                anti_commute_probability += CoarsePauli.intersection_probability(num_qubits, z, x, k)
        return 1-anti_commute_probability

    def product_distribution(self, pauli):
        z_distribution = CoarsePauli.z_product_distribution(self.num_qubits, self.z, pauli.z)
        x_distribution = CoarsePauli.z_product_distribution(self.num_qubits, self.x, pauli.x)
        return NodeDistribution(np.outer(z_distribution, x_distribution))

    @staticmethod
    def z_product_distribution(num_qubits, n1, n2):
        n1, n2 = sorted([n1, n2])
        kmin, kmax = CoarsePauli.limits_intersections(num_qubits, n1, n2)
        zz = np.zeros(num_qubits+1)
        for k in range(kmin, kmax+1):
            # If two Z pauli strings intersect along k positions, their product has this Z weight.
            n = n1 + n2 - 2*k
            zz[n] = CoarsePauli.intersection_probability(num_qubits, n1, n2, k)

        return zz

    @staticmethod
    def from_pauli(pauli):
        return CoarsePauli(pauli.num_qubits, sum(pauli.z), sum(pauli.x))

    @staticmethod
    def intersection_probability(num_qubits, n1, n2, k):
        n1, n2 = sorted([n1, n2])
        # Is this numerically stable?
        probability = binom(n1, k) * binom(num_qubits - n1, n2 - k) / binom(num_qubits, n2)
        return probability

    @staticmethod
    def limits_intersections(num_qubits, n1, n2):
        k_max = min(n1, n2)
        k_min = max(0, n1 + n2 - num_qubits)
        return k_min, k_max

    def __repr__(self):
        return f'({self.z}, {self.x})'


