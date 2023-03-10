import os

import dill
import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import random_pauli, Pauli
from scipy.special import binom
from tqdm import tqdm

from fourier_vqa import PauliCircuit, FourierExpansionVQA


def save_experiment(experiment, save_to=None):
    if experiment.save_to:
        save_to = experiment.save_to
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, 'wb') as f:
        dill.dump(experiment, f)


def load_experiment(load_from):
    with open(load_from, 'rb') as f:
        return dill.load(f)


class NodeDistributionExperiment:
    def __init__(self, num_qubits, num_paulis, num_samples, save_to=None):
        self.num_qubits = num_qubits
        self.num_paulis = num_paulis
        self.num_samples = num_samples
        self.all_node_samples = None
        if save_to is None:
            save_to = f'results/NDE_{num_qubits}_{num_paulis}_{num_samples}'
        self.save_to = save_to

    def run(self, seed=0):
        all_node_samples = []

        np.random.seed(seed)
        seeds = np.random.randint(0, 10 ** 6, size=self.num_samples + 1)

        for seed in tqdm(seeds[:-1]):
            pauli_circuit = PauliCircuit.random(self.num_qubits, self.num_paulis, seed=seed)
            observable = random_pauli(self.num_qubits, seed=seeds[-1])

            fourier_computation = FourierExpansionVQA(pauli_circuit, observable)
            fourier_computation.compute(check_admissible=False, verbose=False)

            all_node_samples.append(fourier_computation.order_statistics())

        self.all_node_samples = np.array(all_node_samples)

    def node_stats(self):
        M = self.num_paulis
        normilized_node_samples = self.all_node_samples / (3 / 2) ** M

        node_means = np.mean(normilized_node_samples, axis=0)
        node_variations = np.std(normilized_node_samples, axis=0)

        return node_means, node_variations

    def norm_stats(self):
        M = self.num_paulis
        all_norm_samples = self.all_node_samples * 2. ** (-np.arange(M + 1))

        norm_means = np.mean(all_norm_samples, axis=0)
        norm_variations = np.std(all_norm_samples, axis=0)

        return norm_means, norm_variations

    def plot(self):
        M = self.num_paulis

        node_means, node_variations = self.node_stats()
        norm_means, norm_variations = self.norm_stats()

        c_node = 'blue'
        c_norm = 'orange'

        plt.plot(range(M + 1), node_means, color=c_node);
        plt.plot(range(M + 1), norm_means, color=c_norm);

        plt.plot(range(M + 1), random_node_distribution(M), color=c_node, linestyle='--')
        plt.plot(range(M + 1), random_norm_distribution(M), color=c_norm, linestyle='--')

        plt.fill_between(range(M + 1), node_means - node_variations, node_means + node_variations, alpha=0.75,
                         color=c_node);
        plt.fill_between(range(M + 1), norm_means - norm_variations, norm_means + norm_variations, alpha=0.75,
                         color=c_norm);

        plt.title(f'num_qubits={self.num_qubits}, num_paulis={self.num_paulis}, num_samples={self.num_samples}')


class QAOA:
    def __init__(self, graph, num_layers):
        self.graph = graph
        self.num_qubits = len(graph.nodes)
        self.num_layers = num_layers

    def circuit(self):
        qc = QuantumCircuit(self.num_qubits)

        # Hadamard gates
        for n in range(qc.num_qubits):
            qc.h(n)

        for p in range(self.num_layers):
            self.add_layer(qc, p)

        return qc

    def observables(self):

        labels = []
        for edge in self.graph.edges:
            i, j = edge
            label = ['I'] * self.num_qubits
            label[i] = 'Z'
            label[j] = 'Z'
            labels.append(''.join(label))

        return [Pauli(label) for label in labels]

    def add_layer(self, qc, p):
        x_parameters = [Parameter(f'x_{p}_{n}') for n in range(self.num_qubits)]
        z_parameters = [Parameter(f'z_{p}_{e}') for e in range(len(self.graph.edges))]

        for edge, z in zip(self.graph.edges, z_parameters):
            i, j = edge
            qc.rzz(z, i, j)

        for n, x in enumerate(x_parameters):
            qc.rx(x, n)


def two_local_circuit(num_qubits, num_paulis):
    if num_paulis < 2*num_qubits:
        missing_parameters = 0
        parameters0 = [Parameter(f'x_0{i}') for i in range(num_paulis)]
        parameters = []
    else:
        missing_parameters = (num_paulis-2*num_qubits) % 4
        parameters0 = [Parameter(f'x_0{i}') for i in range(2 * num_qubits, num_paulis)]
        parameters = [Parameter(f'x{i}') for i in range(num_paulis-2*num_qubits)] + [0] * missing_parameters
    blocks = int(len(parameters) / 4)

    print(missing_parameters)
    print(parameters0)
    print(parameters)
    print(blocks)

    qc = QuantumCircuit(num_qubits)

    for n in range(num_qubits):
        qc.rx(parameters0[n], n)
        qc.rz(parameters0[num_qubits + n], n)

    n = 0
    for b in range(blocks):
        p = parameters[b * 4:(b + 1) * 4]

        n0 = n
        n1 = (n + 1) % num_qubits

        qc.cz(n0, n1)
        if p[0]:
            qc.rx(p[0], n0)
        if p[1]:
            qc.rz(p[1], n0)
        if p[2]:
            qc.rx(p[2], n1)
        if p[3]:
            qc.rz(p[3], n1)

        n = (n + 1) % num_qubits

    return qc


def random_local_pauli(num_qubits, weight, seed=0):
    np.random.seed(seed)
    w = 0
    while w < weight:
        seed = np.random.randint(0, 2*32)
        local_pauli = random_pauli(weight, seed=seed)
        w = len(local_pauli.to_label().replace('I', ''))

    qubits = list(np.random.choice(range(num_qubits), weight, replace=False))
    pauli = Pauli('I'*num_qubits).compose(local_pauli, qargs=qubits)
    return pauli


def random_local_pauli_circuit(num_qubits, num_paulis, weight, seed=0):
    np.random.seed(seed)
    seeds = np.random.randint(0, 2**32, num_paulis)
    paulis = [random_local_pauli(num_qubits, weight, seed) for seed in seeds]
    return PauliCircuit(paulis)


class RandomLocalCircuit:
    def __init__(self, num_qubits, paulis):
        self.num_qubits = num_qubits
        self.paulis = paulis

    def estimate_node_count(self, coarse_observable):
        node_distribution = NodeDistribution.initialize_with_observable(coarse_observable)
        count_history = []
        for pauli in self.paulis[::-1]:
            node_distribution.update(pauli)
            count_history.append(node_distribution.counts_array.sum())
        return node_distribution, count_history

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


