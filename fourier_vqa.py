import copy

import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QiskitError
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Clifford, StabilizerState, Pauli, Statevector, Operator, \
    pauli_basis, random_pauli
from scipy.special import binom
from tqdm import tqdm


# TODO
# - Report number of nodes in status
# - Profile runtime. How many nodes are processed?
# - Translate fourier expansion into JAX transformable loss.
# - Check numerically if additional filtering is feasible?
# -* Can make all computations JAXable?
# -* Can parallelize them for GPU?


class PauliCircuit:
    def __init__(self, paulis, final_clifford=None, parameters=None):
        self.paulis = paulis
        self.final_clifford = final_clifford
        self.parameters = parameters

    @property
    def num_qubits(self):
        return self.paulis[0].num_qubits

    @property
    def num_paulis(self):
        return len(self.paulis)

    def default_parameters(self):
        return [Parameter(f'x{i}') for i in range(len(self.paulis))]

    def to_parameterized_circuit(self, default_parameters=False):
        if self.parameters is not None and default_parameters is False:
            parameters = self.parameters
        else:
            parameters = self.default_parameters()

        qc = QuantumCircuit(self.num_qubits)
        for pauli, parameter in zip(self.paulis, parameters):
            gate = PauliEvolutionGate(pauli, 1/2*parameter)
            qc.append(gate, range(qc.num_qubits))
        if self.final_clifford:
            qc.append(self.final_clifford.to_instruction(), range(qc.num_qubits))

        return qc

    @staticmethod
    def default_parameters_to_values_dict(parameters, values):
        parameter_indices = [int(p.name[1:]) for p in parameters]
        return {parameter: values[i] for parameter, i in zip(parameters, parameter_indices)}

    def expectation_value(self, observable, parameter_values):
        qc = self.to_parameterized_circuit(default_parameters=True)
        parameters_dict = self.default_parameters_to_values_dict(qc.parameters, parameter_values)

        state = Statevector(qc.bind_parameters(parameters_dict))
        return state.expectation_value(observable)

    @staticmethod
    def from_parameterized_circuit(qc):
        gates, parameters = PauliCircuit.clifford_pauli_data(qc)
        pauli_gates, final_clifford = PauliCircuit.commute_all_cliffords_to_the_end(gates)
        return PauliCircuit(pauli_gates, final_clifford, parameters)

    @staticmethod
    def random(num_qubits, num_paulis, seed=0):
        np.random.seed(seed)
        seeds = np.random.randint(0, 10000, size=num_paulis)
        paulis = [random_pauli(num_qubits, seed=s) for s in seeds]

        return PauliCircuit(paulis)

    @staticmethod
    def commute_all_cliffords_to_the_end(gates):
        commuted_something = True
        while commuted_something:
            commuted_something, gates = PauliCircuit.commute_the_first_clifford(gates)
        num_pauli_gates = len([True for gate_type, _, _ in gates if gate_type == 'pauli'])
        pauli_gates = [gate for _, gate, _ in gates[:num_pauli_gates]]
        clifford_gates = gates[num_pauli_gates:]

        num_qubits = pauli_gates[0].num_qubits
        qc = QuantumCircuit(num_qubits)
        for _, gate, qubits in clifford_gates:
            qc.append(gate.to_instruction(), qubits)
        final_clifford = Clifford.from_circuit(qc)
        return pauli_gates, final_clifford

    @staticmethod
    def commute_the_first_clifford(gates):

        for i in range(len(gates) - 1):
            gate_type, gate, qubits = gates[i]
            next_gate_type, next_gate, next_qubits = gates[i + 1]
            if gate_type == 'clifford' and next_gate_type == 'pauli':
                gates[i] = 'pauli', next_gate.evolve(gate, qubits), next_qubits
                gates[i + 1] = 'clifford', gate, qubits
                commuted_something = True
                break
        else:
            commuted_something = False

        return commuted_something, gates

    @staticmethod
    def clifford_pauli_data(qc, parametric_to_pauli_dict=None):
        """Unroll the circuit according to circuit.data and identify each gate either as a Clifford gate
        or as a Pauli rotation."""

        if not parametric_to_pauli_dict:
            parametric_to_pauli_dict = {}
        gates = []
        parameters = []
        for gate, qargs, cargs in qc.data:
            qubits = [q._index for q in qargs]
            try:
                new_gate = Clifford(gate)
                gate_type = 'clifford'
            except (QiskitError, TypeError):
                if not gate.is_parameterized():
                    raise ValueError(f'Gate {gate.name} is neither clifford nor parametric.')
                if len(gate.params) != 1:
                    raise ValueError(
                        f'Parametric gates must have a single parameter. Got gate {gate.name} with gate.params={gate.params})')
                if gate.name in parametric_to_pauli_dict:
                    pauli = parametric_to_pauli_dict[gate.name]
                else:
                    pauli = PauliCircuit.pauli_generator_from_parametric_gate(gate)
                    parametric_to_pauli_dict[gate.name] = pauli

                parameter = gate.params[0]
                parameters.append(parameter)
                new_gate = PauliCircuit.full_pauli_generator(pauli, qubit_indices=qubits, num_qubits=qc.num_qubits)
                qubits = list(range(qc.num_qubits))
                gate_type = 'pauli'

            gates.append((gate_type, new_gate, qubits))

        return gates, parameters

    @staticmethod
    def pauli_generator_from_parametric_gate(gate):
        gate = copy.deepcopy(gate)
        gate.params = [Parameter('theta')]  # In case gate came with parameter like 0.5*Parameter('theta')
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, range(gate.num_qubits))

        # Check that gate(pi) is a pauli gate.
        gate_at_pi = Operator(qc.bind_parameters([np.pi])).data

        for pauli in pauli_basis(gate.num_qubits):
            if np.allclose(gate_at_pi, -1j * pauli.to_matrix()):
                pauli = pauli
                break
        else:
            raise ValueError(f'Gate {gate.name} at pi is not a pauli gate.')

        # Check that gate(x) is exponential of the pauli gate for other parameters.
        # This will fail e.g. for exp(i x (Z1+Z2)).
        xx = np.linspace(0, 2 * np.pi, 19)
        gate_values = np.array([Operator(qc.bind_parameters([x])).data for x in xx])
        pauli_rotation_values = np.array([PauliCircuit.pauli_rotation_matrix(pauli, x) for x in xx])

        if not np.allclose(gate_values, pauli_rotation_values):
            raise ValueError(f'Gate {gate.name} is not a Pauli rotation.')

        return pauli

    @staticmethod
    def pauli_rotation_matrix(pauli, x):
        return np.cos(x/2)*np.eye(2**pauli.num_qubits)-1j*pauli.to_matrix()*np.sin(x/2)

    @staticmethod
    def full_pauli_generator(pauli, qubit_indices, num_qubits):
        short_generator = pauli.to_label()
        generator = ['I'] * num_qubits
        for label, q in zip(short_generator, qubit_indices[::-1]):
            generator[q] = label

        return Pauli(''.join(generator)[::-1])


class PauliSpace:
    def __init__(self, paulis):
        self.paulis = paulis

        self.independent_paulis = None
        self.dependent_paulis = None
        self.normal_form = None
        self.decomposition_matrix = None

    @property
    def dim(self):
        return self.paulis[0].num_qubits

    @property
    def num_paulis(self):
        return len(self.paulis)

    def rank(self, up_to_pauli_num):
        return len([n for n in self.independent_paulis if n <= up_to_pauli_num])

    def list_decomposition(self, decomposition):
        if decomposition is None:
            return None
        return [self.independent_paulis[n] for n, coefficient in enumerate(decomposition) if coefficient]

    def decomposition_contains_pauli(self, decomposition, num_pauli):
        return num_pauli in self.list_decomposition(decomposition)

    def is_independent(self, num_pauli):
        return num_pauli in self.independent_paulis

    def decomposition_requires_paulis(self, decomposition):
        # if decomposition:
        if np.all(decomposition == 0):
            return 0
        return max(self.list_decomposition(decomposition))

    def compute_decomposition(self, observable, num_paulis):
        normal_form = self.normal_form[:num_paulis]
        decomposition_matrix = self.decomposition_matrix[:num_paulis]
        normal_form, decomposition_matrix = self.extend_normal_form(normal_form, decomposition_matrix, observable.x)

        # Decomposition successful.
        if np.all(normal_form[-1] == 0):
            return decomposition_matrix[-1]
        else:
            return None

    def construct(self):
        independent_paulis, dependent_paulis, normal_form, decomposition_matrix = self.basis_and_decompositions(self.paulis)
        self.independent_paulis = independent_paulis
        self.dependent_paulis = dependent_paulis
        self.normal_form = normal_form
        self.decomposition_matrix = decomposition_matrix

    @staticmethod
    def basis_and_decompositions(paulis):
        paulis_x = [pauli.x for pauli in paulis]

        independent_paulis = []
        dependent_paulis = []

        normal_form = []
        decomposition_matrix = []

        # Find first non-zero pauli.
        for n, pauli in enumerate(paulis_x):
            if np.all(pauli == 0):
                dependent_paulis.append(n)

                normal_form.append(np.zeros(len(pauli), dtype=bool))
                decomposition_matrix.append(np.zeros(len(pauli), dtype=bool))
            else:
                independent_paulis.append(n)

                d = np.zeros(len(pauli), dtype=bool)
                d[0] = True

                normal_form.append(pauli)
                decomposition_matrix.append(d)
                break
        else:
            raise ValueError('No non-trivial paulis provided.')

        for n, pauli in list(enumerate(paulis_x))[independent_paulis[0]+1:]:

            normal_form, decomposition_matrix = PauliSpace.extend_normal_form(normal_form, decomposition_matrix, pauli)

            if np.all(normal_form[-1] == 0):
                dependent_paulis.append(n)
            else:
                independent_paulis.append(n)
                decomposition_matrix[-1][len(independent_paulis)-1] = True

            if len(independent_paulis) == len(pauli):
                break

        return independent_paulis, dependent_paulis, normal_form, decomposition_matrix

    @staticmethod
    def extend_normal_form(normal_form, decomposition_matrix, pauli):
        pauli = copy.deepcopy(pauli)

        nonzero_rows = []
        starting_indices = []
        for n, row in enumerate(normal_form):
            if not np.all(row == 0):
                (i,) = row.nonzero()
                starting_indices.append(i[0])
                nonzero_rows.append(n)

        decomposition = np.zeros(len(pauli), dtype=bool)
        for i, n_row in sorted(zip(starting_indices, nonzero_rows)):
            if pauli[i]:
                pauli ^= normal_form[n_row]
                decomposition ^= decomposition_matrix[n_row]

        return normal_form+[pauli], decomposition_matrix+[decomposition]

    def update_decomposition(self, decomposition, num_pauli):
        if decomposition is None:
            return None

        if num_pauli in self.dependent_paulis:
            # For dependent paulis decompositions are stored in the decomposition matrix
            update = self.decomposition_matrix[num_pauli]
        else:
            # For independent paulis,
            # decomposition vector is one-hot vector with True at the number of pauli in the basis.
            update = np.eye(self.dim, dtype=bool)[self.independent_paulis.index(num_pauli)]

        return decomposition ^ update


class FourierExpansionVQA:
    def __init__(self, pauli_circuit, pauli_observable):
        self.pauli_circuit = pauli_circuit
        pauli_space = PauliSpace(pauli_circuit.paulis)
        pauli_space.construct()
        self.pauli_space = pauli_space

        self.original_observable = pauli_observable
        if self.pauli_circuit.final_clifford:
            self.observable = self.original_observable.evolve(self.pauli_circuit.final_clifford)
        else:
            self.observable = self.original_observable

        self.complete_nodes = []
        self.incomplete_nodes = []
        self.num_iterations = 0

    @property
    def num_qubits(self):
        return self.pauli_circuit.num_qubits

    @property
    def num_paulis(self):
        return self.pauli_circuit.num_paulis

    @staticmethod
    def random(num_qubits, num_paulis, seed=0):
        np.random.seed(seed)
        seeds = np.random.randint(0, 10**6, 2)
        pauli_circuit = PauliCircuit.random(num_qubits, num_paulis, seed=seeds[0])
        observable = random_pauli(num_qubits, seed=seeds[1])
        return FourierExpansionVQA(pauli_circuit, observable)

    def sample(self, check_admissible, seed):
        np.random.seed(seed)
        node = FourierComputationNode(self.pauli_circuit.num_paulis, self.observable, ())
        node.remove_commuting_paulis(self.pauli_space)

        branch_is_admissible = True
        probability = 1
        # while node.num_paulis > 0 and branch_is_admissible:
        # print(node)
        while node.num_paulis > 0:

            if check_admissible and self.pauli_space.rank(node.num_paulis) < self.pauli_space.dim:
                node.update_admissibility(self.pauli_space)
                branch_is_admissible = node.is_admissible
                if not branch_is_admissible:
                    break
                cos_admissible, sin_admissible = node.cos_sin_admissible(self.pauli_space)
            else:
                cos_admissible, sin_admissible = True, True

            # If both branches are admissible, we choose one randomly and decrease the probability.
            # If only a single branch is admissible, we choose that branch, without decreasing the probability.
            if cos_admissible and sin_admissible:
                # With probability 1/2 observable is unchanged or multiplied by the next pauli.
                probability /= 2.
                random_bool = np.random.randint(2)
                if random_bool:
                    branching_function = node.branch_cos
                else:
                    branching_function = node.branch_sin
            elif cos_admissible and not sin_admissible:
                branching_function = node.branch_cos
            elif not cos_admissible and sin_admissible:
                branching_function = node.branch_sin
            else:
                raise RuntimeError('Should not be here')

            node = branching_function(self.pauli_space)
            node.remove_commuting_paulis(self.pauli_space)
            # if check_admissible and self.pauli_space.rank(node.num_paulis) < self.pauli_space.dim:
            #     node.update_admissibility(self.pauli_space)
            #     branch_is_admissible = node.is_admissible

        if branch_is_admissible:
            node.compute_expectation_value()
        else:
            node.expectation_value = 0
        return node, probability

    def estimate_node_count_monte_carlo(self, num_samples=1000, check_admissible=False, seed=0):
        np.random.seed(seed)
        seeds = np.random.randint(0, 2**32, num_samples)

        all_nodes = [self.sample(check_admissible, seed) for seed in tqdm(seeds)]
        nonzero_nodes = [[node, prob] for node, prob in all_nodes if node.expectation_value != 0]

        estimated_num_all_nodes = sum([1./prob for _, prob in all_nodes]) / len(all_nodes)
        estimated_num_nonzero_nodes = sum([1./prob for _, prob in nonzero_nodes]) / len(all_nodes)

        return estimated_num_all_nodes, estimated_num_nonzero_nodes

    def estimate_node_count_limited_volume(self, max_nodes=1000, seed=0, verbose=True):
        np.random.seed(seed)

        self.initialize_computation()
        progress_bar = tqdm(range(self.pauli_space.num_paulis), disable=not verbose)

        node_stats = np.zeros(self.pauli_space.num_paulis + 1)
        current_multiplier = 1
        for level in progress_bar:
            num_complete_nodes = len(self.complete_nodes)
            self.incomplete_nodes, self.complete_nodes = self.iteration(
                self.incomplete_nodes, self.complete_nodes, check_admissible=False)

            num_completed_nodes = len(self.complete_nodes) - num_complete_nodes

            num_incomplete_nodes = len(self.incomplete_nodes)
            if num_incomplete_nodes > max_nodes:
                self.incomplete_nodes = list(np.random.choice(self.incomplete_nodes, max_nodes, replace=False))
                current_multiplier *= num_incomplete_nodes / max_nodes

            node_stats[level] += num_completed_nodes*current_multiplier

            if len(self.incomplete_nodes) == 0:
                break

        return node_stats

    def initialize_computation(self):
        root = FourierComputationNode(self.pauli_space.num_paulis, self.observable, ())
        root.remove_commuting_paulis(self.pauli_space)
        if root.is_complete:
            self.complete_nodes = [root]
            self.incomplete_nodes = []
        else:
            self.incomplete_nodes = [root]
            self.complete_nodes = []

    def compute(self, check_admissible=True, verbose=True):

        self.initialize_computation()

        # Run recursive algorithm. Each iteration computes all Fourier terms of at the next level.
        progress_bar = tqdm(range(self.pauli_space.num_paulis), disable=not verbose)
        for _ in progress_bar:
            self.incomplete_nodes, self.complete_nodes = self.iteration(
                self.incomplete_nodes, self.complete_nodes, check_admissible)

            if verbose:
                progress_bar.set_description(self.status(), refresh=True)

            if len(self.incomplete_nodes) == 0:
                break

    def iteration(self, incomplete_nodes, complete_nodes, check_admissible):
        if not incomplete_nodes:
            return [], complete_nodes

        new_incomplete_nodes = []
        for node in incomplete_nodes:
            incomplete_nodes, completed_nodes = node.branch_and_refine(self.pauli_space, check_admissible)

            new_incomplete_nodes.extend(incomplete_nodes)
            complete_nodes.extend(completed_nodes)

        return new_incomplete_nodes, complete_nodes

    def status_data(self):
        norm_covered = sum(self.stats(only_nonzero=False).norm_stats)
        norm_found = sum(self.stats(only_nonzero=True).norm_stats)
        norm_remaining_bound = self.bound_remaining_norm()
        try:
            relative_norm_found = norm_found/(norm_found+norm_remaining_bound)
            relative_norm_covered = norm_covered/(norm_covered+norm_remaining_bound)
        except ZeroDivisionError:
            relative_norm_covered = 1
            relative_norm_found = 1

        return relative_norm_covered, relative_norm_found, norm_found, norm_remaining_bound

    def status(self):
        covered, relative, absolute, remaining = self.status_data()
        num_incomplete_nodes = len(self.incomplete_nodes)
        volume = num_incomplete_nodes*self.num_qubits

        return f'cov {covered:.2%} abs {absolute:.2g} rel {relative:.2%} rem {remaining:.2g} nodes {num_incomplete_nodes:.2e} vol {volume:.2e}'

    @staticmethod
    def level_statistic(nodes, M):
        levels = [node.level for node in nodes]
        stats = [levels.count(m) for m in range(M + 1)]
        return stats

    def bound_remaining_norm(self):
        M = self.num_paulis
        stats = self.level_statistic(self.incomplete_nodes, M)
        return sum([n / (2 ** m) for m, n in enumerate(stats)])

    def evaluate_loss_at(self, parameters):
        state0 = StabilizerState(Pauli('I'*self.pauli_circuit.num_qubits))
        results = [state0.expectation_value(node.observable)*node.monomial(parameters) for node in self.complete_nodes]

        return sum(results)

    def stats(self, only_nonzero=False):
        if only_nonzero:
            nodes = [node for node in self.complete_nodes if node.observable != 0]
        else:
            nodes = self.complete_nodes

        return FourierStats(nodes, self.num_paulis)


class FourierComputationNode:
    def __init__(self, num_paulis, observable, branch_history):
        self.num_paulis = num_paulis
        self.observable = observable
        self.branch_history = branch_history
        self.is_admissible = None
        self.observable_decomposition = None
        self.expectation_value = None

    @property
    def level(self):
        return len(self.branch_history)

    @property
    def is_complete(self):
        # The node is complete is no Paulis remain.
        node_is_complete = self.num_paulis == 0
        if node_is_complete:
            self.compute_expectation_value()
        return node_is_complete

    def __repr__(self):
        return f'{self.expectation_value} {self.num_paulis} {self.observable} {self.branch_history}'

    def remove_commuting_paulis(self, pauli_space):
        paulis = pauli_space.paulis[:self.num_paulis]
        num_commuting = 0
        for pauli in paulis[::-1]:
            if pauli.commutes(self.observable):
                num_commuting += 1
            else:
                break
        self.num_paulis -= num_commuting

    def branch_cos(self, pauli_space):
        node_cos = FourierComputationNode(
            self.num_paulis-1,
            self.observable,
            self.branch_history+((self.num_paulis-1, 0), ))
        node_cos.observable_decomposition = self.observable_decomposition
        return node_cos

    def branch_sin(self, pauli_space):
        pauli_product = self.observable.compose(pauli_space.paulis[self.num_paulis-1])
        pauli_product.phase += 3  # Imitates multiplying by 1j but is faster.
        node_sin = FourierComputationNode(
            self.num_paulis - 1,
            pauli_product,
            self.branch_history + ((self.num_paulis-1, 1), ))
        decomposition = pauli_space.update_decomposition(self.observable_decomposition, self.num_paulis-1)
        node_sin.observable_decomposition = decomposition
        return node_sin

    def update_admissibility(self, pauli_space):
        self.is_admissible = False

        # If the observable was not decomposed before, decompose it now.
        if self.observable_decomposition is None:
            self.observable_decomposition = pauli_space.compute_decomposition(self.observable, self.num_paulis)

        # If decomposition exists.
        if self.observable_decomposition is not None:
            # If the number of available pauli matrices is enough for decomposition, the node is admissible.
            if pauli_space.decomposition_requires_paulis(self.observable_decomposition) <= self.num_paulis:
                self.is_admissible = True

    def cos_sin_admissible(self, pauli_space):
        # If the branching Pauli is dependent, both branches are admissible.
        # If it is independent a single branch is admissible.

        idx_branching_pauli = self.num_paulis - 1
        if pauli_space.is_independent(idx_branching_pauli):
            # If observable requires the branching Pauli in its decomposition the cos branch is not admissible.
            if pauli_space.decomposition_contains_pauli(self.observable_decomposition, idx_branching_pauli):
                cos_admissible, sin_admissible = False, True
            # Else the sin branch is not admissible.
            else:
                cos_admissible, sin_admissible = True, False
        else:
            cos_admissible, sin_admissible = True, True
        return cos_admissible, sin_admissible

    def branch_and_refine(self, pauli_space, check_admissible):

        # Each node processes here if at the branching point. Before branching,
        # we can check admissibility of the current node itself, and if admissible, of its two branches.

        cos_admissible = True
        sin_admissible = True

        # Admissibility is checked if the corresponding flag is true,
        # and the remaining pauli operators are insufficient to decompose any observable.

        if check_admissible and pauli_space.rank(self.num_paulis) < pauli_space.dim:
            self.update_admissibility(pauli_space)

            # If the node is not admissible, in the end, abort the computation.
            if not self.is_admissible:
                self.expectation_value = 0
                complete_nodes = [self]
                incomplete_nodes = []
                return incomplete_nodes, complete_nodes

            cos_admissible, sin_admissible = self.cos_sin_admissible(pauli_space)

        # Branch and remove commuting paulis from each branch.
        nodes = []
        if cos_admissible:
            nodes.append(self.branch_cos(pauli_space))
        if sin_admissible:
            nodes.append(self.branch_sin(pauli_space))

        for node in nodes:
            node.remove_commuting_paulis(pauli_space)

        incomplete_nodes = []
        complete_nodes = []
        for node in nodes:
            if node.is_complete:
                complete_nodes.append(node)
            else:
                incomplete_nodes.append(node)

        return incomplete_nodes, complete_nodes

    def monomial(self, parameters):
        res = 1
        for num_parameter, branch in self.branch_history:
            x = parameters[num_parameter]
            if branch == 0:
                res *= np.cos(x)
            elif branch == 1:
                res *= np.sin(x)

        return res

    def compute_expectation_value(self): 
        # state = StabilizerState(Pauli('I'*self.observable.num_qubits))
        # expectation_value = state.expectation_value(self.observable)

        absolute_expectation_value = not np.any(self.observable.x)  # Observable should have no X operators.
        sign = -1*(self.observable.phase == 2)  # Phases 1 and 3 correspond to -1j and 1j and never appear. Phase 2 is -1.
        self.expectation_value = absolute_expectation_value * sign
        # return expectation_value


class FourierStats:
    def __init__(self, nodes, num_paulis):
        self.nodes = nodes
        self.num_paulis = num_paulis

        self.node_stats = self._stats_from_nodes(self.nodes, self.num_paulis)
        self.norm_stats = self._norm_from_node_stats(self.node_stats)
        self.node_stats_normalized = np.array(self.node_stats) / sum(self.node_stats)
        self.norm_stats_normalized = np.array(self.norm_stats) / sum(self.norm_stats)
        self.num_nodes = sum(self.node_stats)
        self.num_qubits = self.nodes[0].observable.num_qubits

    node_color = 'blue'
    norm_color = 'orange'

    @staticmethod
    def _norm_from_node_stats(node_stats):
        node_stats = np.array(node_stats)
        norm_stats = node_stats * 2. ** (-np.arange(len(node_stats)))
        return norm_stats

    @staticmethod
    def _stats_from_nodes(nodes, max_level):

        stats = []
        for lvl in range(max_level+1):
            nodes_at_level = [node for node in nodes if node.level == lvl]
            stats.append(len(nodes_at_level))

        return stats

    @staticmethod
    def random_node_distribution(M):
        return [binom(M, m) * 2 ** (m - M) / (3 / 2) ** M for m in range(M + 1)]

    @staticmethod
    def random_norm_distribution(M):
        return [binom(M, m) * 2 ** (-M) for m in range(M + 1)]

    def plot(self, plot_random=True, max_level=None):

        M = len(self.node_stats) - 1

        if plot_random:
            self.plot_random(M)

        if max_level is None:
            max_level = M

        self.plot_scatter(self.node_stats_normalized, self.norm_stats_normalized, max_level)
        plt.title(f'number of  qubits={self.num_qubits}, number of paulis={self.num_paulis}')
        plt.legend()

    @staticmethod
    def plot_scatter(node_stats, norm_stats, max_level):
        plt.scatter(range(max_level + 1), node_stats, color=FourierStats.node_color, edgecolors='black',
                    label='node distribution');
        plt.scatter(range(max_level + 1), norm_stats, color=FourierStats.norm_color, edgecolors='black',
                    label='norm distribution');

        plt.grid(linestyle='--')
        plt.ylabel('Probability', fontsize=12)
        plt.xlabel('Level', fontsize=12)

    @staticmethod
    def plot_random(M):
        plt.plot(range(M + 1), FourierStats.random_node_distribution(M), color=FourierStats.node_color, linewidth=2, alpha=0.9,
                 label='random node distribution')
        plt.plot(range(M + 1), FourierStats.random_norm_distribution(M), color=FourierStats.norm_color, linewidth=2, alpha=0.9,
                 label='random norm distribution')

    @staticmethod
    def plot_several(samples, plot_random=True, max_level=None):

        M = samples[0].num_paulis

        if plot_random:
            FourierStats.plot_random(M)

        if max_level is None:
            max_level = M

        node_samples = np.array([sample.node_stats_normalized for sample in samples])
        norm_samples = np.array([sample.norm_stats_normalized for sample in samples])

        # norm_samples = all_node_samples * 2. ** (-np.arange(M + 1))
        # normilized_node_samples = all_node_samples / all_node_samples.sum(axis=1, keepdims=True)

        node_means = np.mean(node_samples, axis=0)[:max_level + 1]
        node_variations = np.std(node_samples, axis=0)[:max_level + 1]

        norm_means = np.mean(norm_samples, axis=0)[:max_level + 1]
        norm_variations = np.std(norm_samples, axis=0)[:max_level + 1]

        plt.fill_between(range(max_level + 1), node_means - node_variations, node_means + node_variations, alpha=0.5,
                         color=FourierStats.node_color, edgecolors='black', label='node variance');
        plt.fill_between(range(max_level + 1), norm_means - norm_variations, norm_means + norm_variations, alpha=0.5,
                         color=FourierStats.norm_color, edgecolors='black', label='norm variance');

        FourierStats.plot_scatter(node_means, norm_means, max_level)
        # plt.scatter(range(max_level + 1), node_means, color=node_color, edgecolors='black', label='node distribution');
        # plt.scatter(range(max_level + 1), norm_means, color=norm_color, edgecolors='black', label='norm distribution');

        # plt.grid(linestyle='--')
        # plt.ylabel('Probability', fontsize=12)
        # plt.xlabel('Level', fontsize=12)
        # #     plt.title(f'num_qubits={num_qubits}, num_paulis={M}, num_samples={num_samples}')
        # plt.legend()