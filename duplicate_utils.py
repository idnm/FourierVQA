from copy import deepcopy

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from jax_utils import all_parameters_from_parameters_func
from wave_expansion import CliffordPhi


def lift_duplicate_parameters(qc: QuantumCircuit):
    qc = CliffordPhi.from_quantum_circuit(qc)

    new_parameters = [Parameter(f'x{i}') for i in range(len(qc.all_parameters))]
    lifted_data = deepcopy(qc.data)

    num_parameter = 0
    for gate, qargs, cargs in lifted_data:
        if gate.is_parameterized():
            gate.params[0] = new_parameters[num_parameter]
            num_parameter += 1

    parameter_converter_original = all_parameters_from_parameters_func(qc.parameters, qc.all_parameters)

    qc_lifted = QuantumCircuit(qc.num_qubits)
    qc_lifted.data = lifted_data

    qc_lifted_clifford = CliffordPhi.from_quantum_circuit(qc_lifted)
    parameter_permutation = qc_lifted_clifford.num_instruction_from_num_parameter

    def converter(p):
        vals = parameter_converter_original(p)
        return [vals[i] for i in parameter_permutation]

    return qc_lifted, converter
