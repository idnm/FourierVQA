from typing import Union

from qiskit.quantum_info import Statevector, Pauli, Clifford
import jax.numpy as jnp

from wave_expansion import CliffordPhi


def jax_tensor(qc: CliffordPhi, initial_state: Union[str, jnp.array] = '0'):
    """JAX-transformable unitary or state of a Clifford Phi circuit."""

    num_qubits = qc.num_qubits

    if initial_state == '0':
        initial_shape = (2**num_qubits, )
        initial_state = Statevector.from_label('0'*num_qubits).data
        initial_state = initial_state.reshape([2]*num_qubits)
    elif initial_state == 'id':
        initial_shape = (2**num_qubits, 2**num_qubits)
        initial_state = jnp.identity(2**num_qubits)
        initial_state = initial_state.reshape([2] * 2 * num_qubits)
    else:
        initial_shape = initial_state.shape

    def tensor(parameters):
        permutation = qc.num_instruction_from_num_parameter
        inverse_permutation = jnp.argsort(jnp.array(permutation))

        parameters = jnp.array([parameters[i] for i in inverse_permutation])
        i = 0
        s = initial_state
        for gate, q_indices in qc.clifford_pauli_data:
            if CliffordPhi.is_clifford(gate):
                unitary = Clifford(gate).to_matrix()
            else:
                unitary = jax_pauli_rotation(gate.pauli, parameters[i])
                q_indices = reversed_indices(q_indices, num_qubits)  # Dirty solution, should be cleared up.
                i += 1

            unitary_tensor = unitary.reshape([2]*2*len(q_indices))
            s = apply_gate_to_tensor(unitary_tensor, s, list(q_indices))

        return s.reshape(initial_shape)

    return tensor


def reversed_indices(q_indices, num_qubits):
    return [range(num_qubits)[::-1][q] for q in q_indices][::-1]


def jax_pauli_rotation(pauli: Pauli, x):
    m = jnp.array(pauli.to_matrix())
    return jnp.cos(x/2)*jnp.identity(2**pauli.num_qubits)-1j*m*jnp.sin(x/2)


def gate_transposition(placement):
    """Determine transposition associated with initial placement of gate."""

    position_index = [(placement[i], i) for i in range(len(placement))]
    position_index.sort()
    trans = [i for _, i in position_index]
    return trans


def transposition(n_qubits, placement):
    """Return a transposition that relabels tensor axes correctly.
    Example (from the figure above): n=6, placement=[1, 3] gives [2, 0, 3, 1, 4, 5].
    Twiseted: n=6, placement=[3, 1] gives [2, 1, 3, 0, 4, 5]."""

    gate_width = len(placement)

    t = list(range(gate_width, n_qubits))

    for position, insertion in zip(sorted(placement), gate_transposition(placement)):
        t.insert(position, insertion)

    return t


def apply_gate_to_tensor(gate, tensor, placement):
    """Append `gate` to `tensor` along legs specified by `placement`. Transpose the output axes properly."""

    gate_width = int(len(gate.shape) / 2)
    tensor_width = int(len(tensor.shape) / 2)

    # contraction axes for `tensor` are input axes (=last half of all axes)
    gate_contraction_axes = list(range(gate_width, 2*gate_width))

    contraction = jnp.tensordot(gate, tensor, axes=[gate_contraction_axes, placement])

    # input(=last half) indices are intact
    t = transposition(tensor_width, placement) + list(range(tensor_width, 2*tensor_width))

    return jnp.transpose(contraction, axes=t)