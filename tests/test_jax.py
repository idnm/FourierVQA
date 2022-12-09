import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap, jit
from matplotlib import pyplot as plt
from mynimize import OptOptions, mynimize
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector, random_clifford, random_statevector, random_unitary

from experiments_utils import Experiment
from jax_utils import jax_tensor, jax_fourier_mode, jax_loss
from test_wave_expansion import random_clifford_phi
from wave_expansion import CliffordPhi, Loss, CliffordPhiVQA


def equal_up_to_global_phase(s0, s1):
    index_of_max = np.unravel_index(np.abs(s0).argmax(), s0.shape)

    return np.allclose(s0/s0[index_of_max], s1/s1[index_of_max])


def test_jax_unitary(max_num_qubits=3, max_num_parameters=5):

    # Unitary test
    for num_qubits in range(2, max_num_qubits+1):
        for num_parameters in range(max_num_parameters+1):
            qc = random_clifford_phi(num_qubits, num_parameters)

            np.random.seed(0)
            parameters = np.random.rand(qc.num_parameters)
            qiskit_unitary = Operator(qc.bind_parameters(parameters)).data
            jax_unitary = jax_tensor(qc, 'id')(parameters)

            assert equal_up_to_global_phase(qiskit_unitary, jax_unitary)

    # State test
    for num_qubits in range(2, max_num_qubits+1):
        for num_parameters in range(max_num_parameters+1):
            qc = random_clifford_phi(num_qubits, num_parameters)

            np.random.seed(0)
            parameters = np.random.rand(qc.num_parameters)
            qiskit_state = Statevector.from_label('0'*num_qubits).evolve(qc.bind_parameters(parameters)).data
            jax_state = jax_tensor(qc, '0')(parameters)

            assert equal_up_to_global_phase(qiskit_state, jax_state)


def test_jax_fourier_mode(num_qubits=2, num_parameters=4):
    qc = random_clifford_phi(num_qubits, num_parameters)
    loss = Loss.from_state(random_statevector(2 ** num_qubits))

    vqa = CliffordPhiVQA(qc, loss)
    vqa.fourier_expansion()

    np.random.seed(0)
    num_samples = 10
    for fmode in vqa.fourier_modes:
        xx = np.random.rand(num_samples, qc.num_parameters)
        direct_values = [fmode.evaluate_at(x) for x in xx]
        # jax_values = [jax_fourier_mode(fmode)(x) for x in xx]
        jax_values = vmap(jit(jax_fourier_mode(fmode)))(xx)
        assert np.allclose(direct_values, jax_values)


def _test_jax_loss(num_qubits, num_parameters, loss):
    qc = random_clifford_phi(num_qubits, num_parameters)
    vqa = CliffordPhiVQA(qc, loss)

    key = jax.random.PRNGKey(0)
    num_samples = 10
    xx = jax.random.uniform(key, (num_samples, qc.num_parameters))

    direct_values = jnp.array([vqa.evaluate_loss_at(np.array(x)) for x in xx])
    jax_values = vmap(jit(jax_loss(vqa.circuit, vqa.loss.hamiltonian)))(xx)

    assert jnp.allclose(direct_values, jax_values)


def test_jax_loss(num_qubits=3, num_parameters=12):
    loss_state = Loss.from_state(random_statevector(2 ** num_qubits))
    loss_unitary = Loss.from_unitary(random_unitary(2 ** (num_qubits - 1)))  # One less qubit for speed.

    _test_jax_loss(num_qubits, num_parameters, loss_state)
    _test_jax_loss(num_qubits-1, num_parameters, loss_unitary)


def test_jit():
    num_qubits = 5
    depth = 0
    num_pauli_terms = 40

    e = Experiment(num_qubits, depth, num_pauli_terms)
    opt_options = OptOptions(learning_rate=0.01, num_iterations=5000)

    vqa = e.vqa
    vqa.fourier_expansion(up_to_order=2)
    f0 = jax_fourier_mode(vqa.fourier_modes[0])
    f1 = jax_fourier_mode(vqa.fourier_modes[1])
    f2 = jax_fourier_mode(vqa.fourier_modes[2])

    num_samples = 500
    angles_batch = Experiment.random_parameter_batch(num_samples, vqa.circuit.num_parameters)
    f1re = lambda x: jnp.real(f1(x))
    proxy_res = mynimize(f1re, angles_batch, opt_options)
    # plt.hist(np.array(proxy_res.all_best_losses))