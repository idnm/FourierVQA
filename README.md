# Fourier expansion in variational quantum algorithms

Proof-of-principle implementation of the algorithm computing the Fourier series expansion of loss functions in variational quantum algorithms, see
https://arxiv.org/abs/?? for the original paper. Can handle e.g. random 50-qubit circuits with ~80 non-local Pauli rotations (and more or less arbitrary Clifford gates) on a laptop in minutes. A more efficient implementation is likely to allow scaling up circuit sizes by 50-100%. [This notebook](https://github.com/idnm/FourierVQA/blob/master/experiments.ipynb) contains usage examples and numerical data presented in the paper. 

## Minimal example

One way to use the package is to compute expectation values for large parameterized quantum circuits, here is a minimal example.
```python
# Imports
from qiskit.circuit.library import TwoLocal
from fourier_vqa import *

# Define a parameterized quantum circuit and a Pauli observable.
num_qubits = 20

qc = TwoLocal(num_qubits, rotation_blocks='ry', entanglement_blocks='cz', entanglement='linear', reps=7).decompose()
observable = Pauli('ZX'+'I'*(num_qubits-2))

# Parameterized quantum circuit needs to be parsed and transformed into the form containing Pauli rotations only.
pauli_circuit = PauliCircuit.from_parameterized_circuit(qc)

# A Pauli circuit and an observable is the input to the Fourier series computation.
fourier_expansion = FourierExpansionVQA(pauli_circuit, observable)
fourier_expansion.compute()
```
The result can be used to evaluate the loss function. Here we compare against the statevector simulation.
```python
# Sample random parameters.
np.random.seed(0)
random_angles = np.random.uniform(0, 2*np.pi, size=(qc.num_parameters))
# Because of the possible ordering ambiguity, we need to pass parameter values as dict.
random_parameters = {p: val for p, val in zip(qc.parameters, random_angles)} 

# Verify against the statevector simulation.
state = Statevector(qc.bind_parameters(random_parameters))
print('Statevector value:', state.expectation_value(observable))
print('Fourier value:', fourier_expansion.evaluate_loss_at(random_parameters))
```
Output:
```sh
Statevector value: 0.5618203770612687
Fourier value: 0.5618203770612659
```

