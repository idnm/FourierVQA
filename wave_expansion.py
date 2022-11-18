from functools import reduce

import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)


def multi_kronecker(matrix_list):
    return reduce(lambda m1, m2: np.kron(m1, m2), matrix_list)


class PauliString:
    def __init__(self, z, x):
        assert len(z) == len(x), f'Z and X stabilizers must have the same length, got Z:{len(z)} X:{len(x)}'
        self.z = z
        self.x = x
        self.num_qubits = len(z)

    def unitary(self):
        z_list = [Z if z == 1 else np.identity(2) for z in self.z]
        x_list = [X if x == 1 else np.identity(2) for x in self.x]
        return multi_kronecker(x_list) @ multi_kronecker(z_list)

    @staticmethod
    def product(s1, s2):
        z12 = (s1.z + s2.z) % 2
        x12 = (s1.x + s2.x) % 2
        return PauliString(z12, x12)

    @staticmethod
    def trivial(num_qubits):
        z = np.zeros(num_qubits, dtype=int)
        x = np.zeros(num_qubits, dtype=int)
        return PauliString(z, x)

    def __repr__(self):
        z_str = ''.join([str(z) for z in self.z])
        x_str = ''.join([str(x) for x in self.x])
        return f'PauliString({z_str}|{x_str})'

    def __eq__(self, other):
        if isinstance(other, PauliString):
            return np.array_equal(self.z, other.z) and np.array_equal(self.x, other.x)
        return False

    def __hash__(self):
        return int(''.join([str(i) for i in np.concatenate([self.z, self.x])]), 2)

