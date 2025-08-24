from itertools import combinations

from openfermion import QubitOperator
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..utils.conversions import openfermion_to_qiskit
from ..utils.molecules import Molecule
from .pool import Pool


class QEBPool(Pool):
    """
    The Qubit excitations pool: https://www.nature.com/articles/s42005-021-00730-0.
    Equivalent to the generalized excitations pools,
    but without the antisymmetry Z strings in the jordan wigner representation.
    """

    def __init__(self: Self, molecule: Molecule, n_excitations: int = 2) -> None:
        super().__init__(molecule)

        self.n_qubits = molecule.data.n_qubits
        self.n_excitations = n_excitations

        self.operators, self.labels = self.make_operators_and_labels()

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "qeb_pool"

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        Returns the config attributes of this class. Used in `Serializable`.
        """

        return ["n_excitations"]

    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        """
        The method that generates the pool operators for the molecule as well as a label for each operator
        Should return a tuple of two equal length lists, where each element in the first list
        is the pool operator and each element in the second list is the label for that operator
        """
        operators = []
        labels = []
        for n in range(1, self.n_excitations + 1):
            occupied = [*combinations(range(self.n_qubits), n)]
            virtual = [*combinations(range(self.n_qubits), n)]
            ops = [
                openfermion_to_qiskit(
                    2 ** (-2 * n)
                    * (
                        self._concat_ops(
                            [
                                QubitOperator(f"X{j}") + 1j * QubitOperator(f"Y{j}")
                                for j in v
                            ]
                        )
                        * self._concat_ops(
                            [
                                QubitOperator(f"X{j}") - 1j * QubitOperator(f"Y{j}")
                                for j in o
                            ]
                        )
                    ),
                    self.n_qubits,
                )
                for o in occupied
                for v in virtual
                if set(o).isdisjoint(set(v)) and min(v) > min(o)
            ]
            ops = [(op - op.conjugate().transpose()).simplify() for op in ops]
            operators += ops
            labels += [
                f"o{o}v{v}"
                for v in virtual
                for o in occupied
                if set(o).isdisjoint(set(v)) and min(v) > min(o)
            ]

        rem = set()
        for j, op in enumerate(operators):
            if all(j == 0 for j in op.coeffs):
                rem.add(j)
        operators = [k for j, k in enumerate(operators) if j not in rem]
        labels = [k for j, k in enumerate(labels) if j not in rem]

        return operators, labels

    def _concat_ops(self: Self, ops: list[QubitOperator]) -> QubitOperator:
        k = ops[0]
        for i in range(1, len(ops)):
            k = k * ops[i]
        return k

    @override
    def get_op(self: Self, idx: int) -> LinearOp:
        return self.operators[idx]

    @override
    def get_label(self: Self, idx: int) -> str:
        return self.labels[idx]

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
