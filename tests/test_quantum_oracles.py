import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from src.grover_search import (
    _add_qft_unconditional,
    _apply_iqft,
    _apply_qft,
    build_oracle_incircuit,
    build_penalty_oracle,
)
from src.qaoa_search import build_cost_operator, build_cost_operator_local


class QuantumOracleTests(unittest.TestCase):
    def assert_diagonal(self, circuit):
        matrix = Operator(circuit).data
        off_diagonal = matrix - np.diag(np.diag(matrix))
        self.assertLess(np.max(np.abs(off_diagonal)), 1e-10)

    def test_cost_and_penalty_operators_are_diagonal(self):
        self.assert_diagonal(
            build_cost_operator(2, [("00", None, 0.0), ("11", None, 1.0)], 0.7)
        )
        self.assert_diagonal(build_penalty_oracle(2, [], ["11"], 0.7))

        variables = ["A", "B", "C"]
        edge_list = [(source, target) for source in variables for target in variables if source != target]
        local_scores = {}
        for node_index, node in enumerate(variables):
            others = [candidate for candidate in variables if candidate != node]
            for mask in range(2 ** len(others)):
                parents = frozenset(
                    candidate for bit, candidate in enumerate(others) if mask & (1 << bit)
                )
                local_scores[(node, parents)] = float(node_index + mask)
        self.assert_diagonal(
            build_cost_operator_local(len(edge_list), edge_list, variables, local_scores, 0.7)
        )

    def test_qft_constant_adder_preserves_basis_states(self):
        n_bits = 3
        for value in range(2**n_bits):
            circuit = QuantumCircuit(n_bits)
            register = list(range(n_bits))
            _apply_qft(circuit, register)
            _add_qft_unconditional(circuit, register, value, n_bits)
            _apply_iqft(circuit, register)
            state = Statevector.from_instruction(circuit).data
            self.assertEqual(int(np.argmax(np.abs(state))), value)
            self.assertGreater(abs(state[value]), 1 - 1e-10)

    def test_incircuit_oracle_only_marks_qualifying_states(self):
        variables = ["A", "B"]
        edge_list = [("A", "B"), ("B", "A")]
        local_scores = {
            ("A", frozenset()): 0.0,
            ("A", frozenset({"B"})): 1.0,
            ("B", frozenset()): 0.0,
            ("B", frozenset({"A"})): 2.0,
        }
        oracle, n_total, _ = build_oracle_incircuit(
            2, edge_list, variables, local_scores, threshold_ratio=0.5, n_score_bits=4
        )

        # Quantized scores make edge states 01 and 11 qualify at this threshold.
        expected_phase = {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0}
        for basis_state, phase in expected_phase.items():
            initial = np.zeros(2**n_total, dtype=complex)
            initial[basis_state] = 1.0
            result = Statevector(initial).evolve(oracle).data
            self.assertAlmostEqual(result[basis_state].real, phase, places=10)
            self.assertAlmostEqual(result[basis_state].imag, 0.0, places=10)
            self.assertLess(np.linalg.norm(np.delete(result, basis_state)), 1e-10)

    def test_incircuit_oracle_uncomputes_three_variable_ancillas(self):
        variables = ["A", "B", "C"]
        edge_list = [(source, target) for source in variables for target in variables if source != target]
        local_scores = {}
        for node_index, node in enumerate(variables):
            others = [candidate for candidate in variables if candidate != node]
            for mask in range(2 ** len(others)):
                parents = frozenset(
                    candidate for bit, candidate in enumerate(others) if mask & (1 << bit)
                )
                local_scores[(node, parents)] = float(node_index + mask)

        oracle, n_total, _ = build_oracle_incircuit(
            len(edge_list), edge_list, variables, local_scores, threshold_ratio=0.5, n_score_bits=4
        )
        for basis_state in (0, 3, 63):
            initial = np.zeros(2**n_total, dtype=complex)
            initial[basis_state] = 1.0
            result = Statevector(initial).evolve(oracle).data
            self.assertAlmostEqual(abs(result[basis_state]), 1.0, places=10)
            self.assertLess(np.linalg.norm(np.delete(result, basis_state)), 1e-10)


if __name__ == "__main__":
    unittest.main()
