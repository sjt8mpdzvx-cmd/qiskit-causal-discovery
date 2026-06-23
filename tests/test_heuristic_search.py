import unittest

import numpy as np
import pandas as pd

from src.heuristic_search import hill_climb_search


class HeuristicSearchTests(unittest.TestCase):
    def test_hill_climbing_returns_dag_with_parent_limit(self):
        rng = np.random.default_rng(19)
        data = pd.DataFrame(
            {
                "A": rng.integers(0, 3, size=40),
                "B": rng.integers(0, 3, size=40),
                "C": rng.integers(0, 3, size=40),
                "D": rng.integers(0, 3, size=40),
                "E": rng.integers(0, 3, size=40),
            }
        )
        result = hill_climb_search(data, list(data.columns), max_parents=2)
        graph = result["best_dag"]
        self.assertTrue(all(graph.in_degree(node) <= 2 for node in graph.nodes))
        self.assertGreaterEqual(result["evaluations"], 1)
        self.assertEqual(len(result["best_bitstring"]), 20)


if __name__ == "__main__":
    unittest.main()
