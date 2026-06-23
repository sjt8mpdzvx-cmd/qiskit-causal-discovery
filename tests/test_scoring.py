import unittest

import numpy as np
import pandas as pd

from src.dag_utils import enumerate_all_dags
from src.scoring import (
    precompute_local_scores,
    precompute_local_scores_bge,
    score_all_dags,
    score_all_dags_bge,
    score_bitstring_from_local,
)


class LocalScoreDecompositionTests(unittest.TestCase):
    def test_bdeu_local_scores_reproduce_exhaustive_scores(self):
        data = pd.DataFrame(
            {
                "A": [0, 0, 1, 1, 0, 1, 0, 1],
                "B": [0, 1, 0, 1, 0, 1, 1, 0],
                "C": [1, 0, 1, 0, 1, 0, 0, 1],
            }
        )
        variables = list(data.columns)
        dags, edge_list = enumerate_all_dags(variables)

        exhaustive = score_all_dags(data, dags, variables)
        local = precompute_local_scores(data, variables, edge_list)

        for bitstring, _, score in exhaustive:
            self.assertAlmostEqual(
                score,
                score_bitstring_from_local(bitstring, edge_list, variables, local),
                places=10,
            )

    def test_bge_local_scores_reproduce_exhaustive_scores(self):
        rng = np.random.default_rng(7)
        data = pd.DataFrame(
            {
                "A": rng.normal(size=24),
                "B": rng.normal(size=24),
                "C": rng.normal(size=24),
            }
        )
        variables = list(data.columns)
        dags, edge_list = enumerate_all_dags(variables)

        exhaustive = score_all_dags_bge(data, dags, variables)
        local = precompute_local_scores_bge(data, variables, edge_list)

        for bitstring, _, score in exhaustive:
            self.assertAlmostEqual(
                score,
                score_bitstring_from_local(bitstring, edge_list, variables, local),
                places=10,
            )


if __name__ == "__main__":
    unittest.main()
