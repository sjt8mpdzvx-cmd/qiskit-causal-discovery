"""고전 전수조사 기반 인과 구조 탐색."""

import time
import numpy as np
from .dag_utils import enumerate_all_dags, structural_hamming_distance, edge_metrics
from .scoring import score_all_dags


def classical_exhaustive_search(data, variables):
    """모든 유효 DAG를 열거하고 BDeu 점수로 최적 구조를 찾는다.

    Returns:
        result: dict with best_dag, best_bitstring, best_score, all_scores,
                n_valid_dags, n_total_candidates, elapsed_time
    """
    start_time = time.time()

    # 1. 모든 유효 DAG 열거
    valid_dags, edge_list = enumerate_all_dags(variables)
    n_edges = len(edge_list)
    n_total = 2 ** n_edges

    # 2. BDeu 점수 계산 및 정렬
    scored = score_all_dags(data, valid_dags, variables)

    elapsed = time.time() - start_time

    best_bitstring, best_dag, best_score = scored[0]

    return {
        "best_dag": best_dag,
        "best_bitstring": best_bitstring,
        "best_score": best_score,
        "best_bic": best_score,  # Backward-compatible alias.
        "all_scores": scored,
        "n_valid_dags": len(valid_dags),
        "n_total_candidates": n_total,
        "edge_list": edge_list,
        "elapsed_time": elapsed,
        "evaluations": len(valid_dags),
    }
