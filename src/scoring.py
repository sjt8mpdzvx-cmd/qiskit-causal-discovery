"""DAG 점수 함수 — BDeu (Bayesian Dirichlet equivalent uniform) 기반.

Sachs 데이터는 이산(0,1,2)이므로, 연속 BIC 대신
이산 데이터에 적합한 BDeu 점수를 사용한다.
"""

import numpy as np
import pandas as pd
from math import lgamma


def compute_local_bdeu(data, node, parents, equivalent_sample_size=10):
    """단일 노드의 BDeu 점수 계산.

    BDeu는 베이지안 네트워크에서 이산 변수의 표준 점수 함수.
    점수가 높을수록(덜 음수) 좋은 모델.

    alpha = equivalent_sample_size (ESS): 사전분포 강도.
    값이 작을수록 희소한 구조를 선호 (오컴의 면도날).
    """
    node_values = sorted(data[node].unique())
    r_i = len(node_values)  # 노드의 가능한 값 수

    if len(parents) == 0:
        # 부모 없음: 단순 다항 분포
        q_i = 1  # 부모 조합 수 = 1
        alpha_ij = equivalent_sample_size / q_i
        alpha_ijk = alpha_ij / r_i

        counts = data[node].value_counts()
        n_ij = len(data)

        score = lgamma(alpha_ij) - lgamma(alpha_ij + n_ij)
        for val in node_values:
            n_ijk = counts.get(val, 0)
            score += lgamma(alpha_ijk + n_ijk) - lgamma(alpha_ijk)
        return score

    # 부모 있음: 각 부모 값 조합별로 계산
    parent_list = list(parents)
    parent_combos = data.groupby(parent_list)

    q_i = 1
    for p in parent_list:
        q_i *= len(data[p].unique())

    alpha_ij = equivalent_sample_size / q_i
    alpha_ijk = alpha_ij / r_i

    score = 0.0
    seen_combos = set()

    for combo, group in parent_combos:
        if not isinstance(combo, tuple):
            combo = (combo,)
        seen_combos.add(combo)

        n_ij = len(group)
        score += lgamma(alpha_ij) - lgamma(alpha_ij + n_ij)

        counts = group[node].value_counts()
        for val in node_values:
            n_ijk = counts.get(val, 0)
            score += lgamma(alpha_ijk + n_ijk) - lgamma(alpha_ijk)

    # 데이터에 없는 부모 조합: N_ij = 0이므로 기여 = 0 (lgamma 상쇄)
    return score


def compute_dag_score(data, dag, variables, equivalent_sample_size=10):
    """전체 DAG의 BDeu 점수 (각 노드의 local score 합).
    높을수록 좋은 모델."""
    total = 0.0
    for node in variables:
        parents = list(dag.predecessors(node))
        total += compute_local_bdeu(data, node, parents, equivalent_sample_size)
    return total


def score_all_dags(data, valid_dags, variables, equivalent_sample_size=10):
    """모든 유효 DAG에 대해 BDeu 점수 계산.

    Returns:
        scores: list of (bitstring, dag, score) sorted by score descending (= better first)
    """
    scored = []
    for bitstring, dag in valid_dags:
        s = compute_dag_score(data, dag, variables, equivalent_sample_size)
        scored.append((bitstring, dag, s))

    # BDeu는 높을수록 좋으므로 내림차순 정렬
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored
