"""DAG 점수 함수 — BDeu / BGe 기반.

BDeu (Bayesian Dirichlet equivalent uniform): 이산 데이터 점수 함수.
BGe  (Bayesian Gaussian equivalent):          연속 데이터 점수 함수.
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


# ---------------------------------------------------------------------------
# BGe (Bayesian Gaussian equivalent) — 연속 데이터용 점수 함수
# ---------------------------------------------------------------------------

def _bge_marginal_likelihood(X, alpha_mu=1.0, alpha_w=None):
    """Normal-Wishart 사전분포 하에서 변수 집합의 로그 주변 우도 계산.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        연속 데이터 행렬 (n 샘플, p 변수).
    alpha_mu : float
        사전분포 평균 정밀도 스케일링. 클수록 강한 사전분포.
    alpha_w : float or None
        자유도 파라미터. None이면 ``p + 2`` (표준 기본값).

    Returns
    -------
    float
        로그 주변 우도 (log marginal likelihood).
    """
    n, p = X.shape

    if n < 2 or p < 1:
        return -np.inf

    if alpha_w is None:
        alpha_w = p + 2

    # alpha_w must be > p - 1 for a proper prior
    if alpha_w <= p - 1:
        alpha_w = p + 2

    # Sample mean
    mean_x = X.mean(axis=0)
    X_centered = X - mean_x

    # Data scatter matrix  (= (n-1) * sample covariance when n > 1)
    S = X_centered.T @ X_centered  # shape (p, p)

    # Prior scatter: T_0 = (alpha_w - p - 1) * I
    # This ensures E[Sigma] = I under the prior Wishart parametrization.
    t0_scale = alpha_w - p - 1
    if t0_scale <= 0:
        t0_scale = 1.0  # fallback for small alpha_w
    T_0 = np.eye(p) * t0_scale

    # Prior mean — non-informative: set to sample mean so correction = 0
    mu_0 = mean_x

    # Posterior scatter
    correction = (alpha_mu * n / (alpha_mu + n)) * np.outer(
        mean_x - mu_0, mean_x - mu_0
    )
    T_n = T_0 + S + correction  # correction is 0 when mu_0 = mean_x

    alpha_w_n = alpha_w + n

    # --- Log marginal likelihood ---
    score = -n * p / 2.0 * np.log(np.pi)

    # log ratio of prior / posterior precision determinants for mean
    score += (p / 2.0) * (np.log(alpha_mu) - np.log(alpha_mu + n))

    # Multivariate log-gamma ratio
    for i in range(1, p + 1):
        score += lgamma((alpha_w_n - i + 1) / 2.0) - lgamma(
            (alpha_w - i + 1) / 2.0
        )

    # Determinant terms (use slogdet for numerical stability)
    sign_0, logdet_0 = np.linalg.slogdet(T_0)
    sign_n, logdet_n = np.linalg.slogdet(T_n)

    if sign_0 <= 0 or sign_n <= 0:
        return -np.inf  # degenerate / singular matrix

    score += (alpha_w / 2.0) * logdet_0 - (alpha_w_n / 2.0) * logdet_n

    return score


def compute_local_bge(data, node, parents, alpha_mu=1.0, alpha_w=None):
    """BGe (Bayesian Gaussian equivalent) 로컬 점수.

    연속 데이터에 대해 이산화 없이 직접 주변 우도를 계산한다.
    Normal-Wishart 켤레 사전분포를 사용하므로 닫힌 형태로 계산 가능.

    local_bge(node, parents) = ML(node ∪ parents) − ML(parents)

    Parameters
    ----------
    data : pd.DataFrame
        연속 데이터.
    node : str
        대상 노드 이름.
    parents : list[str]
        부모 노드 이름 리스트.
    alpha_mu : float
        사전분포 평균 정밀도 (기본 1.0).
    alpha_w : float or None
        자유도. None이면 변수 수 + 2 (family 크기 기준).

    Returns
    -------
    float
        BGe 로컬 점수. 높을수록 좋은 모델.
    """
    n = len(data)
    if n < 2:
        return -np.inf

    parents = list(parents)
    family = [node] + parents

    # Extract numpy arrays
    X_family = data[family].values.astype(float)

    # Score for the full family (node + parents)
    score_family = _bge_marginal_likelihood(X_family, alpha_mu, alpha_w)

    # Score for parents only (subtract to get conditional contribution)
    if len(parents) == 0:
        score_parents = 0.0
    else:
        X_parents = data[parents].values.astype(float)
        score_parents = _bge_marginal_likelihood(X_parents, alpha_mu, alpha_w)

    return score_family - score_parents


def compute_dag_score_bge(data, dag, variables, alpha_mu=1.0, alpha_w=None):
    """전체 DAG의 BGe 점수 (각 노드의 local BGe score 합).

    높을수록 좋은 모델.

    Parameters
    ----------
    data : pd.DataFrame
        연속 데이터.
    dag : nx.DiGraph
        방향성 비순환 그래프.
    variables : list[str]
        변수 이름 리스트.
    alpha_mu : float
        사전분포 평균 정밀도.
    alpha_w : float or None
        자유도.

    Returns
    -------
    float
        DAG 전체 BGe 점수.
    """
    total = 0.0
    for node in variables:
        parents = list(dag.predecessors(node))
        total += compute_local_bge(data, node, parents, alpha_mu, alpha_w)
    return total


def score_all_dags_bge(data, valid_dags, variables, alpha_mu=1.0, alpha_w=None):
    """모든 유효 DAG에 대해 BGe 점수 계산.

    Parameters
    ----------
    data : pd.DataFrame
        연속 데이터.
    valid_dags : list of (bitstring, nx.DiGraph)
        유효한 DAG 리스트.
    variables : list[str]
        변수 이름 리스트.
    alpha_mu : float
        사전분포 평균 정밀도.
    alpha_w : float or None
        자유도.

    Returns
    -------
    list of (bitstring, nx.DiGraph, float)
        점수 내림차순 정렬 (좋은 모델 먼저).
    """
    scored = []
    for bitstring, dag in valid_dags:
        s = compute_dag_score_bge(data, dag, variables, alpha_mu, alpha_w)
        scored.append((bitstring, dag, s))

    # BGe도 높을수록 좋으므로 내림차순 정렬
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Local Score Decomposition — 다항적 사전 계산
# ---------------------------------------------------------------------------


def precompute_local_scores(data, variables, edge_list, equivalent_sample_size=10):
    """모든 (노드, 부모집합) 조합의 로컬 BDeu 점수를 사전 계산.

    BDeu는 분해 가능: Score(DAG) = Σ_node LocalScore(node, Parents(node)).
    따라서 전체 DAG를 전수조사(O(2^|E|))하지 않고,
    각 노드의 가능한 부모 조합만 평가하면 충분하다.

    복잡도: O(|V| × 2^(|V|-1)) — 검색 공간 O(2^(|V|×(|V|-1))) 대비 지수적 감소.
        3변수: 3 × 4  =  12 평가  (vs 전수조사 64)
        4변수: 4 × 8  =  32 평가  (vs 전수조사 4,096)
        5변수: 5 × 16 =  80 평가  (vs 전수조사 1,048,576)

    Returns
    -------
    dict[(str, frozenset[str]), float]
        (노드 이름, 부모 노드 집합) → 로컬 BDeu 점수
    """
    local_scores = {}
    for node in variables:
        other_vars = [v for v in variables if v != node]
        for mask in range(2 ** len(other_vars)):
            parents = frozenset(
                other_vars[bit] for bit in range(len(other_vars)) if mask & (1 << bit)
            )
            score = compute_local_bdeu(data, node, list(parents), equivalent_sample_size)
            local_scores[(node, parents)] = score
    return local_scores


def precompute_local_scores_bge(data, variables, edge_list, alpha_mu=1.0, alpha_w=None):
    """BGe 버전의 로컬 점수 사전 계산."""
    local_scores = {}
    for node in variables:
        other_vars = [v for v in variables if v != node]
        for mask in range(2 ** len(other_vars)):
            parents = frozenset(
                other_vars[bit] for bit in range(len(other_vars)) if mask & (1 << bit)
            )
            score = compute_local_bge(data, node, list(parents), alpha_mu, alpha_w)
            local_scores[(node, parents)] = score
    return local_scores


def score_bitstring_from_local(bitstring, edge_list, variables, local_scores):
    """로컬 점수 테이블에서 비트 문자열의 DAG 점수를 계산.

    데이터 접근 없이 O(|V|) 딕셔너리 룩업만으로 점수 산출.
    """
    total = 0.0
    for node in variables:
        parents = frozenset(
            src
            for q_idx, (src, dst) in enumerate(edge_list)
            if dst == node and bitstring[q_idx] == "1"
        )
        total += local_scores.get((node, parents), -np.inf)
    return total
