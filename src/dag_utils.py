"""DAG 인코딩/디코딩 및 비순환 검증 유틸리티."""

import itertools
import networkx as nx
import numpy as np


def get_variable_names(n=3):
    """분석에 사용할 단백질 변수명 반환.
    n=3: MAPK 캐스케이드 핵심 (Raf, Mek, Erk)
    n=4: + Akt 포함
    """
    all_vars = ["Raf", "Mek", "Erk", "Akt"]
    return all_vars[:n]


def get_edge_list(variables):
    """변수 목록에서 가능한 모든 방향 엣지 목록 생성.
    4개 변수 → 12개 가능 엣지."""
    return [(v1, v2) for v1 in variables for v2 in variables if v1 != v2]


def bitstring_to_dag(bitstring, edge_list):
    """12비트 문자열을 DAG로 변환.
    bitstring[i] == '1'이면 edge_list[i] 엣지 존재."""
    G = nx.DiGraph()
    variables = set()
    for e in edge_list:
        variables.add(e[0])
        variables.add(e[1])
    G.add_nodes_from(sorted(variables))

    for i, bit in enumerate(bitstring):
        if bit == "1":
            G.add_edge(edge_list[i][0], edge_list[i][1])
    return G


def is_dag(G):
    """그래프가 DAG(비순환 방향 그래프)인지 검증."""
    return nx.is_directed_acyclic_graph(G)


def dag_to_bitstring(G, edge_list):
    """DAG를 비트 문자열로 인코딩."""
    bits = []
    for src, dst in edge_list:
        bits.append("1" if G.has_edge(src, dst) else "0")
    return "".join(bits)


def get_ground_truth_dag(variables=None):
    """Sachs(2005) 논문의 정답 DAG에서 선택 변수에 해당하는 부분 추출.

    전체 정답 네트워크 중 Raf/Mek/Erk/Akt 관련:
    Raf→Mek, Mek→Erk, Erk→Akt
    """
    if variables is None:
        variables = get_variable_names(3)

    all_edges = [("Raf", "Mek"), ("Mek", "Erk"), ("Erk", "Akt")]
    var_set = set(variables)

    G = nx.DiGraph()
    G.add_nodes_from(variables)
    for src, dst in all_edges:
        if src in var_set and dst in var_set:
            G.add_edge(src, dst)
    return G


def enumerate_all_dags(variables):
    """주어진 변수들의 모든 유효 DAG를 열거.
    4개 변수 → 12개 가능 엣지 → 2^12 후보 중 비순환인 것만."""
    edge_list = get_edge_list(variables)
    n_edges = len(edge_list)
    valid_dags = []

    for i in range(2 ** n_edges):
        bitstring = format(i, f"0{n_edges}b")
        G = bitstring_to_dag(bitstring, edge_list)
        if is_dag(G):
            valid_dags.append((bitstring, G))

    return valid_dags, edge_list


def structural_hamming_distance(G_true, G_est):
    """두 DAG 사이의 SHD (Structural Hamming Distance).
    엣지 추가, 삭제, 방향 반전의 합."""
    true_edges = set(G_true.edges())
    est_edges = set(G_est.edges())

    # 정답에는 있는데 추정에 없는 엣지 (삭제)
    missing = true_edges - est_edges
    # 추정에는 있는데 정답에 없는 엣지 (추가)
    extra = est_edges - true_edges

    # 방향 반전: extra 중 역방향이 missing에 있는 경우
    reversed_edges = 0
    for src, dst in list(extra):
        if (dst, src) in missing:
            reversed_edges += 1

    # SHD = 누락 + 추가 - 반전(중복 카운트 보정)
    shd = len(missing) + len(extra) - reversed_edges
    return shd


def edge_metrics(G_true, G_est):
    """엣지 단위 Precision, Recall, F1."""
    true_edges = set(G_true.edges())
    est_edges = set(G_est.edges())

    tp = len(true_edges & est_edges)
    fp = len(est_edges - true_edges)
    fn = len(true_edges - est_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
