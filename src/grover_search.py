"""Grover 알고리즘 기반 인과 구조 탐색.

전략:
1. 고전적으로 각 DAG의 BDeu 점수를 미리 계산 (Oracle 구성에 필요)
2. BDeu 점수 상위 k개를 "좋은 DAG"로 정의
3. Grover Oracle: 좋은 DAG 비트 문자열에 위상 반전 적용
4. Grover 반복으로 좋은 DAG를 높은 확률로 측정
5. 측정 결과의 DAG를 최종 답으로 반환
"""

import time
import math
import numpy as np


def _load_qiskit():
    """Qiskit is only required when the quantum experiment is executed."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    return QuantumCircuit, AerSimulator


def build_oracle(n_qubits, good_bitstrings):
    """좋은 DAG 비트 문자열들에 위상 반전을 적용하는 Oracle 회로 구성.

    |x⟩ → -|x⟩  if x in good_bitstrings
    |x⟩ →  |x⟩  otherwise
    """
    QuantumCircuit, _ = _load_qiskit()
    oracle = QuantumCircuit(n_qubits, name="Oracle")

    for target_bits in good_bitstrings:
        # 0인 비트에 X 게이트 적용 (|0⟩→|1⟩로 바꿔서 multi-controlled-Z 적용)
        flip_qubits = []
        for i, bit in enumerate(target_bits):
            if bit == "0":
                oracle.x(i)
                flip_qubits.append(i)

        # Multi-controlled Z gate: 모든 큐비트가 |1⟩일 때 위상 반전
        if n_qubits == 1:
            oracle.z(0)
        else:
            # MCZ = H on last qubit, MCX, H on last qubit
            oracle.h(n_qubits - 1)
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oracle.h(n_qubits - 1)

        # X 게이트 원복
        for i in flip_qubits:
            oracle.x(i)

    return oracle


def build_diffuser(n_qubits):
    """Grover diffusion operator (진폭 증폭).

    D = 2|s⟩⟨s| - I, where |s⟩ = H^n|0⟩
    """
    QuantumCircuit, _ = _load_qiskit()
    diffuser = QuantumCircuit(n_qubits, name="Diffuser")

    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))

    # Multi-controlled Z
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)

    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    return diffuser


def run_grover_search(n_qubits, good_bitstrings, n_iterations=None, shots=1024):
    """Grover 알고리즘 실행.

    Args:
        n_qubits: 큐비트 수 (= 엣지 후보 수)
        good_bitstrings: 좋은 DAG의 비트 문자열 목록
        n_iterations: Grover 반복 횟수 (None이면 최적값 자동 계산)
        shots: 측정 횟수

    Returns:
        result dict with counts, circuit, top_result, etc.
    """
    QuantumCircuit, AerSimulator = _load_qiskit()

    N = 2 ** n_qubits
    M = len(good_bitstrings)

    if n_iterations is None:
        if M > 0:
            n_iterations = max(1, int(math.pi / 4 * math.sqrt(N / M)))
        else:
            n_iterations = 1

    start_time = time.time()

    # 회로 구성
    qc = QuantumCircuit(n_qubits, n_qubits)

    # 1. 초기 균등 중첩
    qc.h(range(n_qubits))
    qc.barrier()

    # 2. Grover 반복
    oracle = build_oracle(n_qubits, good_bitstrings)
    diffuser = build_diffuser(n_qubits)

    for i in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.barrier()
        qc.compose(diffuser, inplace=True)
        qc.barrier()

    # 3. 측정
    qc.measure(range(n_qubits), range(n_qubits))

    # 4. 시뮬레이터 실행
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()

    elapsed = time.time() - start_time

    # Qiskit은 비트 순서를 반전시키므로 보정
    corrected_counts = {}
    for bitstr, count in counts.items():
        corrected_counts[bitstr[::-1]] = count

    # 최빈 결과
    top_bitstring = max(corrected_counts, key=corrected_counts.get)

    # 좋은 DAG가 측정된 확률
    good_count = sum(corrected_counts.get(bs, 0) for bs in good_bitstrings)
    good_probability = good_count / shots

    return {
        "counts": corrected_counts,
        "top_bitstring": top_bitstring,
        "n_iterations": n_iterations,
        "n_qubits": n_qubits,
        "circuit": qc,
        "circuit_depth": qc.depth(),
        "gate_count": dict(qc.count_ops()),
        "elapsed_time": elapsed,
        "good_probability": good_probability,
        "shots": shots,
        "N": N,
        "M": M,
    }


def grover_causal_search(data, variables, scored_dags, edge_list, top_k=10, shots=2048):
    """Grover로 인과 구조 탐색 전체 파이프라인.

    Args:
        data: 데이터프레임
        variables: 변수 목록
        scored_dags: BDeu 점수로 정렬된 (bitstring, dag, score) 리스트
        edge_list: 엣지 목록
        top_k: 상위 k개를 "좋은 DAG"로 정의
        shots: 측정 횟수

    Returns:
        result dict
    """
    from .dag_utils import bitstring_to_dag

    n_qubits = len(edge_list)

    # 상위 k개의 좋은 DAG 비트 문자열
    good_bitstrings = [scored_dags[i][0] for i in range(min(top_k, len(scored_dags)))]

    # Grover 실행
    grover_result = run_grover_search(n_qubits, good_bitstrings, shots=shots)

    # 최빈 결과를 DAG로 변환
    top_bitstring = grover_result["top_bitstring"]
    top_dag = bitstring_to_dag(top_bitstring, edge_list)

    # 해당 DAG의 BDeu 점수 찾기
    top_score = None
    for bs, dag, score in scored_dags:
        if bs == top_bitstring:
            top_score = score
            break

    grover_result["best_dag"] = top_dag
    grover_result["best_bitstring"] = top_bitstring
    grover_result["best_score"] = top_score
    grover_result["best_bic"] = top_score  # Backward-compatible alias.
    grover_result["good_bitstrings"] = good_bitstrings
    grover_result["top_k"] = top_k

    return grover_result


# ---------------------------------------------------------------------------
# Penalty Oracle: 좋은 DAG에 위상 반전 + 순환 DAG에 부분 위상 패널티
# ---------------------------------------------------------------------------


def build_penalty_oracle(n_qubits, good_bitstrings, cyclic_bitstrings, penalty_angle=None):
    """Penalty Oracle: 좋은 DAG를 마킹하고 순환 DAG를 억제하는 Oracle.

    동작:
    1. good_bitstrings에 대해 완전한 위상 반전 (π) — Grover 마킹
    2. cyclic_bitstrings에 대해 부분 위상 패널티 (기본 π/4) — 진폭 억제

    부분 위상 패널티는 순환(비유효) DAG의 진폭을 줄여서
    측정 시 유효한 DAG가 선택될 확률을 높입니다.

    Args:
        n_qubits: 큐비트 수
        good_bitstrings: 좋은 DAG 비트 문자열 목록 (완전 위상 반전)
        cyclic_bitstrings: 순환 DAG 비트 문자열 목록 (부분 패널티)
        penalty_angle: 패널티 회전 각도 (기본 π/4). 값이 클수록 강한 억제.

    Returns:
        QuantumCircuit: Penalty Oracle 회로
    """
    if penalty_angle is None:
        penalty_angle = math.pi / 4

    QuantumCircuit, _ = _load_qiskit()
    oracle = QuantumCircuit(n_qubits, name="PenaltyOracle")

    # --- Part 1: 좋은 DAG에 완전 위상 반전 (기존 Oracle과 동일) ---
    for target_bits in good_bitstrings:
        flip_qubits = []
        for i, bit in enumerate(target_bits):
            if bit == "0":
                oracle.x(i)
                flip_qubits.append(i)

        if n_qubits == 1:
            oracle.z(0)
        else:
            oracle.h(n_qubits - 1)
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oracle.h(n_qubits - 1)

        for i in flip_qubits:
            oracle.x(i)

    # --- Part 2: 순환 DAG에 부분 위상 패널티 (RZ 회전) ---
    for target_bits in cyclic_bitstrings:
        flip_qubits = []
        for i, bit in enumerate(target_bits):
            if bit == "0":
                oracle.x(i)
                flip_qubits.append(i)

        # Multi-controlled RZ: 해당 비트 문자열에만 부분 위상 적용
        if n_qubits == 1:
            oracle.rz(penalty_angle, 0)
        else:
            oracle.h(n_qubits - 1)
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oracle.rz(penalty_angle, n_qubits - 1)
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oracle.h(n_qubits - 1)

        for i in flip_qubits:
            oracle.x(i)

    return oracle


def run_grover_search_with_penalty(
    n_qubits, good_bitstrings, cyclic_bitstrings, n_iterations=None, shots=1024, penalty_angle=None
):
    """Penalty Oracle을 사용한 Grover 알고리즘 실행.

    좋은 DAG의 진폭을 증폭하면서 동시에 순환(비유효) DAG의 진폭을 억제합니다.
    이를 통해 유효 DAG 측정 확률이 기본 Grover보다 향상될 수 있습니다.

    Args:
        n_qubits: 큐비트 수 (= 엣지 후보 수)
        good_bitstrings: 좋은 DAG의 비트 문자열 목록
        cyclic_bitstrings: 순환 DAG의 비트 문자열 목록
        n_iterations: Grover 반복 횟수 (None이면 최적값 자동 계산)
        shots: 측정 횟수
        penalty_angle: 순환 DAG 패널티 각도 (기본 π/4)

    Returns:
        result dict — run_grover_search와 동일한 형식 + penalty_oracle: True
    """
    QuantumCircuit, AerSimulator = _load_qiskit()

    N = 2 ** n_qubits
    M = len(good_bitstrings)

    if n_iterations is None:
        if M > 0:
            n_iterations = max(1, int(math.pi / 4 * math.sqrt(N / M)))
        else:
            n_iterations = 1

    start_time = time.time()

    # 회로 구성
    qc = QuantumCircuit(n_qubits, n_qubits)

    # 1. 초기 균등 중첩
    qc.h(range(n_qubits))
    qc.barrier()

    # 2. Grover 반복 (Penalty Oracle + Diffuser)
    oracle = build_penalty_oracle(n_qubits, good_bitstrings, cyclic_bitstrings, penalty_angle)
    diffuser = build_diffuser(n_qubits)

    for i in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.barrier()
        qc.compose(diffuser, inplace=True)
        qc.barrier()

    # 3. 측정
    qc.measure(range(n_qubits), range(n_qubits))

    # 4. 시뮬레이터 실행
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()

    elapsed = time.time() - start_time

    # Qiskit은 비트 순서를 반전시키므로 보정
    corrected_counts = {}
    for bitstr, count in counts.items():
        corrected_counts[bitstr[::-1]] = count

    # 최빈 결과
    top_bitstring = max(corrected_counts, key=corrected_counts.get)

    # 좋은 DAG가 측정된 확률
    good_count = sum(corrected_counts.get(bs, 0) for bs in good_bitstrings)
    good_probability = good_count / shots

    return {
        "counts": corrected_counts,
        "top_bitstring": top_bitstring,
        "n_iterations": n_iterations,
        "n_qubits": n_qubits,
        "circuit": qc,
        "circuit_depth": qc.depth(),
        "gate_count": dict(qc.count_ops()),
        "elapsed_time": elapsed,
        "good_probability": good_probability,
        "shots": shots,
        "N": N,
        "M": M,
        "penalty_oracle": True,
    }
