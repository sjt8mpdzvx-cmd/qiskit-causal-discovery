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


def _apply_pattern_phase(qc, qubits, bitstring, phase):
    """Apply a diagonal conditional phase to one computational-basis state."""
    from qiskit.circuit.library import PhaseGate

    if len(qubits) != len(bitstring):
        raise ValueError("bitstring length must match the number of qubits")
    if not qubits:
        raise ValueError("at least one qubit is required")

    flipped = []
    for qubit, bit in zip(qubits, bitstring):
        if bit not in {"0", "1"}:
            raise ValueError("bitstrings must contain only '0' and '1'")
        if bit == "0":
            qc.x(qubit)
            flipped.append(qubit)

    if len(qubits) == 1:
        qc.p(phase, qubits[0])
    else:
        target = qubits[-1]
        qc.append(PhaseGate(phase).control(len(qubits) - 1), list(qubits[:-1]) + [target])

    for qubit in reversed(flipped):
        qc.x(qubit)


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

    부분 위상 패널티는 순환(비유효) DAG의 위상을 바꿉니다. Diffuser와의
    간섭에 따라 유효 DAG 측정 확률이 달라질 수 있으나, 증가를 보장하지는 않습니다.

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

    # --- Part 2: 순환 DAG에 부분 위상 패널티 ---
    for target_bits in cyclic_bitstrings:
        _apply_pattern_phase(
            oracle,
            list(range(n_qubits)),
            target_bits,
            -penalty_angle,
        )

    return oracle


def run_grover_search_with_penalty(
    n_qubits, good_bitstrings, cyclic_bitstrings, n_iterations=None, shots=1024, penalty_angle=None
):
    """Penalty Oracle을 사용한 Grover 알고리즘 실행.

    좋은 DAG의 진폭을 증폭하고 순환(비유효) DAG의 위상을 별도로 바꿉니다.
    유효 DAG 측정 확률의 변화는 회로·반복 횟수에 따라 달라지므로 실험값으로 비교해야 합니다.

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


# ---------------------------------------------------------------------------
# In-circuit Score Evaluation Oracle — 전수조사 없는 Grover
# ---------------------------------------------------------------------------
# QFT 기반 Draper Adder로 로컬 점수를 ancilla 레지스터에 누적하고,
# threshold 비교로 "좋은 DAG"를 회로 내에서 판별한다.
# ---------------------------------------------------------------------------


def _apply_qft(qc, qubits):
    """Append a canonical no-swap QFT to an arbitrary qubit subset."""
    from qiskit.synthesis.qft import synth_qft_full

    qc.compose(synth_qft_full(len(qubits), do_swaps=False), qubits=list(qubits), inplace=True)


def _apply_iqft(qc, qubits):
    """Append the inverse of :func:`_apply_qft`."""
    from qiskit.synthesis.qft import synth_qft_full

    qc.compose(
        synth_qft_full(len(qubits), do_swaps=False).inverse(),
        qubits=list(qubits),
        inplace=True,
    )


def _add_qft_unconditional(qc, reg, value, n_bits):
    """QFT 도메인에서 레지스터에 정수 value를 무조건 가산."""
    for i in range(n_bits):
        # _apply_qft uses a no-swap QFT, whose physical qubit i carries the
        # 2**i Fourier weight.
        angle = 2 * math.pi * value / (2 ** (i + 1))
        if abs(angle) < 1e-12:
            continue
        qc.p(angle, reg[i])


def _ccp(qc, theta, c0, c1, target):
    """이중 제어 위상 게이트 CCP(θ): c0, c1 모두 |1⟩일 때 target에 P(θ) 적용.

    분해: CP(θ/2, c0, t) · CX(c1, c0) · CP(-θ/2, c0, t) · CX(c1, c0) · CP(θ/2, c1, t)
    """
    qc.cp(theta / 2, c0, target)
    qc.cx(c1, c0)
    qc.cp(-theta / 2, c0, target)
    qc.cx(c1, c0)
    qc.cp(theta / 2, c1, target)


def _controlled_add_qft(qc, controls, reg, value, n_bits):
    """QFT 도메인에서 제어 가산: controls가 모두 |1⟩일 때 reg += value.

    0-2개 제어 큐비트를 지원 (3변수: max 2, 4변수: max 3).
    3개 이상은 PhaseGate.control() 사용.
    """
    from qiskit.circuit.library import PhaseGate

    for i in range(n_bits):
        angle = 2 * math.pi * value / (2 ** (i + 1))
        if abs(angle) < 1e-12:
            continue

        nc = len(controls)
        if nc == 0:
            qc.p(angle, reg[i])
        elif nc == 1:
            qc.cp(angle, controls[0], reg[i])
        elif nc == 2:
            _ccp(qc, angle, controls[0], controls[1], reg[i])
        else:
            # 일반적인 다중 제어: Qiskit PhaseGate.control(n) 사용
            gate = PhaseGate(angle).control(nc)
            qc.append(gate, list(controls) + [reg[i]])


def _controlled_sub_qft(qc, controls, reg, value, n_bits):
    """QFT 도메인에서 제어 감산 (가산의 역연산: 음의 각도)."""
    from qiskit.circuit.library import PhaseGate

    for i in range(n_bits):
        angle = -2 * math.pi * value / (2 ** (i + 1))
        if abs(angle) < 1e-12:
            continue

        nc = len(controls)
        if nc == 0:
            qc.p(angle, reg[i])
        elif nc == 1:
            qc.cp(angle, controls[0], reg[i])
        elif nc == 2:
            _ccp(qc, angle, controls[0], controls[1], reg[i])
        else:
            gate = PhaseGate(angle).control(nc)
            qc.append(gate, list(controls) + [reg[i]])


def _quantize_local_scores(local_scores, variables, n_bits):
    """로컬 점수를 n_bits 정수로 양자화.

    Returns
    -------
    quantized : dict  — 양자화된 로컬 점수
    threshold_max : int — 가능한 최대 합
    threshold_min : int — 가능한 최소 합
    """
    all_scores = list(local_scores.values())
    min_s = min(all_scores)
    # 0 이상으로 이동
    shifted = {k: v - min_s for k, v in local_scores.items()}

    # 노드별 최대 → 합의 최대값
    max_per_node = {}
    for (node, _), val in shifted.items():
        max_per_node[node] = max(max_per_node.get(node, 0), val)
    max_possible_sum = sum(max_per_node[n] for n in variables)

    if max_possible_sum <= 0:
        scale = 1.0
    else:
        # 합이 2^(n_bits-1) - 1 이내에 들도록 (MSB를 부호 비트로 사용)
        scale = (2 ** (n_bits - 1) - 1) / max_possible_sum

    quantized = {k: max(0, int(round(v * scale))) for k, v in shifted.items()}

    q_max = sum(max(quantized[k] for k in quantized if k[0] == n) for n in variables)
    q_min = sum(min(quantized[k] for k in quantized if k[0] == n) for n in variables)

    return quantized, q_max, q_min


def build_oracle_incircuit(n_qubits, edge_list, variables, local_scores,
                           threshold_ratio=0.7, n_score_bits=8):
    """In-circuit Score Evaluation Oracle — 전수조사 없는 Grover Oracle.

    회로 내에서 BDeu 점수를 평가하여 좋은 DAG를 표시한다.
    Pre-computed good bitstrings가 **필요 없다**.

    구조:
    1. QFT(score_reg)
    2. 각 노드 × 각 부모 구성: 제어 QFT 가산으로 점수 누적
    3. iQFT(score_reg) → 점수가 계산 기저에 나타남
    4. threshold 감산 → MSB 확인 → flag 설정
    5. Z(flag) — 위상 반전
    6. 전부 역연산 (uncompute)

    큐비트 배치: [edge × n_qubits | score × n_score_bits | flag × 1]
    3변수: 6 + 8 + 1 = 15 큐비트 (Aer 시뮬레이터로 충분)
    """
    QuantumCircuit, _ = _load_qiskit()

    n_score = n_score_bits
    n_total = n_qubits + n_score + 1
    edge_q = list(range(n_qubits))
    score_q = list(range(n_qubits, n_qubits + n_score))
    flag_q = n_qubits + n_score

    quantized, q_max, q_min = _quantize_local_scores(local_scores, variables, n_score)
    threshold_int = int(q_min + threshold_ratio * (q_max - q_min))

    oracle = QuantumCircuit(n_total, name="InCircuitOracle")

    # ── COMPUTE: 로컬 점수 누적 ──
    _apply_qft(oracle, score_q)

    # 각 노드에 대해 수신 엣지 큐비트 기반 제어 가산
    _score_accumulation_ops = []  # uncompute 시 역순 재생용

    for node in variables:
        incoming = [
            (q_idx, src) for q_idx, (src, dst) in enumerate(edge_list) if dst == node
        ]
        n_inc = len(incoming)

        for mask in range(2 ** n_inc):
            parents = frozenset(
                src for bit, (q_idx, src) in enumerate(incoming) if mask & (1 << bit)
            )
            score_int = quantized.get((node, parents), 0)
            if score_int == 0:
                continue

            # X gates: 부모가 아닌 수신 엣지 → |0⟩을 |1⟩로 변환
            flip_indices = []
            for bit, (q_idx, src) in enumerate(incoming):
                if not (mask & (1 << bit)):
                    oracle.x(q_idx)
                    flip_indices.append(q_idx)

            controls = [q_idx for q_idx, _ in incoming]
            _controlled_add_qft(oracle, controls, score_q, score_int, n_score)

            # X gates 복원
            for q_idx in flip_indices:
                oracle.x(q_idx)

            _score_accumulation_ops.append((controls, score_int, flip_indices, incoming, mask))

    # ── iQFT → 점수가 계산 기저에 나타남 ──
    _apply_iqft(oracle, score_q)

    # ── COMPARE: threshold 감산 후 MSB 확인 ──
    _apply_qft(oracle, score_q)
    sub_val = (2 ** n_score - threshold_int) % (2 ** n_score)
    _add_qft_unconditional(oracle, score_q, sub_val, n_score)
    _apply_iqft(oracle, score_q)

    # MSB = 0 이면 score >= threshold
    oracle.x(score_q[-1])
    oracle.cx(score_q[-1], flag_q)
    oracle.x(score_q[-1])

    # ── PHASE FLIP ──
    oracle.z(flag_q)

    # ── UNCOMPUTE (역순) ──
    # flag 해제
    oracle.x(score_q[-1])
    oracle.cx(score_q[-1], flag_q)
    oracle.x(score_q[-1])

    # threshold 복원
    _apply_qft(oracle, score_q)
    _add_qft_unconditional(oracle, score_q, threshold_int, n_score)
    _apply_iqft(oracle, score_q)

    # 점수 누적 역연산
    _apply_qft(oracle, score_q)
    for controls, score_int, flip_indices, incoming, mask in reversed(_score_accumulation_ops):
        for q_idx in flip_indices:
            oracle.x(q_idx)
        _controlled_sub_qft(oracle, controls, score_q, score_int, n_score)
        for q_idx in flip_indices:
            oracle.x(q_idx)
    _apply_iqft(oracle, score_q)

    return oracle, n_total, flag_q


def build_diffuser_incircuit(n_qubits, n_total):
    """In-circuit oracle용 diffuser — 엣지 큐비트에만 작용."""
    QuantumCircuit, _ = _load_qiskit()
    diffuser = QuantumCircuit(n_total, name="Diffuser")

    edge_q = list(range(n_qubits))
    diffuser.h(edge_q)
    diffuser.x(edge_q)

    # Multi-controlled Z on edge qubits only
    diffuser.h(edge_q[-1])
    diffuser.mcx(edge_q[:-1], edge_q[-1])
    diffuser.h(edge_q[-1])

    diffuser.x(edge_q)
    diffuser.h(edge_q)

    return diffuser


def run_grover_incircuit(n_qubits, edge_list, variables, local_scores,
                         threshold_ratio=0.7, n_score_bits=8,
                         n_iterations=None, shots=1024):
    """In-circuit Score Evaluation Grover — 전수조사 완전 제거.

    Oracle이 회로 내에서 점수를 직접 평가하므로,
    사전에 좋은 DAG를 알 필요가 없다.

    고전적 사전 계산: O(|V| × 2^(|V|-1)) 로컬 점수만.
    """
    from .scoring import score_bitstring_from_local

    QuantumCircuit, AerSimulator = _load_qiskit()

    # 예상 good 비율로 반복 횟수 추정
    N = 2 ** n_qubits
    M_est = max(1, int(N * (1 - threshold_ratio)))
    if n_iterations is None:
        n_iterations = max(1, int(math.pi / 4 * math.sqrt(N / M_est)))

    start_time = time.time()

    oracle, n_total, flag_q = build_oracle_incircuit(
        n_qubits, edge_list, variables, local_scores,
        threshold_ratio, n_score_bits,
    )
    diffuser = build_diffuser_incircuit(n_qubits, n_total)

    qc = QuantumCircuit(n_total, n_qubits)  # 측정은 엣지 큐비트만

    # 엣지 큐비트만 초기 중첩
    qc.h(range(n_qubits))
    qc.barrier()

    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.barrier()
        qc.compose(diffuser, inplace=True)
        qc.barrier()

    qc.measure(range(n_qubits), range(n_qubits))

    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()

    elapsed = time.time() - start_time

    corrected = {bs[::-1]: cnt for bs, cnt in counts.items()}
    top_bitstring = max(corrected, key=corrected.get)

    # 결과 분석: 측정된 비트 문자열의 점수를 로컬 테이블에서 조회
    measured_scores = {
        bs: score_bitstring_from_local(bs, edge_list, variables, local_scores)
        for bs in corrected
    }
    # Threshold success must be evaluated in the same quantized score domain
    # used by the oracle.  Comparing a de-quantized floating-point score can
    # classify states differently around the rounding boundary.
    quantized, q_max, q_min = _quantize_local_scores(local_scores, variables, n_score_bits)
    threshold_int = int(q_min + threshold_ratio * (q_max - q_min))
    quantized_measured_scores = {
        bs: score_bitstring_from_local(bs, edge_list, variables, quantized)
        for bs in corrected
    }

    good_count = sum(
        cnt for bs, cnt in corrected.items()
        if quantized_measured_scores.get(bs, -np.inf) >= threshold_int
    )
    good_probability = good_count / shots

    return {
        "counts": corrected,
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
        "M": M_est,
        "incircuit_oracle": True,
        "local_decomposition": True,
        "n_total_qubits": n_total,
        "n_score_bits": n_score_bits,
        "threshold_ratio": threshold_ratio,
        "threshold_int": threshold_int,
    }
