"""QAOA 기반 인과 구조 탐색.

BDeu 점수를 cost Hamiltonian으로 인코딩하여,
고전적 전수조사 없이 좋은 DAG를 찾는 양자 근사 최적화 접근법.

Grover 방식과의 차이:
- Grover: 좋은 답을 미리 알아야 함 (Pre-computed Oracle)
- QAOA: 점수 함수만 있으면 됨 → Oracle 사전 계산 불필요의 방향성 제시
"""

import time
import math
import numpy as np


def _load_qiskit():
    """Qiskit is only required when the quantum experiment is executed."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    return QuantumCircuit, AerSimulator


def build_cost_operator(n_qubits, scored_dags, gamma):
    """BDeu 점수를 RZ 회전으로 인코딩하는 cost operator.

    Each bitstring's BDeu score is mapped to a phase rotation.
    Higher score -> larger phase -> QAOA favors it.

    For practical circuit construction with few qubits, we use
    diagonal unitary encoding: for each scored bitstring, apply
    a conditional phase proportional to its normalized score.

    Args:
        n_qubits: 큐비트 수
        scored_dags: (bitstring, dag, score) 리스트
        gamma: QAOA gamma 파라미터 (위상 스케일링)

    Returns:
        QuantumCircuit: cost operator 회로
    """
    QuantumCircuit, _ = _load_qiskit()

    # Normalize scores to [0, 1] range
    scores = [s for _, _, s in scored_dags]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score if max_score != min_score else 1.0

    qc = QuantumCircuit(n_qubits, name="Cost")

    for bitstring, _, score in scored_dags:
        # Normalized score: higher is better
        norm_score = (score - min_score) / score_range
        phase = gamma * norm_score * math.pi

        if abs(phase) < 1e-10:
            continue

        # Apply conditional phase: flip qubits where bit is '0',
        # apply multi-controlled phase, flip back
        flip_qubits = []
        for i, bit in enumerate(bitstring):
            if bit == "0":
                qc.x(i)
                flip_qubits.append(i)

        # Multi-controlled RZ for the phase
        if n_qubits == 1:
            qc.rz(2 * phase, 0)
        else:
            # Use multi-controlled phase gate
            # MCZ with phase = apply RZ on last qubit controlled by all others
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.rz(2 * phase, n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        for i in flip_qubits:
            qc.x(i)

    return qc


def build_mixer_operator(n_qubits, beta):
    """Standard QAOA mixer: RX rotations on all qubits.

    Args:
        n_qubits: 큐비트 수
        beta: QAOA beta 파라미터 (혼합 강도)

    Returns:
        QuantumCircuit: mixer operator 회로
    """
    QuantumCircuit, _ = _load_qiskit()
    qc = QuantumCircuit(n_qubits, name="Mixer")
    for i in range(n_qubits):
        qc.rx(2 * beta, i)
    return qc


def run_qaoa_search(n_qubits, scored_dags, p_layers=2, shots=4096, n_optimization_steps=20):
    """Run QAOA for causal structure search.

    BDeu 점수를 cost Hamiltonian으로 인코딩하고 QAOA를 실행하여
    높은 점수의 DAG를 찾습니다. 고전적 최적화 루프로 gamma/beta 파라미터를
    탐색합니다.

    Args:
        n_qubits: 큐비트 수 (= 엣지 후보 수)
        scored_dags: (bitstring, dag, score) 리스트 (BDeu 점수순 정렬)
        p_layers: QAOA 레이어 수 (깊이 파라미터)
        shots: 측정 횟수
        n_optimization_steps: 고전 최적화 반복 횟수 (grid search 해상도 결정)

    Returns:
        result dict — app.py 표시와 호환되는 형식
    """
    QuantumCircuit, AerSimulator = _load_qiskit()

    start_time = time.time()

    # Classical optimization of gamma and beta parameters
    # Use simple grid search for reproducibility (vs scipy optimizer)
    best_params = None
    best_expectation = -np.inf
    best_counts = None
    best_circuit = None

    # Grid search over gamma and beta for each layer
    # For simplicity with small problems, use coarse grid
    n_grid = max(3, int(math.sqrt(n_optimization_steps)))
    gamma_range = np.linspace(0.1, math.pi, n_grid)
    beta_range = np.linspace(0.1, math.pi / 2, n_grid)

    # Build score lookup
    score_lookup = {}
    for bs, _, sc in scored_dags:
        score_lookup[bs] = sc

    scores_list = [s for _, _, s in scored_dags]
    min_score = min(scores_list)
    max_score = max(scores_list)
    score_range = max_score - min_score if max_score != min_score else 1.0

    simulator = AerSimulator()
    eval_count = 0

    for gamma_val in gamma_range:
        for beta_val in beta_range:
            # Build QAOA circuit
            qc = QuantumCircuit(n_qubits, n_qubits)
            qc.h(range(n_qubits))  # Initial superposition

            for layer in range(p_layers):
                # Scale gamma/beta slightly per layer for expressiveness
                g = gamma_val * (1 + 0.1 * layer)
                b = beta_val * (1 - 0.05 * layer)
                qc.compose(build_cost_operator(n_qubits, scored_dags, g), inplace=True)
                qc.compose(build_mixer_operator(n_qubits, b), inplace=True)

            qc.measure(range(n_qubits), range(n_qubits))

            result = simulator.run(qc, shots=shots).result()
            counts = result.get_counts()

            # Correct bit order (Qiskit reverses qubit ordering)
            corrected = {}
            for bs, cnt in counts.items():
                corrected[bs[::-1]] = cnt

            # Compute expectation value of normalized score
            expectation = 0.0
            for bs, cnt in corrected.items():
                if bs in score_lookup:
                    norm_s = (score_lookup[bs] - min_score) / score_range
                else:
                    norm_s = 0.0
                expectation += norm_s * cnt / shots

            eval_count += 1

            if expectation > best_expectation:
                best_expectation = expectation
                best_params = (gamma_val, beta_val)
                best_counts = corrected
                best_circuit = qc

    elapsed = time.time() - start_time

    # Get top bitstring from best run
    top_bitstring = max(best_counts, key=best_counts.get)

    # Compute good probability (same definition as Grover: top-k DAGs)
    top_k = min(6, len(scored_dags))
    good_bitstrings = [scored_dags[i][0] for i in range(top_k)]
    good_count = sum(best_counts.get(bs, 0) for bs in good_bitstrings)
    good_probability = good_count / shots

    return {
        "counts": best_counts,
        "top_bitstring": top_bitstring,
        "n_qubits": n_qubits,
        "circuit": best_circuit,
        "circuit_depth": best_circuit.depth(),
        "gate_count": dict(best_circuit.count_ops()),
        "elapsed_time": elapsed,
        "good_probability": good_probability,
        "shots": shots,
        "N": 2 ** n_qubits,
        "M": top_k,
        "p_layers": p_layers,
        "best_gamma": best_params[0],
        "best_beta": best_params[1],
        "best_expectation": best_expectation,
        "optimization_evals": eval_count,
        "qaoa": True,
    }


def qaoa_causal_search(data, variables, scored_dags, edge_list, p_layers=2, shots=4096):
    """QAOA로 인과 구조 탐색 전체 파이프라인.

    Grover의 grover_causal_search에 대응하는 QAOA 버전입니다.

    Args:
        data: 데이터프레임
        variables: 변수 목록
        scored_dags: BDeu 점수로 정렬된 (bitstring, dag, score) 리스트
        edge_list: 엣지 목록
        p_layers: QAOA 레이어 수
        shots: 측정 횟수

    Returns:
        result dict
    """
    from .dag_utils import bitstring_to_dag

    n_qubits = len(edge_list)

    # QAOA 실행 (모든 scored_dags를 cost operator에 인코딩)
    qaoa_result = run_qaoa_search(n_qubits, scored_dags, p_layers=p_layers, shots=shots)

    # 최빈 결과를 DAG로 변환
    top_bitstring = qaoa_result["top_bitstring"]
    top_dag = bitstring_to_dag(top_bitstring, edge_list)

    # 해당 DAG의 BDeu 점수 찾기
    top_score = None
    for bs, dag, score in scored_dags:
        if bs == top_bitstring:
            top_score = score
            break

    qaoa_result["best_dag"] = top_dag
    qaoa_result["best_bitstring"] = top_bitstring
    qaoa_result["best_score"] = top_score
    qaoa_result["best_bic"] = top_score  # Backward-compatible alias.
    qaoa_result["good_bitstrings"] = [scored_dags[i][0] for i in range(min(6, len(scored_dags)))]

    return qaoa_result


# ---------------------------------------------------------------------------
# Local Score Decomposition — 전수조사 없는 QAOA
# ---------------------------------------------------------------------------


def build_cost_operator_local(n_qubits, edge_list, variables, local_scores, gamma):
    """로컬 점수 분해 기반 QAOA cost operator.

    핵심 차이 — build_cost_operator와 비교:
    ┌────────────────────┬──────────────────────────────────────┐
    │ build_cost_operator│ 전수조사된 scored_dags 필요 (O(2^|E|))│
    │ 이 함수            │ 로컬 점수만 필요 (O(|V|×2^(|V|-1)))  │
    └────────────────────┴──────────────────────────────────────┘

    BDeu 분해 성질: Score(G) = Σ_node LocalScore(node, Pa(node)).
    각 노드의 기여분은 해당 노드의 **수신 엣지 큐비트**만으로 결정되므로,
    전체 큐비트가 아닌 (|V|-1)개 큐비트에만 조건부 위상을 적용한다.

    회로 항 수: |V| × 2^(|V|-1)   (3변수: 12, 5변수: 80)
    최대 제어:  |V|-1 큐비트       (3변수: 2,  5변수: 4)
    """
    QuantumCircuit, _ = _load_qiskit()
    qc = QuantumCircuit(n_qubits, name="Cost_Local")

    # Normalize all local scores to [0, 1]
    all_scores = list(local_scores.values())
    min_s = min(all_scores)
    max_s = max(all_scores)
    score_range = max_s - min_s if max_s != min_s else 1.0

    for node in variables:
        # 이 노드로 들어오는 엣지에 해당하는 큐비트 인덱스
        incoming = [
            (q_idx, src)
            for q_idx, (src, dst) in enumerate(edge_list)
            if dst == node
        ]
        n_incoming = len(incoming)

        for mask in range(2 ** n_incoming):
            # mask로 부모 집합 결정
            parents = frozenset(
                src for bit, (q_idx, src) in enumerate(incoming) if mask & (1 << bit)
            )

            score = local_scores.get((node, parents), 0)
            norm_score = (score - min_s) / score_range
            phase = gamma * norm_score * math.pi

            if abs(phase) < 1e-10:
                continue

            # 부모가 아닌(엣지 없는) 큐비트에 X → 모든 수신 큐비트가 |1⟩이 되도록
            flip_qubits = []
            for bit, (q_idx, src) in enumerate(incoming):
                if not (mask & (1 << bit)):
                    qc.x(q_idx)
                    flip_qubits.append(q_idx)

            # 수신 큐비트 전체에 대한 다중 제어 RZ
            all_incoming_q = [q_idx for q_idx, _ in incoming]

            if len(all_incoming_q) == 0:
                # 수신 엣지 없음 (발생하지 않지만 안전 처리)
                qc.rz(2 * phase, 0)
            elif len(all_incoming_q) == 1:
                qc.rz(2 * phase, all_incoming_q[0])
            else:
                target = all_incoming_q[-1]
                controls = all_incoming_q[:-1]
                qc.h(target)
                qc.mcx(controls, target)
                qc.rz(2 * phase, target)
                qc.mcx(controls, target)
                qc.h(target)

            for q_idx in flip_qubits:
                qc.x(q_idx)

    return qc


def run_qaoa_local_search(
    n_qubits, edge_list, variables, local_scores,
    p_layers=2, shots=4096, n_optimization_steps=20,
):
    """로컬 점수 분해 기반 QAOA 실행 — 전수조사 불필요.

    기존 run_qaoa_search와의 차이:
    - scored_dags (전수조사 결과) 대신 local_scores (다항적 사전 계산) 사용
    - cost operator가 O(|V|×2^(|V|-1)) 항으로 구성
    - 측정 결과의 점수도 로컬 테이블에서 O(|V|) 룩업으로 계산
    """
    from .scoring import score_bitstring_from_local

    QuantumCircuit, AerSimulator = _load_qiskit()

    start_time = time.time()

    n_grid = max(3, int(math.sqrt(n_optimization_steps)))
    gamma_range = np.linspace(0.1, math.pi, n_grid)
    beta_range = np.linspace(0.1, math.pi / 2, n_grid)

    # 정규화를 위한 점수 범위 (로컬 점수에서 계산, 전수조사 불필요)
    max_possible = sum(
        max(v for k, v in local_scores.items() if k[0] == node)
        for node in variables
    )
    min_possible = sum(
        min(v for k, v in local_scores.items() if k[0] == node)
        for node in variables
    )
    score_range = max_possible - min_possible if max_possible != min_possible else 1.0

    simulator = AerSimulator()
    best_params = None
    best_expectation = -np.inf
    best_counts = None
    best_circuit = None
    eval_count = 0

    for gamma_val in gamma_range:
        for beta_val in beta_range:
            qc = QuantumCircuit(n_qubits, n_qubits)
            qc.h(range(n_qubits))

            for layer in range(p_layers):
                g = gamma_val * (1 + 0.1 * layer)
                b = beta_val * (1 - 0.05 * layer)
                qc.compose(
                    build_cost_operator_local(n_qubits, edge_list, variables, local_scores, g),
                    inplace=True,
                )
                qc.compose(build_mixer_operator(n_qubits, b), inplace=True)

            qc.measure(range(n_qubits), range(n_qubits))

            result = simulator.run(qc, shots=shots).result()
            counts = result.get_counts()
            corrected = {bs[::-1]: cnt for bs, cnt in counts.items()}

            # 기대값 계산 — 로컬 점수 룩업 (데이터 접근 없음)
            expectation = 0.0
            for bs, cnt in corrected.items():
                s = score_bitstring_from_local(bs, edge_list, variables, local_scores)
                norm_s = (s - min_possible) / score_range
                expectation += norm_s * cnt / shots

            eval_count += 1

            if expectation > best_expectation:
                best_expectation = expectation
                best_params = (gamma_val, beta_val)
                best_counts = corrected
                best_circuit = qc

    elapsed = time.time() - start_time
    top_bitstring = max(best_counts, key=best_counts.get)

    # 상위 k개 측정 결과 (점수순)
    measured_scores = [
        (bs, score_bitstring_from_local(bs, edge_list, variables, local_scores))
        for bs in best_counts
    ]
    measured_scores.sort(key=lambda x: x[1], reverse=True)
    top_k = min(6, len(measured_scores))
    good_bitstrings = [x[0] for x in measured_scores[:top_k]]
    good_count = sum(best_counts.get(bs, 0) for bs in good_bitstrings)
    good_probability = good_count / shots

    n_local_terms = sum(
        2 ** len([1 for _, (src, dst) in enumerate(edge_list) if dst == node])
        for node in variables
    )

    return {
        "counts": best_counts,
        "top_bitstring": top_bitstring,
        "n_qubits": n_qubits,
        "circuit": best_circuit,
        "circuit_depth": best_circuit.depth(),
        "gate_count": dict(best_circuit.count_ops()),
        "elapsed_time": elapsed,
        "good_probability": good_probability,
        "shots": shots,
        "N": 2 ** n_qubits,
        "M": top_k,
        "p_layers": p_layers,
        "best_gamma": best_params[0],
        "best_beta": best_params[1],
        "best_expectation": best_expectation,
        "optimization_evals": eval_count,
        "qaoa": True,
        "local_decomposition": True,
        "n_local_terms": n_local_terms,
    }
