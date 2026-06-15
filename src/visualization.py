"""DAG 시각화 및 결과 비교 차트."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

import networkx as nx
import numpy as np


def draw_dag(G, title="DAG", ax=None, highlight_edges=None, pos=None):
    """DAG를 시각적으로 그린다."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    if pos is None:
        pos = {
            "Raf": (0, 1),
            "Mek": (1, 1),
            "Erk": (2, 1),
            "Akt": (2, 0),
        }

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#4ECDC4", node_size=1500,
                           edgecolors="#2C3E50", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight="bold",
                            font_color="#2C3E50")

    # 엣지 그리기
    edge_colors = []
    for e in G.edges():
        if highlight_edges and e in highlight_edges:
            edge_colors.append("#E74C3C")  # 빨강: 정답과 다른 엣지
        else:
            edge_colors.append("#2C3E50")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=2, arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")
    return ax


def compare_dags(G_true, G_classical, G_grover, metrics_classical, metrics_grover):
    """정답, 고전, 양자 DAG 세 개를 나란히 비교."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    pos = {"Raf": (0, 1), "Mek": (1, 1), "Erk": (2, 1), "Akt": (2, 0)}

    draw_dag(G_true, "Ground Truth\n(Sachs 2005)", axes[0], pos=pos)

    # 고전 결과 - 정답과 다른 엣지 하이라이트
    true_edges = set(G_true.edges())
    cl_extra = set(G_classical.edges()) - true_edges
    draw_dag(G_classical,
             f"Classical Search\nSHD={metrics_classical['shd']}, F1={metrics_classical['f1']:.2f}",
             axes[1], highlight_edges=cl_extra, pos=pos)

    gr_extra = set(G_grover.edges()) - true_edges
    draw_dag(G_grover,
             f"Grover Search\nSHD={metrics_grover['shd']}, F1={metrics_grover['f1']:.2f}",
             axes[2], highlight_edges=gr_extra, pos=pos)

    plt.tight_layout()
    return fig


def plot_grover_counts(counts, good_bitstrings, top_n=20):
    """Grover 측정 결과 히스토그램."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # 상위 N개만 표시
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    bitstrings = [x[0] for x in sorted_counts]
    values = [x[1] for x in sorted_counts]

    colors = ["#E74C3C" if bs in good_bitstrings else "#BDC3C7" for bs in bitstrings]

    bars = ax.bar(range(len(bitstrings)), values, color=colors, edgecolor="#2C3E50")
    ax.set_xticks(range(len(bitstrings)))
    ax.set_xticklabels(bitstrings, rotation=90, fontsize=7)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title("Grover Measurement Results (red = target DAGs)", fontsize=14)

    plt.tight_layout()
    return fig


def plot_bic_distribution(scored_dags, best_classical_bs, best_grover_bs, top_n_highlight=10):
    """BDeu 점수 분포와 고전/양자 결과 표시."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scores = [s[2] for s in scored_dags]
    ax.hist(scores, bins=50, color="#BDC3C7", edgecolor="#7F8C8D", alpha=0.7)

    # 고전 최적
    cl_score = None
    for bs, dag, score in scored_dags:
        if bs == best_classical_bs:
            cl_score = score
            break
    if cl_score is not None:
        ax.axvline(cl_score, color="#2980B9", linewidth=2, linestyle="--",
                   label=f"Classical best (BDeu={cl_score:.1f})")

    # 양자 최적
    gr_score = None
    for bs, dag, score in scored_dags:
        if bs == best_grover_bs:
            gr_score = score
            break
    if gr_score is not None:
        ax.axvline(gr_score, color="#E74C3C", linewidth=2, linestyle="-.",
                   label=f"Grover best (BDeu={gr_score:.1f})")

    ax.set_xlabel("BDeu Score", fontsize=12)
    ax.set_ylabel("Number of DAGs", fontsize=12)
    ax.set_title("BDeu Score Distribution across Valid DAGs", fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    return fig


def plot_comparison_table(metrics_classical, metrics_grover, classical_result, grover_result):
    """고전 vs 양자 비교 표."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    table_data = [
        ["", "Classical\n(Exhaustive)", "Grover\n(Quantum)"],
        ["Best BDeu", f"{classical_result['best_bic']:.1f}",
         f"{grover_result['best_bic']:.1f}" if grover_result['best_bic'] else "N/A"],
        ["SHD", str(metrics_classical["shd"]), str(metrics_grover["shd"])],
        ["F1 Score", f"{metrics_classical['f1']:.3f}", f"{metrics_grover['f1']:.3f}"],
        ["Precision", f"{metrics_classical['precision']:.3f}", f"{metrics_grover['precision']:.3f}"],
        ["Recall", f"{metrics_classical['recall']:.3f}", f"{metrics_grover['recall']:.3f}"],
        ["Evaluations", str(classical_result["evaluations"]),
         f"~{grover_result['n_iterations']} iterations"],
        ["Time (s)", f"{classical_result['elapsed_time']:.3f}",
         f"{grover_result['elapsed_time']:.3f}"],
    ]

    table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # 헤더 스타일
    for j in range(3):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(table_data)):
        table[i, 0].set_facecolor("#ECF0F1")
        table[i, 0].set_text_props(fontweight="bold")

    ax.set_title("Classical vs Quantum: Comparison", fontsize=14,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    return fig
