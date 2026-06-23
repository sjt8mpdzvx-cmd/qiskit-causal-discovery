"""
Quantum Causal Discovery Lab

데이터에서 인과 구조(DAG)를 찾고, Grover 탐색으로 좋은 구조의 측정 확률을
증폭한 뒤, 발견된 구조를 이용해 개입 타겟 후보를 비교하는 Streamlit 앱.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "qiskit_causal_discovery_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "qiskit_causal_discovery_cache"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": ["AppleGothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafbfc",
    "axes.edgecolor": "#e2e8f0",
    "axes.grid": True,
    "grid.color": "#f1f5f9",
    "grid.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "axes.labelsize": 10,
    "axes.labelcolor": "#475569",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.shadow": False,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#e2e8f0",
    "legend.fontsize": 9,
})

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
sys.path.insert(0, APP_DIR)

from src.dag_utils import (  # noqa: E402
    bitstring_to_dag,
    edge_metrics,
    enumerate_all_dags,
    structural_hamming_distance,
)
from src.grover_search import run_grover_search, run_grover_search_with_penalty  # noqa: E402
from src.scoring import score_all_dags, score_all_dags_bge  # noqa: E402
from src.qaoa_search import run_qaoa_search  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    file: str
    description: str
    default_vars: tuple[str, ...]
    ground_truth: tuple[tuple[str, str], ...]
    outcome_hint: str | None
    story: str
    outcome_higher_is_better: bool | None = None  # True=높을수록 좋음, False=높을수록 나쁨


DATASETS: dict[str, DatasetSpec] = {
    "Sachs protein signaling": DatasetSpec(
        file="sachs_raw.csv",
        description="T세포 단일세포 단백질 신호 데이터. Raf-Mek-Erk MAPK 경로를 중심으로 분석합니다.",
        default_vars=("Raf", "Mek", "Erk", "Akt"),
        ground_truth=(("Raf", "Mek"), ("Mek", "Erk"), ("Erk", "Akt")),
        outcome_hint="Erk",
        story="암세포에서 특정 단백질 신호가 과활성화되었을 때, 어느 상위 단백질에 개입해야 하류 신호를 줄일 수 있는지 찾는 문제로 해석할 수 있습니다.",
        outcome_higher_is_better=False,
    ),
    "Asia lung diagnosis": DatasetSpec(
        file="asia.csv",
        description="폐질환 진단 베이지안 네트워크 벤치마크. 흡연, 기관지염, 폐암 등의 관계를 다룹니다.",
        default_vars=("smoke", "bronc", "lung", "dysp"),
        ground_truth=(
            ("smoke", "bronc"),
            ("smoke", "lung"),
            ("asia", "tub"),
            ("tub", "either"),
            ("lung", "either"),
            ("either", "xray"),
            ("either", "dysp"),
            ("bronc", "dysp"),
        ),
        outcome_hint="dysp",
        story="증상 변수에 영향을 주는 원인을 구조적으로 찾는 진단 보조 문제로 볼 수 있습니다.",
        outcome_higher_is_better=False,
    ),
    "Sprinkler weather": DatasetSpec(
        file="sprinkler.csv",
        description="날씨-스프링클러-잔디 젖음 관계를 담은 교육용 인과 네트워크입니다.",
        default_vars=("Cloudy", "Sprinkler", "Rain", "Wet_Grass"),
        ground_truth=(
            ("Cloudy", "Sprinkler"),
            ("Cloudy", "Rain"),
            ("Sprinkler", "Wet_Grass"),
            ("Rain", "Wet_Grass"),
        ),
        outcome_hint="Wet_Grass",
        story="상관관계만으로는 비와 스프링클러의 방향을 구분하기 어렵다는 점을 보여주는 작은 예제입니다.",
        outcome_higher_is_better=False,
    ),
    "Alarm ICU monitoring": DatasetSpec(
        file="alarm.csv",
        description="ICU 환자 모니터링 베이지안 네트워크에서 추출한 이산 데이터입니다.",
        default_vars=("HYPOVOLEMIA", "LVEDVOLUME", "STROKEVOLUME", "CVP"),
        ground_truth=(),
        outcome_hint=None,
        story="의료 모니터링 변수 사이의 의존 구조를 작은 부분 문제로 잘라 탐색합니다.",
    ),
    "Auto MPG (자동차 연비)": DatasetSpec(
        file="auto_mpg.csv",
        description="1970-82년 자동차 392대의 엔진/차체 사양과 연비(MPG) 데이터입니다. 배기량, 마력, 무게가 연비에 미치는 인과 관계를 분석합니다.",
        default_vars=("Displacement", "Horsepower", "Weight", "MPG"),
        ground_truth=(
            ("Displacement", "Horsepower"),
            ("Displacement", "Weight"),
            ("Horsepower", "MPG"),
            ("Weight", "MPG"),
        ),
        outcome_hint="MPG",
        story="자동차 제조사가 연비를 개선하려 할 때, 엔진 배기량을 줄여야 하는지 차체 무게를 줄여야 하는지 — 인과 구조를 통해 가장 효과적인 개입 지점을 찾는 문제입니다.",
        outcome_higher_is_better=True,
    ),
    "Framingham Heart Study (심장병)": DatasetSpec(
        file="framingham_heart.csv",
        description="70년 이상 추적된 Framingham 심장 연구 데이터. 흡연, 콜레스테롤, 혈압이 심장병 위험에 미치는 인과 경로를 분석합니다.",
        default_vars=("CigsPerDay", "Cholesterol", "SysBP", "HeartDisease"),
        ground_truth=(
            ("CigsPerDay", "Cholesterol"),
            ("CigsPerDay", "SysBP"),
            ("Cholesterol", "HeartDisease"),
            ("SysBP", "HeartDisease"),
        ),
        outcome_hint="HeartDisease",
        story="공중보건 기관이 심장병 발생률을 줄이려 할 때, 금연 캠페인·콜레스테롤 약·혈압 약 중 어디에 자원을 집중해야 하는지 인과 분석으로 판단하는 문제입니다.",
        outcome_higher_is_better=False,
    ),
}


OUTCOME_DIRECTION_DEFAULTS: dict[str, bool] = {
    # Higher is better
    "MPG": True,
    "StudyTime": True,
    # Higher is usually worse in the built-in stories
    "Erk": False,
    "Akt": False,
    "Mek": False,
    "Raf": False,
    "dysp": False,
    "bronc": False,
    "lung": False,
    "smoke": False,
    "Wet_Grass": False,
    "HeartDisease": False,
    "CigsPerDay": False,
    "Cholesterol": False,
    "SysBP": False,
    "Displacement": False,
    "Horsepower": False,
    "Weight": False,
}


def infer_outcome_direction(outcome: str, spec: DatasetSpec | None) -> tuple[bool, str]:
    """Return default desirability direction and a short confidence label."""
    if outcome in OUTCOME_DIRECTION_DEFAULTS:
        return OUTCOME_DIRECTION_DEFAULTS[outcome], "known"
    if spec is not None and spec.outcome_hint == outcome and spec.outcome_higher_is_better is not None:
        return spec.outcome_higher_is_better, "dataset"
    return False, "unknown"


def available_datasets() -> dict[str, DatasetSpec]:
    return {
        name: spec
        for name, spec in DATASETS.items()
        if os.path.exists(os.path.join(DATA_DIR, spec.file))
    }


def format_edges(edges: Iterable[tuple[str, str]]) -> str:
    edge_list = list(edges)
    if not edge_list:
        return "엣지 없음"
    return ", ".join(f"{src}->{dst}" for src, dst in edge_list)


def parse_edge_text(text: str) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "->" not in line:
            continue
        src, dst = line.split("->", 1)
        src, dst = src.strip(), dst.strip()
        if src and dst and src != dst:
            edges.append((src, dst))
    return edges


def build_ground_truth(variables: list[str], edges: Iterable[tuple[str, str]]) -> nx.DiGraph:
    var_set = set(variables)
    graph = nx.DiGraph()
    graph.add_nodes_from(variables)
    for src, dst in edges:
        if src in var_set and dst in var_set:
            graph.add_edge(src, dst)
    return graph


@st.cache_data(show_spinner=False)
def load_csv(file_name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, file_name))


def read_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes with user-facing validation errors."""
    if not csv_bytes:
        raise ValueError("CSV 파일이 비어 있습니다.")
    try:
        raw = pd.read_csv(io.BytesIO(csv_bytes))
    except UnicodeDecodeError:
        try:
            raw = pd.read_csv(io.BytesIO(csv_bytes), encoding="cp949")
        except Exception as exc:
            raise ValueError(f"CSV 인코딩을 읽을 수 없습니다: {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("CSV 파일이 비어 있습니다.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV 형식을 파싱할 수 없습니다: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"CSV를 읽을 수 없습니다: {exc}") from exc

    raw.columns = [str(col).strip() for col in raw.columns]
    if raw.empty:
        raise ValueError("CSV에 데이터 행이 없습니다.")
    if len(raw.columns) < 2:
        raise ValueError("CSV에는 최소 2개 이상의 열이 필요합니다.")
    if any(col == "" for col in raw.columns):
        raise ValueError("빈 컬럼명이 있습니다. CSV 헤더를 확인하세요.")
    duplicated = raw.columns[raw.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"중복 컬럼명이 있습니다: {', '.join(map(str, duplicated))}")
    return raw


def validate_selected_variables(raw: pd.DataFrame, variables: list[str]) -> None:
    if len(variables) < 2:
        raise ValueError("분석하려면 변수를 2개 이상 선택해야 합니다.")
    if len(variables) > 4:
        raise ValueError("현재 Grover 시뮬레이션은 변수 4개까지 안정적으로 지원합니다.")
    missing = [var for var in variables if var not in raw.columns]
    if missing:
        raise ValueError(f"CSV에 없는 변수가 선택되었습니다: {', '.join(missing)}")


def discretize_for_bdeu(raw: pd.DataFrame, variables: list[str], auto_discretize: bool) -> pd.DataFrame:
    validate_selected_variables(raw, variables)
    data = raw[variables].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=variables)
    if len(data) < 5:
        raise ValueError("선택한 변수들에서 결측/무한값을 제거한 뒤 분석 가능한 행이 5개 미만입니다.")

    for col in variables:
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            unique_count = series.nunique(dropna=True)
            if unique_count == 0:
                raise ValueError(f"{col} 변수에 분석 가능한 값이 없습니다.")
            if auto_discretize and unique_count > 8:
                ranked = series.rank(method="first")
                data[col] = pd.qcut(ranked, q=3, labels=False, duplicates="drop")
            else:
                data[col] = series
        else:
            data[col] = series.astype("string").astype("category").cat.codes

    data = data.dropna().reset_index(drop=True)
    if data.empty:
        raise ValueError("전처리 후 남은 데이터가 없습니다.")
    if all(data[col].nunique(dropna=True) < 2 for col in variables):
        raise ValueError("선택한 모든 변수가 상수입니다. 변동이 있는 변수를 선택하세요.")
    return data


@st.cache_data(show_spinner=False)
def score_from_csv_bytes(
    csv_bytes: bytes,
    variables_tuple: tuple[str, ...],
    ess: int,
    auto_discretize: bool,
) -> tuple[pd.DataFrame, list[tuple[str, nx.DiGraph]], list[tuple[str, str]], list[tuple[str, nx.DiGraph, float]], float]:
    raw = read_csv_bytes(csv_bytes)
    variables = list(variables_tuple)
    data = discretize_for_bdeu(raw, variables, auto_discretize)
    start = time.time()
    valid_dags, edge_list = enumerate_all_dags(variables)
    scored = score_all_dags(data, valid_dags, variables, equivalent_sample_size=ess)
    elapsed = time.time() - start
    return data, valid_dags, edge_list, scored, elapsed


@st.cache_data(show_spinner=False)
def score_from_csv_bytes_bge(
    csv_bytes: bytes,
    variables_tuple: tuple[str, ...],
) -> tuple[pd.DataFrame, list[tuple[str, nx.DiGraph]], list[tuple[str, str]], list[tuple[str, nx.DiGraph, float]], float]:
    """BGe 점수 — 연속 데이터를 이산화 없이 직접 평가."""
    raw = read_csv_bytes(csv_bytes)
    variables = list(variables_tuple)
    validate_selected_variables(raw, variables)
    data = raw[variables].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=variables)
    if len(data) < 5:
        raise ValueError("선택한 변수들에서 결측/무한값을 제거한 뒤 분석 가능한 행이 5개 미만입니다.")
    # BGe는 연속 데이터에 직접 적용 — 숫자가 아닌 열만 코드로 변환
    for col in variables:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].astype("string").astype("category").cat.codes.astype(float)
    data = data.reset_index(drop=True)
    start = time.time()
    valid_dags, edge_list = enumerate_all_dags(variables)
    scored = score_all_dags_bge(data, valid_dags, variables)
    elapsed = time.time() - start
    return data, valid_dags, edge_list, scored, elapsed


def known_layout(variables: list[str]) -> dict[str, tuple[float, float]] | None:
    protein_pos = {
        "Raf": (0.0, 0.45),
        "Mek": (1.0, 0.45),
        "Erk": (2.0, 0.45),
        "Akt": (1.5, -0.35),
        "PKA": (-0.2, -0.35),
        "PKC": (-0.2, 1.05),
    }
    sprinkler_pos = {
        "Cloudy": (0.0, 1.0),
        "Sprinkler": (-0.9, 0.1),
        "Rain": (0.9, 0.1),
        "Wet_Grass": (0.0, -0.75),
    }
    asia_pos = {
        "asia": (-1.2, 1.0),
        "tub": (-1.2, 0.25),
        "smoke": (0.2, 1.0),
        "lung": (0.0, 0.25),
        "bronc": (1.0, 0.25),
        "either": (-0.45, -0.45),
        "xray": (-1.0, -1.2),
        "dysp": (0.45, -1.2),
    }
    for layout in (protein_pos, sprinkler_pos, asia_pos):
        if all(var in layout for var in variables):
            return {var: layout[var] for var in variables}
    return None


def draw_dag(
    graph: nx.DiGraph,
    variables: list[str],
    title: str,
    reference: nx.DiGraph | None = None,
    subtitle: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#ffffff")
    
    G = graph.copy()
    G.add_nodes_from(variables)
    
    # 1. 지그재그 계층 레이아웃 (엣지 겹침 방지)
    try:
        layers = list(nx.topological_generations(G))
        pos = {}
        for x, nodes in enumerate(layers):
            nodes = sorted(nodes)
            for y, node in enumerate(nodes):
                # x축 간격은 넓게, y축은 중앙 정렬 + 미세한 지그재그(x%2 * 0.2)
                y_coord = -(y - (len(nodes)-1)/2.0) * 1.5
                y_jitter = 0.2 if x % 2 == 0 else -0.2
                pos[node] = (x * 2.5, y_coord + y_jitter)
    except Exception:
        pos = nx.spring_layout(G, k=1.5, seed=42)

    missing = [n for n in variables if n not in pos]
    for i, n in enumerate(missing):
        pos[n] = (-1.5, -(i - (len(missing)-1)/2.0) * 1.5)

    # 2. 노드 그리기
    node_colors = []
    for node in G.nodes():
        if G.out_degree(node) > 0 and G.in_degree(node) == 0:
            node_colors.append("#eef2ff") # Source
        elif G.out_degree(node) == 0:
            node_colors.append("#fef3c7") # Sink
        else:
            node_colors.append("#f0fdf4") # Mid

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=2200,
        edgecolors="#334155", linewidths=1.2, alpha=1.0
    )
    
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=10, font_weight="bold", font_color="#1e293b"
    )

    # 3. 곡선 화살표 그리기 (FancyArrowPatch)
    import matplotlib.patches as patches
    ref_edges = set(reference.edges()) if reference is not None else set()
    
    for u, v in G.edges():
        # 기본/정답/오답 색상 및 굵기
        color = "#6366f1"
        width = 1.6
        if reference is not None:
            if (u, v) in ref_edges:
                color = "#059669"
                width = 2.0
            else:
                color = "#ef4444"
                width = 1.5

        # 화살표 크기를 대폭 줄임 (mutation_scale=10, head_length=1.2)
        arrow = patches.FancyArrowPatch(
            pos[u], pos[v],
            arrowstyle='-|>,head_length=1.5,head_width=0.8',
            connectionstyle="arc3,rad=0.12", # 자연스러운 곡선
            color=color,
            linewidth=width,
            mutation_scale=10, # 화살표 헤드 크기 결정
            shrinkA=15, # 시작 지점 여백
            shrinkB=15, # 끝 지점 여백
            zorder=1,
            alpha=0.8
        )
        ax.add_patch(arrow)

    # 4. 축 범위 및 제목
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    if subtitle:
        ax.text(
            0.5, -0.02, subtitle, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#475569",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#e2e8f0", alpha=0.8)
        )
    
    # 화살표 색상 범례
    if reference is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#059669", label="Correct"),
            Patch(facecolor="#ef4444", label="Wrong / Extra"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9,
                  edgecolor="#e2e8f0", fancybox=True)

    ax.axis("off")
    ax.margins(0.15)
    fig.tight_layout()
    return fig


def plot_correlation(data: pd.DataFrame) -> plt.Figure:
    numeric = data.apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(4.9, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Custom purple-orange diverging colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "custom_div", ["#6366f1", "#e0e7ff", "#ffffff", "#fed7aa", "#ea580c"]
    )

    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontweight="medium")
    ax.set_yticklabels(corr.columns, fontweight="medium")
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            if np.isfinite(value):
                text_color = "#ffffff" if abs(value) > 0.6 else "#334155"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_color)
    ax.set_title("Correlation Matrix", fontsize=12, fontweight="bold", color="#0f172a")

    # Rounded edges on spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.04)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_score_distribution(scored: list[tuple[str, nx.DiGraph, float]], good_bitstrings: list[str]) -> plt.Figure:
    scores = np.array([item[2] for item in scored])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    n, bins, patches = ax.hist(
        scores, bins=min(40, max(8, len(scores) // 4)),
        color="#c7d2fe", edgecolor="#818cf8", linewidth=0.8, alpha=0.85,
    )

    best_score = scored[0][2]
    threshold_score = scored[min(len(scored), len(good_bitstrings)) - 1][2]
    ax.axvline(best_score, color="#6366f1", linewidth=2.5, label="Best BDeu", zorder=5)
    ax.axvline(threshold_score, color="#f59e0b", linewidth=2.0, linestyle="--", label="Oracle threshold", zorder=5)

    # Shade the "good" region
    ax.axvspan(threshold_score, scores.max() + 1, alpha=0.06, color="#6366f1")

    ax.set_xlabel("BDeu Score (higher is better)")
    ax.set_ylabel("DAG count")
    ax.set_title("Score Distribution of Valid DAGs", fontsize=12, fontweight="bold", color="#0f172a")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_grover_counts(
    counts: dict[str, int],
    good_bitstrings: list[str],
    valid_bitstrings: set[str],
    top_n: int = 18,
) -> plt.Figure:
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    bitstrings = [item[0] for item in ranked]
    values = [item[1] for item in ranked]
    colors = []
    for bitstring in bitstrings:
        if bitstring in good_bitstrings:
            colors.append("#6366f1")
        elif bitstring in valid_bitstrings:
            colors.append("#38bdf8")
        else:
            colors.append("#e2e8f0")
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    bars = ax.bar(
        range(len(bitstrings)), values,
        color=colors, edgecolor="#ffffff", linewidth=1.2,
        width=0.75, zorder=3,
    )
    # Add value labels on top of bars for Oracle targets
    for i, (bar, bs) in enumerate(zip(bars, bitstrings)):
        if bs in good_bitstrings:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(values[i]), ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#4338ca",
            )

    ax.set_xticks(range(len(bitstrings)))
    ax.set_xticklabels(bitstrings, rotation=50, ha="right", fontsize=8, fontfamily="monospace")
    ax.set_ylabel("Measurement Count")
    ax.set_title("Grover Measurement Distribution", fontsize=12, fontweight="bold", color="#0f172a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#6366f1", label="Oracle target"),
        Patch(facecolor="#38bdf8", label="Valid DAG"),
        Patch(facecolor="#e2e8f0", label="Invalid"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def candidate_table(
    scored: list[tuple[str, nx.DiGraph, float]],
    reference: nx.DiGraph | None,
    top_n: int = 12,
) -> pd.DataFrame:
    best = scored[0][2]
    rows = []
    for rank, (bitstring, graph, score) in enumerate(scored[:top_n], start=1):
        row = {
            "rank": rank,
            "bitstring": bitstring,
            "BDeu": round(score, 3),
            "gap": round(score - best, 3),
            "edges": format_edges(graph.edges()),
        }
        if reference is not None and len(reference.edges()) > 0:
            metrics = edge_metrics(reference, graph)
            row["SHD"] = structural_hamming_distance(reference, graph)
            row["F1"] = round(metrics["f1"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def graph_metrics(reference: nx.DiGraph | None, graph: nx.DiGraph) -> dict[str, float | int] | None:
    if reference is None or len(reference.edges()) == 0:
        return None
    metrics = edge_metrics(reference, graph)
    metrics["shd"] = structural_hamming_distance(reference, graph)
    return metrics


def grover_iteration_count(n_qubits: int, target_count: int) -> int:
    if target_count <= 0:
        return 1
    return max(1, int(math.pi / 4 * math.sqrt((2**n_qubits) / target_count)))


def enrich_grover_result(
    result: dict,
    valid_bitstrings: set[str],
    scored: list[tuple[str, nx.DiGraph, float]],
) -> dict:
    score_lookup = {bitstring: score for bitstring, _, score in scored}
    rank_lookup = {bitstring: rank for rank, (bitstring, _, _) in enumerate(scored, start=1)}
    valid_counts = {
        bitstring: count
        for bitstring, count in result["counts"].items()
        if bitstring in valid_bitstrings
    }

    # Score-weighted selection: 측정 횟수 × BDeu 점수로 최적 DAG 선택
    if valid_counts:
        max_score = max(score_lookup.values())
        min_score = min(score_lookup.values())
        score_range = max_score - min_score if max_score != min_score else 1.0

        best_combined = None
        best_combined_val = -float("inf")
        for bs, cnt in valid_counts.items():
            s = score_lookup.get(bs, min_score)
            norm_score = (s - min_score) / score_range
            norm_count = cnt / result["shots"]
            combined = 0.6 * norm_score + 0.4 * norm_count
            if combined > best_combined_val:
                best_combined_val = combined
                best_combined = bs

        selected = best_combined or max(valid_counts, key=valid_counts.get)
    else:
        selected = result["top_bitstring"]

    # Top-5 measured valid DAGs by score
    valid_measured = sorted(
        [(bs, cnt, score_lookup.get(bs, 0), rank_lookup.get(bs, 999)) for bs, cnt in valid_counts.items()],
        key=lambda x: x[2], reverse=True,
    )[:5]

    return {
        **result,
        "selected_bitstring": selected,
        "selected_score": score_lookup.get(selected),
        "selected_rank": rank_lookup.get(selected),
        "selected_is_valid": selected in valid_bitstrings,
        "raw_top_is_valid": result["top_bitstring"] in valid_bitstrings,
        "valid_probability": sum(valid_counts.values()) / result["shots"],
        "valid_measured_top5": valid_measured,
        "unique_valid_measured": len(valid_counts),
        "unique_total_measured": len(result["counts"]),
    }


def check_qiskit() -> tuple[bool, str]:
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
    except Exception as exc:  # pragma: no cover - shown to user in the app
        return False, str(exc)
    return True, ""


def prompt_cache_key(prefix: str, prompt: str) -> str:
    """Use the full prompt as the cache identity so changed results are not reused."""
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def compact_groq_error(errors: list[str] | str) -> str:
    """Convert verbose provider errors into an actionable short message."""
    text = " | ".join(errors) if isinstance(errors, list) else str(errors)
    lowered = text.lower()
    if "429" in text or "rate limit" in lowered or "quota" in lowered:
        return (
            "AI 해석 생성 실패: Groq API 사용량 제한에 걸렸습니다. "
            "잠시 뒤 다시 시도하거나 Groq 콘솔의 rate limit/프로젝트 설정을 확인하세요."
        )
    if "404" in text and ("not found" in lowered or "model" in lowered):
        return (
            "AI 해석 생성 실패: 현재 Groq API key에서 요청한 모델을 사용할 수 없습니다. "
            "Groq 콘솔에서 모델 권한을 확인하세요."
        )
    if "401" in text or "403" in text or "api key" in lowered or "permission" in lowered or "unauthorized" in lowered:
        return f"AI 해석 생성 실패: {text}"
    return f"AI 해석 생성 실패: {text}"


def append_local_fallback(message: str, fallback: str | None) -> str:
    if not fallback:
        return message
    return f"{message}\n\nGroq 없이 표시하는 로컬 요약:\n{fallback}"


def groq_error_message(http_error: urllib.error.HTTPError) -> str:
    body = http_error.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(body)
        detail = parsed.get("error", {})
        if isinstance(detail, dict) and detail.get("message"):
            return f"{http_error.code} {detail['message']}"
    except Exception:
        pass
    return f"{http_error.code} {body.strip() or http_error.reason}"


def groq_state_key(api_key: str, cache_key: str) -> str:
    api_key_digest = hashlib.sha1(api_key.encode("utf-8")).hexdigest()[:8]
    return f"groq_{api_key_digest}_{cache_key}"


def call_groq(api_key: str, prompt: str, cache_key: str, fallback: str | None = None) -> str:
    """Call Groq Chat Completions and cache the generated interpretation."""
    state_key = groq_state_key(api_key, cache_key)
    cached = st.session_state.get(state_key)
    if cached and not str(cached).startswith("AI 해석 생성 실패"):
        return cached

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful Korean causal inference assistant. "
                    "Explain only what the provided DAG, BDeu score, and intervention table support."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_completion_tokens": 1500,
    }
    try:
        request = urllib.request.Request(
            GROQ_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        result = data["choices"][0]["message"]["content"].strip()
        if result:
            st.session_state[state_key] = result
            return result
        result = "AI 해석 생성 실패: Groq 응답이 비어 있습니다."
    except urllib.error.HTTPError as exc:
        result = compact_groq_error(groq_error_message(exc))
    except urllib.error.URLError as exc:
        result = compact_groq_error(f"network error: {exc.reason}")
    except TimeoutError:
        result = compact_groq_error("network timeout")
    except Exception as exc:
        result = compact_groq_error(str(exc))
    result = append_local_fallback(result, fallback)
    st.session_state[state_key] = result
    return result


def render_ai_box(content: str):
    """AI 해석 결과를 스타일링된 박스로 렌더링한다."""
    is_error = content.startswith("AI 해석 생성 실패")
    box_bg = "#fef2f2" if is_error else "linear-gradient(135deg, #eef2ff 0%, #f0fdf4 100%)"
    box_border = "#fecaca" if is_error else "#c7d2fe"
    title_color = "#b91c1c" if is_error else "#6366f1"
    title = "AI Interpretation Error" if is_error else "AI Interpretation (Groq)"
    safe_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="
            background: {box_bg};
            border: 1px solid {box_border};
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin: 1rem 0;
        ">
            <div style="font-size: 0.78rem; font-weight: 700; color: {title_color}; text-transform: uppercase;
                        letter-spacing: 0.05em; margin-bottom: 0.5rem;">{title}</div>
            <div style="color: #1e293b; font-size: 0.92rem; line-height: 1.7;">{safe_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def coverage_confidence(coverage: float) -> tuple[str, str, str]:
    """Map backdoor coverage to a compact user-facing confidence badge."""
    if coverage >= 0.8:
        return "신뢰도 높음", "#dcfce7", "#166534"
    if coverage >= 0.5:
        return "제한적 해석", "#fef3c7", "#92400e"
    return "데이터 부족", "#fee2e2", "#991b1b"


def intervention_mean(
    data: pd.DataFrame,
    dag: nx.DiGraph,
    target: str,
    outcome: str,
    target_value,
) -> tuple[float, float, str]:
    baseline = float(data[outcome].mean())
    if target == outcome:
        return baseline, 1.0, "outcome"
    if not nx.has_path(dag, target, outcome):
        return baseline, 1.0, "no directed path"

    parents = list(dag.predecessors(target))
    subset = data[data[target] == target_value]
    if len(parents) == 0:
        if len(subset) == 0:
            return baseline, 0.0, "no matching rows"
        return float(subset[outcome].mean()), len(subset) / len(data), "direct adjustment"

    grouped = data.groupby(parents, dropna=False).size() / len(data)
    weighted_sum = 0.0
    covered_weight = 0.0
    for combo, probability in grouped.items():
        combo_values = combo if isinstance(combo, tuple) else (combo,)
        mask = data[target] == target_value
        for parent, value in zip(parents, combo_values):
            mask &= data[parent] == value
        cell = data[mask]
        if len(cell) == 0:
            continue
        weighted_sum += float(probability) * float(cell[outcome].mean())
        covered_weight += float(probability)

    if covered_weight == 0:
        if len(subset) == 0:
            return baseline, 0.0, "fallback baseline"
        return float(subset[outcome].mean()), len(subset) / len(data), "fallback conditional"
    return weighted_sum / covered_weight, covered_weight, f"adjusted by {', '.join(parents)}"


def intervention_table(
    data: pd.DataFrame,
    dag: nx.DiGraph,
    variables: list[str],
    outcome: str,
    higher_is_better: bool = False,
) -> pd.DataFrame:
    rows = []
    for target in variables:
        if target == outcome:
            continue
        values = sorted(data[target].dropna().unique())
        if len(values) == 0:
            continue
        low_value, high_value = values[0], values[-1]
        y_low, low_coverage, low_note = intervention_mean(data, dag, target, outcome, low_value)
        y_high, high_coverage, high_note = intervention_mean(data, dag, target, outcome, high_value)
        effect = y_high - y_low
        coverage = min(low_coverage, high_coverage)

        if abs(effect) < 1e-9:
            action = "개입 우선순위 낮음"
        elif higher_is_better:
            # outcome을 높이고 싶다 (MPG, FinalGrade)
            action = f"{target} 활성화" if effect > 0 else f"{target} 억제"
        else:
            # outcome을 줄이고 싶다 (HeartDisease, Erk)
            action = f"{target} 억제" if effect > 0 else f"{target} 활성화"

        coverage_note = ""
        if coverage < 0.5:
            coverage_note = " (coverage 부족)"
        reliability, _, _ = coverage_confidence(float(coverage))

        rows.append(
            {
                "target": target,
                "low_value": low_value,
                "high_value": high_value,
                f"E[{outcome}|do(low)]": round(y_low, 4),
                f"E[{outcome}|do(high)]": round(y_high, 4),
                "effect_high_minus_low": round(effect, 4),
                "recommended_action": action + coverage_note,
                "coverage": round(coverage, 3),
                "reliability": reliability,
                "method": low_note if low_note == high_note else f"{low_note}; {high_note}",
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("effect_high_minus_low", key=lambda col: col.abs(), ascending=False).reset_index(drop=True)


def has_actionable_intervention(table: pd.DataFrame, eps: float = 1e-9) -> bool:
    return not table.empty and float(table["effect_high_minus_low"].abs().max()) > eps


def plot_interventions(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    values = table["effect_high_minus_low"].to_numpy()
    targets = table["target"].to_list()
    colors = ["#ef4444" if v > 0 else "#6366f1" if v < 0 else "#cbd5e1" for v in values]

    bars = ax.barh(
        targets, values, color=colors, edgecolor="#ffffff",
        height=0.55, linewidth=1.5, zorder=3,
    )
    ax.axvline(0, color="#334155", linewidth=1.2, zorder=2)

    # Value labels
    for bar, val in zip(bars, values):
        label_x = val + (max(abs(values)) * 0.03 if val >= 0 else -max(abs(values)) * 0.03)
        ha = "left" if val >= 0 else "right"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", ha=ha, va="center", fontsize=9, fontweight="bold", color="#334155")

    ax.set_xlabel("E[outcome | do(high)] − E[outcome | do(low)]")
    ax.set_title("Intervention Effect Estimation", fontsize=12, fontweight="bold", color="#0f172a")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ef4444", label="Increases outcome"),
        Patch(facecolor="#6366f1", label="Decreases outcome"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_shd_explanation(reference: nx.DiGraph, estimated: nx.DiGraph, variables: list[str]) -> plt.Figure:
    """SHD 구성 요소를 시각적으로 분해해서 보여주는 차트."""
    true_edges = set(reference.edges())
    est_edges = set(estimated.edges())

    tp = true_edges & est_edges
    missing = true_edges - est_edges
    extra = est_edges - true_edges
    reversed_set = set()
    for s, d in list(extra):
        if (d, s) in missing:
            reversed_set.add((s, d))

    pure_missing = len(missing) - len(reversed_set)
    pure_extra = len(extra) - len(reversed_set)

    categories = ["Correct", "Missing", "Extra", "Reversed"]
    values = [len(tp), pure_missing, pure_extra, len(reversed_set)]
    colors = ["#059669", "#f59e0b", "#ef4444", "#8b5cf6"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.patch.set_facecolor("#ffffff")

    # Left: donut chart
    ax = axes[0]
    ax.set_facecolor("#ffffff")
    total = sum(values)
    if total == 0:
        values = [1]
        colors = ["#e2e8f0"]
        categories = ["No edges"]
    wedges, texts, autotexts = ax.pie(
        values, labels=categories, colors=colors, autopct=lambda p: f"{int(round(p * total / 100))}" if total > 0 else "",
        startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.4, edgecolor="#ffffff", linewidth=2),
        textprops={"fontsize": 9, "fontweight": "bold"},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color("#ffffff")
    ax.set_title("Edge Classification", fontsize=12, fontweight="bold", color="#0f172a", pad=12)

    # Right: metric bars
    ax2 = axes[1]
    ax2.set_facecolor("#fafbfc")
    metrics = edge_metrics(reference, estimated)
    shd = structural_hamming_distance(reference, estimated)
    metric_names = ["Precision", "Recall", "F1 Score"]
    metric_vals = [metrics["precision"], metrics["recall"], metrics["f1"]]
    bar_colors = ["#6366f1", "#38bdf8", "#059669"]

    bars = ax2.barh(metric_names, metric_vals, color=bar_colors, height=0.5, edgecolor="#ffffff", linewidth=1.5, zorder=3)
    ax2.set_xlim(0, 1.15)
    for bar, val in zip(bars, metric_vals):
        ax2.text(val + 0.03, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=11, fontweight="bold", color="#334155")
    ax2.axvline(1.0, color="#e2e8f0", linewidth=1, linestyle="--", zorder=1)
    ax2.set_title(f"Performance Metrics  (SHD = {shd})", fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()

    fig.tight_layout(w_pad=3)
    return fig


def plot_complexity_comparison(n_edges: int, top_k: int) -> plt.Figure:
    """Grover O(sqrt(N)) vs Classical O(N) 복잡도 비교 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    fig.patch.set_facecolor("#ffffff")

    # Left: scaling curve
    ax = axes[0]
    ax.set_facecolor("#fafbfc")
    edge_range = np.arange(2, 21)
    classical = 2.0 ** edge_range
    quantum = np.sqrt(classical / max(top_k, 1)) * (math.pi / 4)

    ax.semilogy(edge_range, classical, "-o", color="#ef4444", linewidth=2.2, markersize=5, label="Classical O(N)", zorder=3)
    ax.semilogy(edge_range, quantum, "-s", color="#6366f1", linewidth=2.2, markersize=5, label=f"Grover O(√(N/{top_k}))", zorder=3)
    ax.axvline(n_edges, color="#f59e0b", linewidth=2, linestyle="--", alpha=0.7, label=f"Current ({n_edges} edges)")
    ax.fill_between(edge_range, quantum, classical, alpha=0.06, color="#6366f1")
    ax.set_xlabel("Number of edge candidates")
    ax.set_ylabel("Evaluations (log scale)")
    ax.set_title("Search Complexity Scaling", fontsize=12, fontweight="bold", color="#0f172a")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: current speedup bar
    ax2 = axes[1]
    ax2.set_facecolor("#fafbfc")
    N = 2 ** n_edges
    classical_evals = N
    grover_evals = max(1, int(math.pi / 4 * math.sqrt(N / max(top_k, 1))))
    speedup = classical_evals / grover_evals if grover_evals > 0 else 1

    bar_data = [("Classical\n(exhaustive)", classical_evals, "#ef4444"), ("Grover\n(quantum)", grover_evals, "#6366f1")]
    for i, (label, val, color) in enumerate(bar_data):
        ax2.bar(i, val, color=color, width=0.55, edgecolor="#ffffff", linewidth=2, zorder=3)
        ax2.text(i, val + classical_evals * 0.02, f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#334155")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([d[0] for d in bar_data], fontsize=9)
    ax2.set_ylabel("Oracle evaluations")
    ax2.set_title(f"Current Setting: {speedup:.1f}x Speedup", fontsize=12, fontweight="bold", color="#0f172a")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=3)
    return fig


def plot_score_landscape(scored: list[tuple[str, nx.DiGraph, float]], top_k: int) -> plt.Figure:
    """DAG score landscape를 rank-score 곡선으로 시각화."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    scores = [s for _, _, s in scored]
    ranks = list(range(1, len(scores) + 1))

    ax.fill_between(ranks[:top_k], scores[:top_k], min(scores), alpha=0.15, color="#6366f1", zorder=2)
    ax.plot(ranks, scores, "-", color="#94a3b8", linewidth=1.5, zorder=2)
    ax.scatter(ranks[:top_k], scores[:top_k], color="#6366f1", s=30, zorder=4, label=f"Oracle targets (top {top_k})")
    if len(ranks) > top_k:
        ax.scatter(ranks[top_k:], scores[top_k:], color="#e2e8f0", s=8, zorder=3, alpha=0.6)
    ax.axhline(scores[min(top_k, len(scores)) - 1], color="#f59e0b", linewidth=1.5, linestyle="--", alpha=0.7, label="Oracle threshold")

    ax.set_xlabel("DAG rank")
    ax.set_ylabel("BDeu Score")
    ax.set_title("Score Landscape: BDeu by Rank", fontsize=12, fontweight="bold", color="#0f172a")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_radar_comparison(
    classical_metrics: dict | None,
    grover_metrics: dict | None,
    grover_result: dict | None,
    n_edges: int,
    top_k: int,
) -> plt.Figure:
    """Classical vs Quantum 레이더 차트."""
    from matplotlib.patches import FancyBboxPatch

    labels = ["Precision", "Recall", "F1", "BDeu Rank\n(inv.)", "Search\nEfficiency"]
    N_ax = len(labels)
    angles = np.linspace(0, 2 * np.pi, N_ax, endpoint=False).tolist()
    angles += angles[:1]

    classical_vals = [0.0] * N_ax
    grover_vals = [0.0] * N_ax

    if classical_metrics:
        classical_vals[0] = classical_metrics.get("precision", 0)
        classical_vals[1] = classical_metrics.get("recall", 0)
        classical_vals[2] = classical_metrics.get("f1", 0)
        classical_vals[3] = 1.0  # rank 1 always
        classical_vals[4] = 0.3  # exhaustive = low efficiency

    if grover_metrics and grover_result:
        grover_vals[0] = grover_metrics.get("precision", 0)
        grover_vals[1] = grover_metrics.get("recall", 0)
        grover_vals[2] = grover_metrics.get("f1", 0)
        rank = grover_result.get("selected_rank", 999)
        grover_vals[3] = max(0, 1.0 - (rank - 1) / max(20, rank))
        total_N = 2 ** n_edges
        grover_iters = max(1, int(math.pi / 4 * math.sqrt(total_N / max(top_k, 1))))
        grover_vals[4] = min(1.0, 1.0 - grover_iters / total_N) if total_N > 0 else 0

    classical_vals += classical_vals[:1]
    grover_vals += grover_vals[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.plot(angles, classical_vals, "o-", color="#ef4444", linewidth=2.2, markersize=7, label="Classical", zorder=3)
    ax.fill(angles, classical_vals, color="#ef4444", alpha=0.08)
    ax.plot(angles, grover_vals, "s-", color="#6366f1", linewidth=2.2, markersize=7, label="Grover", zorder=3)
    ax.fill(angles, grover_vals, color="#6366f1", alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5, color="#94a3b8")
    ax.spines["polar"].set_color("#e2e8f0")
    ax.grid(color="#e2e8f0", linewidth=0.8)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), framealpha=0.9, fontsize=10)
    ax.set_title("Classical vs Quantum", fontsize=14, fontweight="bold", color="#0f172a", pad=24, y=1.08)

    fig.tight_layout()
    return fig


def plot_gauge(value: float, label: str, max_val: float = 1.0, color_thresholds: list | None = None) -> plt.Figure:
    """반원 게이지 차트."""
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if color_thresholds is None:
        color_thresholds = [(0.33, "#ef4444"), (0.66, "#f59e0b"), (1.01, "#059669")]

    theta_bg = np.linspace(np.pi, 0, 100)
    ax.plot(np.cos(theta_bg) * 1.0, np.sin(theta_bg) * 1.0, color="#e2e8f0", linewidth=14, solid_capstyle="round", zorder=1)

    ratio = min(value / max_val, 1.0) if max_val > 0 else 0
    color = "#94a3b8"
    for threshold, c in color_thresholds:
        if ratio <= threshold:
            color = c
            break

    theta_fill = np.linspace(np.pi, np.pi - ratio * np.pi, 100)
    ax.plot(np.cos(theta_fill) * 1.0, np.sin(theta_fill) * 1.0, color=color, linewidth=14, solid_capstyle="round", zorder=2)

    display = f"{value:.2f}" if max_val <= 1 else f"{value:.0f}"
    ax.text(0, 0.15, display, ha="center", va="center", fontsize=22, fontweight="bold", color="#0f172a")
    ax.text(0, -0.15, label, ha="center", va="center", fontsize=9.5, fontweight="medium", color="#64748b")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.4, 1.2)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def plot_amplification_waterfall(grover_result: dict, n_edges: int, top_k: int) -> plt.Figure:
    """Uniform → Oracle → Valid 후처리 단계별 확률 변화 워터폴 차트."""
    fig, ax = plt.subplots(figsize=(8, 4.0))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    uniform_p = top_k / (2 ** n_edges)
    oracle_p = grover_result["good_probability"]
    valid_p = grover_result["valid_probability"]
    selected_rank = grover_result.get("selected_rank", None)
    selected_in_top = selected_rank is not None and selected_rank <= top_k

    stages = [
        ("Uniform\nbaseline", uniform_p, "#94a3b8"),
        ("After\nGrover", oracle_p, "#6366f1"),
        ("Valid DAG\nfilter", valid_p, "#38bdf8"),
        ("Score-weighted\nselection", 1.0 if selected_in_top else 0.5, "#059669"),
    ]

    bars = []
    for i, (label, val, color) in enumerate(stages):
        bar = ax.bar(i, val, color=color, width=0.55, edgecolor="#ffffff", linewidth=2, zorder=3)
        bars.append(bar)
        ax.text(i, val + 0.02, f"{val*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#334155")
        if i > 0:
            prev_val = stages[i - 1][1]
            if prev_val > 0:
                mult = val / prev_val
                mid_y = min(val, prev_val) + abs(val - prev_val) / 2
                arrow_color = "#059669" if val > prev_val else "#ef4444"
                ax.annotate(
                    f"{mult:.1f}x", xy=(i - 0.5, mid_y), fontsize=9, fontweight="bold",
                    color=arrow_color, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor=arrow_color, alpha=0.9),
                )

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s[0] for s in stages], fontsize=9.5, fontweight="medium")
    ax.set_ylabel("Probability")
    ax.set_title("Grover Pipeline: Stage-by-Stage Probability Amplification", fontsize=12, fontweight="bold", color="#0f172a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(s[1] for s in stages) * 1.25)
    fig.tight_layout()
    return fig


def generate_interpretation(
    has_gt: bool,
    best_metrics: dict | None,
    grover_result: dict | None,
    grover_metrics: dict | None,
    n_edges: int,
    top_k: int,
    scored: list,
    variables: list[str],
    dataset_name: str,
) -> list[tuple[str, str, str]]:
    """정량 결과로부터 해석 bullet 목록 생성. (icon_color, title, body)."""
    insights = []

    # 1. Structure recovery quality
    if has_gt and best_metrics:
        f1 = best_metrics["f1"]
        shd = best_metrics["shd"]
        if f1 >= 0.9:
            insights.append(("#059669", "구조 복원 우수",
                f"고전 전수조사가 정답 DAG를 F1={f1:.2f}, SHD={shd}로 거의 완벽하게 복원했습니다. "
                f"BDeu 점수가 데이터에 내재된 인과 방향성을 잘 포착한 것입니다."))
        elif f1 >= 0.5:
            insights.append(("#f59e0b", "구조 부분 복원",
                f"고전 전수조사 1위 DAG의 F1={f1:.2f}, SHD={shd}입니다. "
                f"일부 엣지의 방향이나 존재 여부에서 정답과 차이가 있어, 데이터 크기나 ESS 조정이 필요할 수 있습니다."))
        else:
            insights.append(("#ef4444", "구조 복원 어려움",
                f"F1={f1:.2f}, SHD={shd}로 정답 구조와 상당한 차이가 있습니다. "
                f"변수 간 인과 신호가 약하거나, 이산화 방식이 정보를 손실시켰을 가능성이 있습니다."))

    # 2. Score landscape
    if len(scored) > 1:
        gap = scored[0][2] - scored[1][2]
        if abs(gap) < 0.5:
            insights.append(("#f59e0b", "상위 DAG 간 점수 접전",
                f"1위와 2위 BDeu 점수 차이가 {abs(gap):.2f}로 매우 작습니다. "
                f"데이터가 여러 인과 구조를 비슷하게 지지하고 있어, 단일 구조보다 상위 k개의 앙상블이 더 신뢰할 수 있습니다."))
        else:
            insights.append(("#059669", "1위 DAG 우위 명확",
                f"1위와 2위 BDeu 점수 차이가 {abs(gap):.2f}로, 최적 구조가 뚜렷하게 구별됩니다. "
                f"Grover Oracle이 이 구조를 타겟으로 삼을 때 증폭 효과가 극대화됩니다."))

    # 3. Grover performance
    if grover_result:
        uniform_p = top_k / (2 ** n_edges)
        amp = grover_result["good_probability"] / uniform_p if uniform_p > 0 else 0

        if amp >= 5:
            insights.append(("#059669", f"Grover 증폭 {amp:.1f}배 달성",
                f"Oracle 타겟의 측정 확률이 uniform baseline {uniform_p*100:.2f}%에서 "
                f"{grover_result['good_probability']*100:.1f}%로 {amp:.1f}배 증폭되었습니다. "
                f"Grover 알고리즘이 탐색 공간에서 좋은 해를 효과적으로 집중시킨 것입니다."))
        elif amp >= 2:
            insights.append(("#f59e0b", f"Grover 증폭 {amp:.1f}배",
                f"Oracle 타겟 확률이 {uniform_p*100:.2f}%에서 {grover_result['good_probability']*100:.1f}%로 증가했습니다. "
                f"유의미한 증폭이지만, shots나 반복 횟수를 늘리면 더 향상될 수 있습니다."))
        else:
            insights.append(("#ef4444", "Grover 증폭 미미",
                f"증폭 배수가 {amp:.1f}x로 이론적 기대보다 낮습니다. "
                f"Oracle 타겟 수(top-k={top_k})를 조정하거나 shots를 늘려 보세요."))

        # Grover vs Classical structure comparison
        if has_gt and grover_metrics and best_metrics:
            g_f1 = grover_metrics["f1"]
            c_f1 = best_metrics["f1"]
            if abs(g_f1 - c_f1) < 0.01:
                insights.append(("#6366f1", "Grover-Classical 동등 구조",
                    f"Grover가 선택한 DAG(F1={g_f1:.2f})와 고전 전수조사 1위(F1={c_f1:.2f})의 정확도가 동등합니다. "
                    f"양자 탐색이 전수조사 없이도 동일한 품질의 구조를 찾을 수 있음을 보여줍니다."))
            elif g_f1 > c_f1:
                insights.append(("#059669", "Grover가 더 좋은 구조 선택",
                    f"Grover 후처리 DAG(F1={g_f1:.2f})가 고전 전수조사 1위(F1={c_f1:.2f})보다 정답에 가깝습니다. "
                    f"Score-weighted 선택이 BDeu 점수만으로는 놓칠 수 있는 구조를 포착한 경우입니다."))
            else:
                rank = grover_result.get("selected_rank", "?")
                insights.append(("#f59e0b", "Grover 선택 구조 차이",
                    f"Grover가 선택한 DAG(rank {rank}, F1={g_f1:.2f})는 고전 1위(F1={c_f1:.2f})보다 약간 낮습니다. "
                    f"이는 Grover 측정의 확률적 특성 때문이며, multi-run으로 보완할 수 있습니다."))

    # 4. Scalability note
    n_vars = len(variables)
    if n_vars >= 4:
        future_5 = 2 ** (5 * 4)
        grover_5 = int(math.pi / 4 * math.sqrt(future_5 / max(top_k, 1)))
        insights.append(("#6366f1", "확장성 전망",
            f"현재 {n_vars}개 변수({n_edges} edge bits)에서는 고전 전수조사가 가능하지만, "
            f"5개 변수(20 bits)로 확장하면 후보가 {future_5:,}개로 폭발합니다. "
            f"Grover는 이때 약 {grover_5:,}회 반복이면 충분해, 이차적 속도 이점이 실질적으로 드러나기 시작합니다."))

    return insights


st.set_page_config(
    page_title="Quantum Causal Discovery Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ── */
    .block-container {
        padding-top: 2.75rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155;
    }
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #64748b !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"] > div,
    section[data-testid="stSidebar"] textarea {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="input"] input,
    section[data-testid="stSidebar"] textarea {
        color: #0f172a !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #334155 !important;
        fill: #334155 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stMultiSelect,
    section[data-testid="stSidebar"] .stTextInput {
        margin-bottom: 0.75rem;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p {
        color: #334155 !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        color: #334155 !important;
    }

    /* ── Hero header ── */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-container .hero-title,
    .hero-title {
        display: block;
        font-size: 1.85rem !important;
        line-height: 1.22 !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        margin: 0 0 0.3rem 0;
        letter-spacing: 0;
        text-shadow: 0 2px 18px rgba(0,0,0,0.35);
    }
    .hero-container .hero-subtitle,
    .hero-subtitle {
        display: block;
        font-size: 0.95rem !important;
        color: #dbeafe !important;
        margin: 0;
        line-height: 1.5;
        max-width: 920px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        margin-bottom: 0.7rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.5rem 1.2rem;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 2px 8px rgba(99,102,241,0.3);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 16px rgba(99,102,241,0.4);
        transform: translateY(-1px);
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.9rem;
        color: #334155;
        background: #f8fafc;
        border-radius: 8px;
    }

    /* ── Story band ── */
    .status-band {
        border-left: 4px solid #6366f1;
        background: linear-gradient(90deg, #eef2ff 0%, #f8fafc 100%);
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 0.4rem 0 1.2rem 0;
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.55;
    }
    .upload-empty-state {
        margin-top: 1.2rem;
        border-left: 4px solid #2563eb;
        background: linear-gradient(90deg, #eff6ff 0%, #ffffff 100%);
        color: #1e3a8a;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.4rem;
        line-height: 1.6;
        font-size: 0.95rem;
        max-width: 920px;
        overflow-wrap: break-word;
        word-break: keep-all;
    }
    .upload-empty-state b {
        color: #1d4ed8;
    }

    /* ── Value proposition cards ── */
    .value-props {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
        margin: 0 0 1.5rem 0;
    }
    .value-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .value-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .value-icon {
        font-size: 1.6rem;
        margin-bottom: 0.4rem;
    }
    .value-title {
        font-weight: 700;
        color: #0f172a;
        font-size: 0.92rem;
        margin-bottom: 0.3rem;
    }
    .value-desc {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.5;
    }

    /* ── Key finding / Next step cards ── */
    .finding-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #86efac;
        border-left: 5px solid #059669;
        border-radius: 0 14px 14px 0;
        padding: 1.3rem 1.5rem;
        margin: 1.2rem 0 0.6rem 0;
    }
    .finding-card .finding-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #059669;
        margin-bottom: 0.4rem;
    }
    .finding-card .finding-body {
        font-size: 0.93rem;
        color: #1e293b;
        line-height: 1.6;
    }
    .nextstep-card {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        border: 1px solid #a5b4fc;
        border-left: 5px solid #6366f1;
        border-radius: 0 14px 14px 0;
        padding: 1rem 1.3rem;
        margin: 0.4rem 0 0.5rem 0;
    }
    .nextstep-card .nextstep-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6366f1;
        margin-bottom: 0.3rem;
    }
    .nextstep-card .nextstep-body {
        font-size: 0.9rem;
        color: #312e81;
        line-height: 1.55;
    }
    .chapter-num {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff;
        font-weight: 800;
        font-size: 0.7rem;
        width: 22px;
        height: 22px;
        line-height: 22px;
        text-align: center;
        border-radius: 50%;
        margin-right: 0.5rem;
        vertical-align: middle;
    }

    /* ── Info / Warning / Success boxes ── */
    .stAlert > div {
        border-radius: 10px;
        border: none;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    /* ── Prevent horizontal overflow ── */
    .block-container, .status-band, .hero-container {
        overflow-x: hidden;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Small note ── */
    .small-note {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ── Section headers ── */
    h2, h3 {
        color: #0f172a;
        letter-spacing: -0.01em;
    }

    /* ── Dividers ── */
    hr {
        border-color: #e2e8f0;
        margin: 1.5rem 0;
    }

    /* ── Project Intro Tab ── */
    .intro-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .intro-section h4 {
        color: #0f172a;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
    }
    .intro-section p, .intro-section li {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.65;
    }
    .intro-accent {
        border-left: 4px solid #6366f1;
        background: linear-gradient(90deg, #eef2ff 0%, #f8fafc 100%);
        padding: 1.2rem 1.4rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        font-size: 0.93rem;
        color: #1e293b;
        line-height: 1.6;
    }
    .concept-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.9rem;
        margin: 0.8rem 0;
    }
    .concept-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .concept-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .concept-card .concept-term {
        font-weight: 700;
        color: #4f46e5;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .concept-card .concept-eng {
        font-size: 0.78rem;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .concept-card .concept-desc {
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.55;
    }
    .step-flow {
        display: flex;
        gap: 0.8rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .step-card {
        flex: 1;
        min-width: 180px;
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        position: relative;
    }
    .step-card .step-num {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.75rem;
        width: 24px;
        height: 24px;
        line-height: 24px;
        text-align: center;
        border-radius: 50%;
        margin-bottom: 0.5rem;
    }
    .step-card .step-title {
        font-weight: 700;
        color: #0f172a;
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }
    .step-card .step-desc {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.5;
    }
    .quantum-highlight {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    .quantum-highlight h4 {
        color: #a5b4fc !important;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
    }
    .quantum-highlight p, .quantum-highlight li {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .quantum-highlight code {
        background: rgba(99,102,241,0.2);
        color: #c7d2fe;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-size: 0.84rem;
    }
    .guide-banner {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    .guide-banner h4 {
        color: #ffffff !important;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .guide-banner p {
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .reflection-box {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
        border: 1px solid #fde68a;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
    }
    .reflection-box h4 {
        color: #92400e;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .reflection-box p {
        color: #78350f;
        font-size: 0.9rem;
        line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


dataset_options = available_datasets()

st.sidebar.title("실험 설정")
data_mode = st.sidebar.radio("데이터 소스", ["내장 데이터셋", "CSV 업로드"], horizontal=False)

uploaded_bytes: bytes | None = None
ground_truth_edges: list[tuple[str, str]] = []

if data_mode == "내장 데이터셋":
    dataset_name = st.sidebar.selectbox("데이터셋", list(dataset_options.keys()))
    spec = dataset_options[dataset_name]
    raw_df = load_csv(spec.file)
    default_vars = [var for var in spec.default_vars if var in raw_df.columns]
    if not default_vars:
        default_vars = list(raw_df.columns[: min(4, len(raw_df.columns))])
    selected_vars = st.sidebar.multiselect(
        "분석 변수",
        list(raw_df.columns),
        default=default_vars[:4],
        max_selections=4,
    )
    ground_truth_edges = list(spec.ground_truth)
    story = spec.story
    outcome_hint = spec.outcome_hint
else:
    dataset_name = "Custom CSV"
    spec = None
    uploaded = st.sidebar.file_uploader("CSV 파일", type=["csv"])
    if uploaded is None:
        st.markdown(
            """
            <div class="upload-empty-state">
            <b>CSV 파일을 기다리는 중입니다.</b><br>
            왼쪽 사이드바의 업로드 영역에서 CSV 파일을 선택하세요.
            파일을 업로드하면 분석 변수, 결과 변수, BDeu 점수 계산, 개입 추천, Groq 해석 기능이 표시됩니다.<br>
            바로 시연하려면 데이터 소스를 <b>내장 데이터셋</b>으로 바꾸고 <b>Sprinkler weather</b>를 선택하세요.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()
    uploaded_bytes = uploaded.getvalue()
    try:
        raw_df = read_csv_bytes(uploaded_bytes)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    selected_vars = st.sidebar.multiselect(
        "분석 변수",
        list(raw_df.columns),
        default=list(raw_df.columns[: min(3, len(raw_df.columns))]),
        max_selections=4,
    )
    story = "사용자가 업로드한 데이터에서 변수 사이의 인과 구조를 탐색합니다."
    outcome_hint = None
    with st.sidebar.expander("정답 엣지 입력"):
        edge_text = st.text_area("한 줄에 하나씩 입력", placeholder="A->B\nB->C")
        ground_truth_edges = parse_edge_text(edge_text)

if len(selected_vars) < 2:
    st.warning("분석하려면 변수를 2개 이상 선택해야 합니다.")
    st.stop()

variables = list(selected_vars)
if len(variables) > 4:
    st.warning("Grover 시뮬레이션 안정성을 위해 변수는 최대 4개까지 선택합니다.")
    st.stop()

if data_mode == "내장 데이터셋":
    with st.sidebar.expander("정답 엣지 확인/수정"):
        default_edge_text = "\n".join(f"{src}->{dst}" for src, dst in ground_truth_edges)
        custom_edge_text = st.text_area("선택 변수에 포함된 엣지만 사용됩니다", value=default_edge_text)
        ground_truth_edges = parse_edge_text(custom_edge_text)

st.sidebar.divider()
scoring_method = st.sidebar.radio(
    "점수 함수",
    ["BDeu (이산)", "BGe (연속)"],
    horizontal=True,
    help="BDeu: 이산 데이터에 최적. BGe: 연속 데이터를 이산화 없이 직접 평가 (정보 손실 없음).",
)
use_bge = scoring_method.startswith("BGe")
ess = int(st.sidebar.slider("BDeu ESS", min_value=1, max_value=50, value=10, disabled=use_bge))
top_k = int(st.sidebar.slider("Grover Oracle top-k", min_value=1, max_value=20, value=6))
shots = int(st.sidebar.slider("측정 shots", min_value=512, max_value=8192, value=4096, step=512))
auto_discretize = st.sidebar.toggle("연속형 변수 3분위 이산화", value=True, disabled=use_bge)

outcome_default = variables[-1]
if outcome_hint in variables:
    outcome_default = outcome_hint
outcome = st.sidebar.selectbox("결과 변수", variables, index=variables.index(outcome_default))

_default_higher_better, _direction_source = infer_outcome_direction(outcome, spec)
outcome_direction = st.sidebar.radio(
    "결과 변수 방향",
    ["높을수록 나쁨 (줄이고 싶다)", "높을수록 좋음 (높이고 싶다)"],
    index=1 if _default_higher_better else 0,
    horizontal=True,
    key=f"outcome_direction_{dataset_name}_{outcome}",
)
outcome_higher_is_better = "좋음" in outcome_direction
if _direction_source == "unknown":
    st.sidebar.warning("이 결과 변수의 방향은 자동 판단이 어렵습니다. 분석 목적에 맞게 반드시 확인하세요.")
else:
    st.sidebar.caption("결과 변수 방향 기본값은 선택한 outcome 기준으로 설정됩니다.")

st.sidebar.divider()
st.sidebar.markdown("**AI 해석 (선택)**")
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Groq 콘솔에서 발급받은 API 키를 입력하면 분석 결과를 AI가 자연어로 해석해 줍니다. 없어도 앱의 모든 기능을 사용할 수 있습니다.",
)
groq_api_key = groq_api_key.strip() if groq_api_key else ""
ai_enabled = bool(groq_api_key)

csv_bytes = uploaded_bytes if uploaded_bytes is not None else raw_df.to_csv(index=False).encode("utf-8")

_scoring_label = "BGe" if use_bge else "BDeu"
with st.spinner(f"DAG 후보를 열거하고 {_scoring_label} 점수를 계산하는 중입니다."):
    try:
        if use_bge:
            data, valid_dags, edge_list, scored, scoring_elapsed = score_from_csv_bytes_bge(
                csv_bytes,
                tuple(variables),
            )
        else:
            data, valid_dags, edge_list, scored, scoring_elapsed = score_from_csv_bytes(
                csv_bytes,
                tuple(variables),
                ess,
                auto_discretize,
            )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"분석 계산 중 오류가 발생했습니다: {exc}")
        st.stop()

if len(data) != len(raw_df):
    st.sidebar.caption(f"전처리 후 분석 행: {len(data):,} / 원본 {len(raw_df):,}")
low_information_cols = [col for col in variables if data[col].nunique(dropna=True) < 2]
if low_information_cols:
    st.sidebar.warning(f"변동이 없는 변수는 구조 식별력이 낮습니다: {', '.join(low_information_cols)}")

ground_truth = build_ground_truth(variables, ground_truth_edges)
has_ground_truth = len(ground_truth.edges()) > 0

n_edges = len(edge_list)
n_total = 2**n_edges
valid_bitstrings = {bitstring for bitstring, _ in valid_dags}
top_k_effective = min(top_k, len(scored))
good_bitstrings = [scored[idx][0] for idx in range(top_k_effective)]

best_bitstring, best_graph, best_score = scored[0]
best_metrics = graph_metrics(ground_truth if has_ground_truth else None, best_graph)
run_key = f"{dataset_name}|{','.join(variables)}|{ess}|{top_k}|{shots}|{auto_discretize}|{len(data)}"

st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-badge">Quantum Causal Discovery Lab</div>
        <div class="hero-title" style="color:#f8fafc !important;font-size:1.85rem;font-weight:800;line-height:1.22;text-shadow:0 2px 18px rgba(0,0,0,0.35);">
            데이터에서 '진짜 원인'을 찾아, 가장 효과적인 개입 지점을 추천합니다
        </div>
        <div class="hero-subtitle" style="color:#dbeafe !important;font-size:0.95rem;line-height:1.5;max-width:920px;">
            "두 변수가 함께 움직인다"는 것은 하나가 다른 하나의 원인이라는 뜻이 아닙니다.
            이 시스템은 관측 데이터에서 변수 간 <b>숨겨진 인과 구조</b>를 자동으로 발견하고,
            <b>어떤 변수를 조절해야 원하는 결과를 얻을 수 있는지</b> 개입 전략을 추천합니다.
            탐색 과정에 양자 컴퓨팅(Grover 알고리즘)을 접목해 속도 향상 가능성을 시연합니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="value-props">
        <div class="value-card">
            <div class="value-icon">🔍</div>
            <div class="value-title">인과 구조 자동 발견</div>
            <div class="value-desc">수천 개의 가능한 인과 그래프 중<br>데이터에 가장 부합하는 구조를 찾습니다</div>
        </div>
        <div class="value-card">
            <div class="value-icon">🎯</div>
            <div class="value-title">개입 타겟 추천</div>
            <div class="value-desc">결과를 바꾸려면 어떤 변수를<br>조절해야 하는지 순위를 매겨 추천합니다</div>
        </div>
        <div class="value-card">
            <div class="value-icon">⚛️</div>
            <div class="value-title">양자 알고리즘 접목</div>
            <div class="value-desc">Grover 탐색으로 최적 구조를 찾는<br>속도를 이론적으로 제곱근만큼 단축합니다</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='status-band'>{story}</div>", unsafe_allow_html=True)

# Compute intervention for top metric display
_preview_intervention = intervention_table(data, best_graph, variables, outcome, outcome_higher_is_better)
if has_actionable_intervention(_preview_intervention):
    _top_target = _preview_intervention.iloc[0]["target"]
    _top_action = _preview_intervention.iloc[0]["recommended_action"]
else:
    _top_target = "-"
    _top_action = "추천 가능한 개입 없음"

metric_cols = st.columns(5)
metric_cols[0].metric("분석 변수", f"{len(variables)}개", ", ".join(variables))
metric_cols[1].metric("유효 DAG", f"{len(valid_dags):,}개", f"전체 {n_total:,}개 중")
metric_cols[2].metric("최적 구조 점수", f"{best_score:.1f}", f"{_scoring_label}" + (f" (ESS={ess})" if not use_bge else " (연속)"))
if best_metrics:
    metric_cols[3].metric("정답 대비 F1", f"{best_metrics['f1']:.2f}", f"SHD={best_metrics['shd']}")
else:
    metric_cols[3].metric("정답 대비", "N/A", "정답 구조 없음")
metric_cols[4].metric("추천 개입 타겟", _top_target, _top_action)

tabs = st.tabs(["프로젝트 소개", "왜 인과관계인가", "인과 구조 발견", "개입 추천", "양자적 접근", "종합 분석"])

with tabs[0]:
    # ═══ 프로젝트 소개 ═══

    # ── 지금 분석 중인 데이터로 시작: "당신의 문제는 이것입니다" ──
    _intro_outcome_dir = "높이" if outcome_higher_is_better else "줄이"
    _intro_best_edges = format_edges(best_graph.edges())
    _intro_has_intv = has_actionable_intervention(_preview_intervention)
    _intro_top = _preview_intervention.iloc[0] if _intro_has_intv else None

    st.markdown(
        f"""
        <div class="intro-section" style="border-left: 4px solid #6366f1; border-radius: 0 14px 14px 0;">
            <h4 style="color:#4f46e5;">지금 보고 있는 분석</h4>
            <p>
            <b>{dataset_name}</b> 데이터의 변수 <b>{', '.join(variables)}</b> 사이에서
            인과 구조를 탐색하고 있습니다.<br>
            목표는 <b><code>{outcome}</code></b> 값을 <b>{_intro_outcome_dir}기 위해</b>
            어떤 변수를 조절해야 하는지 찾는 것입니다.
            </p>
            <p>
            현재까지의 결과: 유효한 DAG 후보 <b>{len(valid_dags):,}개</b> 중 최적 구조는
            <b>{_intro_best_edges}</b>이며{f', 가장 효과적인 개입 타겟은 <b>{_intro_top["target"]}</b>({_intro_top["recommended_action"]})입니다.' if _intro_has_intv else ', 아직 명확한 개입 타겟이 도출되지 않았습니다.'}
            <b>아래 탭을 하나씩 따라가면 이 결과가 어떻게 도출되었는지 직접 확인할 수 있습니다.</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 이 앱은 어떤 문제를 풀고 있는가 ──
    st.markdown("#### 이 앱이 풀고 있는 문제")

    prob_cols = st.columns([1, 1])
    with prob_cols[0]:
        st.markdown(
            """
            **데이터를 보고 의사결정을 내린다고 합시다.**

            매출이 오르면 광고비도 올랐다. 그러면 "광고를 더 하면 매출이 오르는가?"
            아이스크림 판매량과 익사 사고가 함께 올랐다. "아이스크림을 금지하면 익사가 줄어드는가?"

            정답은 **알 수 없다**입니다.
            함께 움직인다(상관관계)는 것은 하나가 다른 하나의 원인(인과관계)이라는 뜻이 **아닙니다.**
            상관관계만 보고 개입하면, **엉뚱한 곳에 돈과 시간을 쓰게 됩니다.**
            """
        )
    with prob_cols[1]:
        st.markdown(
            f"""
            **이 앱이 하는 일:**

            1. 관측 데이터에서 **"무엇이 무엇의 원인인가"** (인과 구조, DAG)를 자동으로 찾습니다
            2. 찾은 구조를 바탕으로 **"어디에 개입하면 결과가 바뀌는가"** 를 계산합니다
            3. 이 과정에서 양자 컴퓨팅(Grover 알고리즘)이 탐색을 가속할 수 있음을 시연합니다

            지금 보고 있는 **{dataset_name}** 데이터에서,
            **{outcome}**을 {_intro_outcome_dir}려면 어떤 변수를 건드려야 하는지 —
            이 앱이 끝까지 답을 제시합니다.
            """
        )

    st.divider()

    # ── 분석 파이프라인: 이 앱이 작동하는 방식 ──
    st.markdown("#### 분석 파이프라인")
    st.markdown(
        f"""
        <p class="small-note" style="margin-bottom:0.6rem;">
        아래 5단계를 각 탭에서 직접 확인하고 조작할 수 있습니다. 사이드바에서 파라미터를 바꾸면 모든 결과가 실시간으로 갱신됩니다.
        </p>
        <div class="step-flow">
            <div class="step-card">
                <div class="step-num">1</div>
                <div class="step-title">상관 vs 인과</div>
                <div class="step-desc">
                    {', '.join(variables)} 변수의 상관행렬을 보여주고,
                    왜 이것만으로는 부족한지 설명합니다.
                </div>
            </div>
            <div class="step-card">
                <div class="step-num">2</div>
                <div class="step-title">구조 탐색</div>
                <div class="step-desc">
                    {n_total:,}개 후보 중 유효한 {len(valid_dags):,}개를
                    BDeu 점수로 평가해 최적 DAG를 찾습니다.
                </div>
            </div>
            <div class="step-card">
                <div class="step-num">3</div>
                <div class="step-title">개입 추천</div>
                <div class="step-desc">
                    발견된 DAG로 do-calculus를 적용,
                    {outcome}에 가장 큰 영향을 주는 변수를 추천합니다.
                </div>
            </div>
            <div class="step-card">
                <div class="step-num">4</div>
                <div class="step-title">양자 탐색</div>
                <div class="step-desc">
                    {n_edges}큐비트 Grover 회로로 고득점 DAG의
                    측정 확률을 증폭시키는 실험입니다.
                </div>
            </div>
            <div class="step-card">
                <div class="step-num">5</div>
                <div class="step-title">종합 판단</div>
                <div class="step-desc">
                    고전 vs 양자 비교, 게이지, 레이더 차트,
                    AI 종합 보고서를 한 곳에서 확인합니다.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 핵심 용어: 펼쳐볼 수 있는 형태 ──
    st.markdown("#### 처음이라면: 핵심 개념 6가지")
    st.markdown("<p class='small-note'>각 항목을 펼쳐서 읽어보세요. 이후 탭에서 이 개념들이 실제 데이터에 어떻게 적용되는지 직접 확인하게 됩니다.</p>", unsafe_allow_html=True)

    _concept_cols = st.columns(2)
    with _concept_cols[0]:
        with st.expander("DAG (방향 비순환 그래프)"):
            st.markdown(
                f"""
                **Directed Acyclic Graph** — 변수 간 인과관계를 화살표로 표현한 그래프입니다.

                - `A → B`는 "A가 B의 원인"이라는 뜻
                - 순환(A→B→A)이 없어야 해서 '비순환'
                - 지금 분석에서는 **{', '.join(variables)}** 사이의 가능한 화살표 조합 **{n_total:,}개** 중
                  비순환 조건을 만족하는 **{len(valid_dags):,}개**가 후보입니다

                **→ [인과 구조 발견] 탭**에서 데이터가 가장 지지하는 DAG를 찾습니다.
                """
            )
        with st.expander("BDeu 점수"):
            st.markdown(
                f"""
                **Bayesian Dirichlet equivalent uniform Score** — 특정 DAG 구조가 관측 데이터에
                얼마나 잘 맞는지 측정하는 점수입니다.

                - 점수가 높을수록 "이 인과 구조가 데이터를 잘 설명한다"
                - ESS(Equivalent Sample Size) 파라미터로 사전분포 강도 조절 (현재 ESS={ess})
                - 현재 최고 점수: **{best_score:.1f}**

                **→ [인과 구조 발견] 탭**에서 점수 분포를 확인하고 상위 DAG를 비교합니다.
                """
            )
        with st.expander("do-calculus / Backdoor Adjustment"):
            st.markdown(
                f"""
                **개입 연산** — "변수 X를 인위적으로 특정 값으로 고정하면 Y가 어떻게 바뀌는가?"를
                계산하는 인과 추론 방법입니다.

                - 단순히 "X가 높을 때 Y도 높더라"(관찰)가 아니라
                - "X를 **강제로** 높이면 Y가 **얼마나** 바뀌는가"(개입)를 추정
                - 교란 변수(confounders)를 통계적으로 차단해서 순수 효과만 계산

                **→ [개입 추천] 탭**에서 {outcome}에 대한 각 변수의 개입 효과를 비교합니다.
                """
            )
    with _concept_cols[1]:
        with st.expander("Grover 알고리즘"):
            st.markdown(
                f"""
                **Grover's Search Algorithm** — 정렬되지 않은 N개 후보에서 원하는 답을 찾는 양자 알고리즘입니다.

                - 고전 컴퓨터: 최악의 경우 N번 모두 확인해야 함 → **O(N)**
                - Grover: 양자 간섭으로 약 √N번만에 찾음 → **O(√N)**
                - 현재 분석: 후보 {n_total:,}개 → 고전 {n_total:,}번 vs Grover ~{int(math.sqrt(n_total))}번

                **→ [양자적 접근] 탭**에서 직접 Grover 회로를 실행해볼 수 있습니다.
                """
            )
        with st.expander("진폭 증폭 (Amplitude Amplification)"):
            st.markdown(
                """
                Grover 알고리즘의 핵심 원리입니다.

                - 모든 후보를 양자 **중첩** 상태로 놓음 (각각 동일한 확률)
                - **Oracle**: "이 상태가 좋은 답인가?" 판별 → 좋은 답에 위상 표시
                - **Diffuser**: 양자 간섭으로 좋은 답의 확률을 **높이고** 나쁜 답의 확률을 **낮춤**
                - 이걸 적절한 횟수만큼 반복하면, 측정 시 좋은 답이 나올 확률이 극대화

                **→ [양자적 접근] 탭**의 측정 분포 차트에서 증폭 효과를 눈으로 확인합니다.
                """
            )
        with st.expander("Oracle (오라클)"):
            st.markdown(
                f"""
                Grover 회로에서 **"이 상태가 정답인가?"를 판별하는 양자 게이트**입니다.

                - 이 앱에서는 BDeu 점수 **상위 {top_k_effective}개** DAG를 "좋은 답"으로 표시
                - Oracle이 표시한 상태의 진폭이 Diffuser를 통해 증폭됨
                - 현재 한계: 점수 계산 자체는 고전적으로 수행 후 결과를 Oracle에 하드코딩
                  (향후 In-circuit Scoring으로 발전 가능)

                **→ [양자적 접근] 탭**에서 Oracle 타겟 목록과 회로 구조를 확인합니다.
                """
            )

    st.divider()

    # ── 양자적 접근: 왜 Grover인가 (데이터에 맞춰 동적 설명) ──
    st.markdown(
        f"""
        <div class="quantum-highlight">
            <h4>왜 양자 컴퓨팅을 접목했는가</h4>
            <p>
            인과 구조 탐색은 본질적으로 <b>비정렬 탐색 문제</b>입니다.
            지금 분석하는 {len(variables)}개 변수만 해도 가능한 엣지 조합이 <b>{n_total:,}개</b>이고,
            변수가 하나 늘 때마다 탐색 공간은 기하급수적으로 커집니다.
            </p>
            <p>
            이 프로젝트는 Qiskit Aer 시뮬레이터를 활용해 이 문제를 양자 도메인으로 정식화했습니다:
            </p>
            <ul style="margin:0.5rem 0; padding-left:1.3rem;">
                <li><b>인코딩</b>: {n_edges}개 엣지 후보 각각을 큐비트 1개로 매핑 → <b>{n_edges}큐비트</b> 회로</li>
                <li><b>Oracle</b>: BDeu 상위 {top_k_effective}개 DAG를 marked state로 설정</li>
                <li><b>증폭</b>: Grover 반복을 통해 고득점 구조의 측정 확률을 유의미하게 높임</li>
                <li><b>의의</b>: 전수조사 <code>O({n_total:,})</code> → Grover <code>O({int(math.sqrt(n_total))})</code> 이차 속도 향상 시연</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Groq API 안내 ──
    st.markdown(
        """
        <div class="guide-banner">
            <h4>AI 해석 기능 안내</h4>
            <p>
            사이드바에 Groq API 키를 입력하면 각 탭에 <b>"AI 해석 생성"</b> 버튼이 활성화됩니다.
            LLaMA 3.3 70B 모델이 통계 결과를 전문 보고서 형태로 변환해줍니다.
            API 키 없이도 모든 분석과 시각화는 정상 작동하며, 앱 자체 해석도 함께 제공됩니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 제작 과정과 회고 ──
    st.markdown("#### 이 앱은 어떻게 만들어졌는가")
    process_cols = st.columns([1, 1])
    with process_cols[0]:
        with st.expander("기술적 도전과 해결", expanded=True):
            st.markdown(
                """
                **BDeu 점수 함수 직접 구현**
                라이브러리에 의존하지 않고 노드별 조건부 분포의
                marginal likelihood를 직접 계산합니다.

                **Backdoor Adjustment 구현**
                교란 변수가 통제된 상태에서의 순수 개입 효과를 추정.
                관측 조합이 부족한 경우를 위해 Coverage 기반 신뢰도 지표를
                도입하여 "이 추정을 얼마나 믿을 수 있는가"까지 제시합니다.

                **Groq API 안정화**
                보안 차단(Error 1010)을 User-Agent 헤더 설정으로 해결.
                API 실패 시에도 앱 자체 해석으로 fallback합니다.
                """
            )
    with process_cols[1]:
        with st.expander("프로젝트의 의의와 한계", expanded=True):
            st.markdown(
                """
                **의의**: 양자 컴퓨팅 수업에서 학습한 Grover 알고리즘을
                실제 데이터 과학의 난제인 인과 추론에 접목했습니다.
                이론적 회로를 넘어, **현실의 의사결정 문제를 양자 알고리즘에
                맞게 정식화(Formulation)하는 과정** 자체가 핵심 기여입니다.

                **현재 한계**: Oracle 내에서 BDeu 점수를 직접 계산하지 않고
                고전적으로 미리 계산한 결과를 사용합니다(Pre-computed Oracle).

                **향후 전망**: In-circuit Scoring — Oracle 안에서
                점수를 양자적으로 계산하는 구조가 구현되면, 변수가 많은
                실제 문제에서 진정한 양자 이점을 실현할 수 있습니다.
                """
            )

with tabs[1]:
    # ═══ 왜 인과관계인가 ═══
    st.markdown(
        f"""<span class="chapter-num">1</span> <b style="font-size:1.15rem;">상관관계 ≠ 인과관계 — 왜 이 구분이 중요한가</b>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-band">
        두 변수가 함께 움직인다고 해서 하나가 다른 하나의 <b>원인</b>은 아닙니다.
        아이스크림 판매량과 익사 사고가 함께 증가하지만, 아이스크림이 익사를 일으키는 것은 아닙니다 — 공통 원인(여름 기온)이 존재합니다.<br><br>
        지금 분석 중인 <b>{dataset_name}</b> 데이터에서도 마찬가지입니다.
        <b>{', '.join(variables)}</b> 사이에 높은 상관관계가 있더라도,
        그것만으로는 <b>"{outcome}을 바꾸려면 어디를 건드려야 하는가"</b>에 답할 수 없습니다.
        아래에서 이 데이터의 상관관계를 직접 확인해 보세요.
        </div>
        """,
        unsafe_allow_html=True,
    )

    corr_cols = st.columns([1, 1])
    with corr_cols[0]:
        st.pyplot(plot_correlation(data), use_container_width=True)
        st.markdown(
            """
            <p class="small-note">
            상관 행렬은 변수 간 <b>선형 관계의 강도</b>만 보여줍니다.<br>
            높은 상관이 있어도 어느 방향인지, 직접 효과인지 제3의 변수를 통한 간접 효과인지 알 수 없습니다.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with corr_cols[1]:
        st.markdown(
            f"""
            #### 왜 DAG(방향 비순환 그래프)가 필요한가?

            | 상관관계 | 인과관계 (DAG) |
            |---|---|
            | A와 B가 같이 움직인다 | A가 B를 **일으킨다** |
            | 방향이 없다 | **화살표 방향**이 있다 |
            | 개입 효과를 예측 못함 | do(A=x)의 효과를 계산 가능 |
            | 교란 변수 구분 불가 | 교란 경로를 차단 가능 |

            이 앱은 관측 데이터 `{dataset_name}`에서 **DAG를 자동으로 발견**하고,
            발견된 구조를 활용해 **어떤 변수에 개입해야 `{outcome}`을 바꿀 수 있는지** 추천합니다.
            """
        )

    st.divider()

    # Data overview
    st.markdown("#### 분석 데이터")
    data_cols = st.columns([1.2, 0.8])
    with data_cols[0]:
        description = spec.description if spec is not None else "업로드한 CSV 데이터입니다."
        st.markdown(
            f"""
            **데이터셋**: {dataset_name} &nbsp;|&nbsp; **샘플 수**: {len(data):,} &nbsp;|&nbsp; **변수**: {', '.join(variables)}

            {description}
            """
        )
        st.dataframe(data.head(8), use_container_width=True)
    with data_cols[1]:
        if has_ground_truth:
            st.pyplot(
                draw_dag(ground_truth, variables, "Ground Truth DAG", subtitle=format_edges(ground_truth.edges())),
                use_container_width=True,
            )
            st.markdown(
                "<p class='small-note'>문헌에서 검증된 인과 구조입니다. 데이터 기반 발견 결과와 비교할 기준이 됩니다.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("이 데이터셋에는 알려진 정답 구조가 없습니다. 발견된 구조를 직접 평가해야 합니다.")

    # ── 핵심 발견 & 다음 단계 ──
    st.markdown(
        f"""
        <div class="finding-card">
            <div class="finding-label">이 탭의 핵심</div>
            <div class="finding-body">
            <b>{', '.join(variables)}</b>의 상관행렬을 확인했습니다.
            상관관계는 변수 간 <b>선형적 동조 여부</b>만 보여줄 뿐, 어느 변수가 다른 변수의 <b>원인</b>인지,
            개입했을 때 결과가 <b>실제로 바뀌는지</b>는 알려주지 않습니다.
            이 한계를 극복하기 위해 <b>인과 구조(DAG)</b>를 데이터로부터 직접 찾아야 합니다.
            </div>
        </div>
        <div class="nextstep-card">
            <div class="nextstep-label">다음 단계 →  인과 구조 발견</div>
            <div class="nextstep-body">
            다음 탭에서는 {n_total:,}개의 가능한 DAG 후보 중 데이터에 가장 부합하는 구조를 <b>BDeu 점수</b>로 자동 탐색합니다.
            {f'문헌의 정답 구조와 비교하여 발견 정확도도 평가합니다.' if has_ground_truth else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[2]:
    # ═══ 인과 구조 발견 ═══
    st.markdown(
        f"""<span class="chapter-num">2</span> <b style="font-size:1.15rem;">데이터가 지지하는 인과 구조 찾기</b>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-band">
        앞서 상관관계만으로는 인과를 알 수 없다는 것을 확인했습니다.
        이제 <b>{', '.join(variables)}</b> 사이에 가능한 인과 화살표 {n_edges}개로 구성되는 DAG 후보 <b>{n_total:,}개</b>를 전수 평가합니다.<br>
        각 DAG에 <b>BDeu 점수</b>(데이터와의 적합도)를 매겨, 관측 데이터를 가장 잘 설명하는 인과 구조를 찾습니다.
        점수가 높을수록 "이 인과 관계가 데이터에 의해 지지된다"는 뜻입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # DAG comparison
    graph_cols = st.columns(3 if has_ground_truth else 2)
    if has_ground_truth:
        with graph_cols[0]:
            st.pyplot(
                draw_dag(ground_truth, variables, "Ground Truth DAG", subtitle=format_edges(ground_truth.edges())),
                use_container_width=True,
            )
    with graph_cols[-2]:
        subtitle = format_edges(best_graph.edges())
        if best_metrics:
            subtitle = f"SHD={best_metrics['shd']}, F1={best_metrics['f1']:.2f}"
        st.pyplot(
            draw_dag(best_graph, variables, "Best DAG by BDeu", reference=ground_truth if has_ground_truth else None, subtitle=subtitle),
            use_container_width=True,
        )
    with graph_cols[-1]:
        st.pyplot(plot_score_distribution(scored, good_bitstrings), use_container_width=True)

    if has_ground_truth and best_metrics:
        st.pyplot(plot_shd_explanation(ground_truth, best_graph, variables), use_container_width=True)

    st.divider()

    st.markdown("#### 상위 DAG 후보")
    st.dataframe(candidate_table(scored, ground_truth if has_ground_truth else None), use_container_width=True, hide_index=True)

    with st.expander("BDeu 점수와 구조 학습의 한계"):
        st.markdown(
            f"""
            **BDeu (Bayesian Dirichlet equivalent uniform)** 는 이산 베이지안 네트워크의 구조 점수입니다.
            각 노드별로 부모-자식 조건부 분포의 marginal likelihood를 계산하고 합산합니다. 높을수록 데이터에 더 잘 맞습니다.

            - **ESS (Equivalent Sample Size)** = {ess}: 사전분포의 강도. 작을수록 데이터에 충실하고 희소한 구조를 선호합니다.
            - 현재 1위 BDeu = **{scored[0][2]:.2f}**, 유효 DAG {len(valid_dags):,}개 중 최고점입니다.

            **왜 정답과 다를 수 있는가?**
            - **Markov equivalence**: 관측 데이터만으로는 동일한 조건부 독립 관계를 만드는 여러 DAG를 구분할 수 없습니다 (예: A→B와 A←B는 주변 분포가 같을 수 있음).
            - **변수 부분 선택**: 전체 네트워크의 일부 변수만 분석하면, 숨은 매개변수/교란변수 효과로 방향이 바뀔 수 있습니다.
            - **이산화**: 연속형 변수를 이산화하면 정보가 손실되어 구조 식별력이 떨어집니다.
            - F1이 낮더라도 **상위 DAG 다수가 비슷한 방향성을 공유**하면, 개입 추천의 방향 자체는 신뢰할 수 있습니다.

            **핵심 한계 문장**: 이 앱은 관측 데이터만으로 완전한 인과관계를 증명하는 도구가 아니라,
            가능한 DAG를 점수화하고 개입 후보를 비교하는 탐색형 도구입니다.
            """
        )

    if ai_enabled:
        _ai_edges = format_edges(best_graph.edges())
        _ai_gt_text = f"알려진 정답 구조: {format_edges(ground_truth.edges())}\nSHD={best_metrics['shd']}, F1={best_metrics['f1']:.2f}" if has_ground_truth and best_metrics else "정답 구조 없음"
        _ai_prompt_structure = (
            f"당신은 인과 추론 전문가입니다. 아래 분석 결과를 비전문가도 이해할 수 있게 한국어 3~4문장으로 해석해주세요. "
            f"마크다운 문법 없이 일반 텍스트로 작성하세요.\n\n"
            f"데이터셋: {dataset_name}\n변수: {', '.join(variables)}\n결과 변수: {outcome}\n"
            f"발견된 최적 DAG (BDeu 1위): {_ai_edges}\n{_ai_gt_text}\n"
            f"유효 DAG 수: {len(valid_dags)}개, 1위 BDeu: {scored[0][2]:.2f}\n\n"
            f"1) 발견된 인과 화살표가 각각 무엇을 뜻하는지, 2) 이 구조가 얼마나 신뢰할 수 있는지 설명하세요."
        )
        _cache_key_s = prompt_cache_key("structure", _ai_prompt_structure)
        _local_structure_summary = (
            f"BDeu 1위 구조는 {_ai_edges}입니다. "
            f"{_ai_gt_text.replace(chr(10), ' ')}. "
            "관측 데이터만으로는 Markov equivalence와 숨은 변수 때문에 정답 방향을 완전히 보장할 수 없으므로, "
            "이 결과는 가능한 DAG 후보 중 데이터에 가장 잘 맞는 탐색 결과로 해석해야 합니다."
        )
        if st.button("AI 해석 생성", key="ai_btn_structure"):
            with st.spinner("Groq가 구조를 해석하는 중..."):
                call_groq(groq_api_key, _ai_prompt_structure, _cache_key_s, _local_structure_summary)
        _cached_s = st.session_state.get(groq_state_key(groq_api_key, _cache_key_s))
        if _cached_s:
            render_ai_box(_cached_s)

    # ── 핵심 발견 & 다음 단계 ──
    _finding2_edges = format_edges(best_graph.edges())
    _finding2_f1 = f" (정답 대비 F1={best_metrics['f1']:.2f})" if best_metrics else ""
    st.markdown(
        f"""
        <div class="finding-card">
            <div class="finding-label">이 탭의 핵심</div>
            <div class="finding-body">
            {len(valid_dags):,}개의 유효 DAG 중 데이터에 가장 부합하는 구조는 <b>{_finding2_edges}</b>입니다{_finding2_f1}.
            이 구조는 "{', '.join(variables)}" 사이에서 <b>어떤 변수가 어떤 변수의 원인인지</b>를 보여줍니다.
            하지만 이것만으로는 "그래서 뭘 해야 하는가?"에 답이 되지 않습니다.
            </div>
        </div>
        <div class="nextstep-card">
            <div class="nextstep-label">다음 단계 → 개입 추천</div>
            <div class="nextstep-body">
            발견된 인과 구조를 활용해, <b>{outcome}</b>을 {'높이' if outcome_higher_is_better else '줄이'}기 위해
            어떤 변수에 개입(do-intervention)해야 가장 효과적인지 계산합니다.
            인과 구조를 아는 것의 <b>실질적 가치</b>가 바로 다음 탭에서 드러납니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[3]:
    # ═══ 개입 추천 (THE STAR) ═══
    st.markdown(
        f"""<span class="chapter-num">3</span> <b style="font-size:1.15rem;">어디에 개입해야 {outcome}이 바뀌는가</b>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-band" style="border-left-color: #059669; background: linear-gradient(90deg, #ecfdf5 0%, #f8fafc 100%);">
        인과 구조를 아는 것의 <b>실질적 가치</b>는 바로 여기에 있습니다.
        발견된 DAG를 이용해 <b>결과 변수 <code>{outcome}</code>을 {'높이' if outcome_higher_is_better else '줄이'}려면 어떤 변수에 개입해야 하는지</b>를
        do-calculus (backdoor adjustment)로 추정합니다.
        {'<br><small>연속형 변수는 3분위로 이산화되어 있으므로, 효과 크기는 이산화된 단위 기준입니다.</small>' if auto_discretize else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    dag_options = {"발견된 최적 구조 (고전)": best_graph}
    _grover_for_intv = st.session_state.get("grover_result")
    if _grover_for_intv is not None and st.session_state.get("grover_run_key") == run_key:
        dag_options["Grover 탐색 결과"] = bitstring_to_dag(_grover_for_intv["selected_bitstring"], edge_list)
    if has_ground_truth:
        dag_options["정답 구조 (기준)"] = ground_truth

    choice = st.radio("개입 분석에 사용할 구조", list(dag_options.keys()), horizontal=True)
    chosen_dag = dag_options[choice]
    chosen_dag_title = {
        "발견된 최적 구조 (고전)": "Classical Best DAG",
        "Grover 탐색 결과": "Grover Selected DAG",
        "정답 구조 (기준)": "Ground Truth DAG",
    }.get(choice, "Selected DAG")
    intervention = intervention_table(data, chosen_dag, variables, outcome, outcome_higher_is_better)

    # Main visualization
    int_cols = st.columns([0.85, 1.15])
    with int_cols[0]:
        st.pyplot(draw_dag(chosen_dag, variables, chosen_dag_title, reference=ground_truth if has_ground_truth else None), use_container_width=True)
    with int_cols[1]:
        if intervention.empty:
            st.warning("개입 효과를 계산할 후보 변수가 없습니다.")
        else:
            st.pyplot(plot_interventions(intervention), use_container_width=True)

    if not intervention.empty:
        best_intv = intervention.iloc[0]
        effect_val = best_intv["effect_high_minus_low"]
        if has_actionable_intervention(intervention):
            direction_icon = "+" if effect_val > 0 else "-" if effect_val < 0 else "="
            confidence_label, confidence_bg, confidence_fg = coverage_confidence(float(best_intv["coverage"]))
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #059669 0%, #047857 100%);
                    border-radius: 12px;
                    padding: 1.3rem 1.5rem;
                    margin: 1rem 0;
                    color: #ffffff;
                ">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.85; margin-bottom: 0.3rem;">추천 개입 타겟</div>
                    <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem;">{best_intv['target']} ({direction_icon}{abs(effect_val):.4f})</div>
                    <div style="font-size: 0.95rem; opacity: 0.9;">
                        <span style="display:inline-block;background:{confidence_bg};color:{confidence_fg};border-radius:999px;padding:0.15rem 0.55rem;font-weight:700;margin-right:0.45rem;">
                            {confidence_label} · coverage {best_intv['coverage']:.0%}
                        </span>
                        권장: <b>{best_intv['recommended_action']}</b> &nbsp;|&nbsp;
                        효과 크기: <b>{effect_val:.4f}</b> &nbsp;|&nbsp;
                        방법: {best_intv['method']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #475569 0%, #334155 100%);
                    border-radius: 12px;
                    padding: 1.3rem 1.5rem;
                    margin: 1rem 0;
                    color: #ffffff;
                ">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.85; margin-bottom: 0.3rem;">개입 추천 결과</div>
                    <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem;">추천 가능한 개입 없음</div>
                    <div style="font-size: 0.95rem; opacity: 0.9;">
                        선택한 DAG에서는 후보 변수에서 <b>{outcome}</b>으로 향하는 directed path가 없거나, 추정 효과가 모두 0입니다.
                        정답 구조를 기준으로 보거나 결과 변수를 바꾸면 다른 개입 효과가 나타날 수 있습니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 개입 효과 상세")
        st.dataframe(intervention, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div class="status-band">
            <b>해석 방법</b>: 각 행은 한 변수에 do-intervention을 적용했을 때 <code>{outcome}</code>의 기대값 변화입니다.<br>
            <b>E[{outcome}|do(low)]</b>: 해당 변수를 최솟값으로 고정했을 때 {outcome}의 기대값<br>
            <b>E[{outcome}|do(high)]</b>: 해당 변수를 최댓값으로 고정했을 때 {outcome}의 기대값<br>
            <b>effect</b>: 두 값의 차이. 절대값이 클수록 해당 변수의 개입 효과가 큽니다.<br>
            부모 변수가 있으면 <b>backdoor adjustment</b>로 교란을 보정합니다 (관측 가능한 부모 조합만 사용한 근사치).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "이 개입 분석은 관측 데이터와 발견된 DAG에 기반한 근사입니다. "
        "Positivity 위반(특정 조건 조합에 데이터가 없는 경우) 시 해당 조합은 제외되며 coverage 열에 반영됩니다. "
        "실제 약물 효과나 의료 판단으로 해석하면 안 되며, 프로젝트 데모용 의사결정 보조 지표입니다."
    )

    if ai_enabled and not intervention.empty:
        _ai_edges_intv = format_edges(chosen_dag.edges())
        _has_actionable_intv = has_actionable_intervention(intervention)
        _intv_summary = "\n".join(
            f"- {row['target']}: effect={row['effect_high_minus_low']:.4f}, action={row['recommended_action']}, method={row['method']}"
            for _, row in intervention.iterrows()
        )
        _intv_instructions = (
            "1) 가장 효과적인 개입 타겟은 무엇이고 왜 그런지\n"
            "2) 각 변수의 개입 효과를 직관적으로 해석\n"
            "3) 실제로 이 결과를 어떻게 활용할 수 있는지 한 줄 제안"
            if _has_actionable_intv
            else
            "1) 이 DAG에서 추천 가능한 개입 타겟이 없다고 판단해야 하는 이유\n"
            "2) directed path가 없거나 효과가 0인 결과를 어떻게 해석해야 하는지\n"
            "3) 정답 구조나 결과 변수 변경으로 다시 확인해야 한다는 점"
        )
        _ai_prompt_intv = (
            f"당신은 인과 추론 전문가입니다. 아래 개입 효과 분석 결과를 바탕으로 한국어 전략 보고서를 작성하세요.\n\n"
            f"- 결과 변수({'높이고 싶은 것' if outcome_higher_is_better else '줄이고 싶은 것'}): {outcome}\n"
            f"- 사용한 인과 구조: {_ai_edges_intv}\n"
            f"- 개입 효과 요약:\n{_intv_summary}\n\n"
            f"작성 지침:\n{_intv_instructions}\n\n"
            f"추천 가능한 개입이 없으면 새로운 타겟을 만들어내지 말고, 추천 불가 사유를 명확히 설명하세요. "
            f"'do-calculus', 'backdoor' 같은 용어는 '인과 관계 보정법' 등으로 쉽게 풀어서 설명하세요."
        )
        _cache_key_i = prompt_cache_key("intervention", _ai_prompt_intv)
        if _has_actionable_intv:
            _top_intv = intervention.iloc[0]
            _local_intv_summary = (
                f"가장 큰 개입 효과를 보인 변수는 {_top_intv['target']}입니다. "
                f"권장 행동은 {_top_intv['recommended_action']}이고, 효과 크기는 {_top_intv['effect_high_minus_low']:.4f}입니다. "
                f"coverage는 {_top_intv['coverage']:.0%}로, coverage가 낮을수록 관측 가능한 조건 조합이 부족해 해석을 보수적으로 해야 합니다."
            )
        else:
            _local_intv_summary = (
                f"선택한 DAG에서는 후보 변수에서 {outcome}으로 향하는 directed path가 없거나 추정 효과가 모두 0입니다. "
                "따라서 이 구조 기준으로는 추천 가능한 개입 타겟이 없으며, 정답 구조 또는 다른 결과 변수로 재확인해야 합니다."
            )
        if st.button("AI 해석 생성", key="ai_btn_intervention"):
            with st.spinner("Groq가 개입 결과를 해석하는 중..."):
                call_groq(groq_api_key, _ai_prompt_intv, _cache_key_i, _local_intv_summary)
        _cached_i = st.session_state.get(groq_state_key(groq_api_key, _cache_key_i))
        if _cached_i:
            render_ai_box(_cached_i)

    # ── 핵심 발견 & 다음 단계 ──
    if _intro_has_intv:
        _finding3_text = (
            f"발견된 인과 구조 기준으로, <b>{outcome}</b>에 가장 큰 영향을 주는 변수는 "
            f"<b>{_intro_top['target']}</b>이며, 권장 행동은 <b>{_intro_top['recommended_action']}</b>입니다. "
            f"이것이 바로 상관관계가 아닌 <b>인과관계</b>를 알아야만 내릴 수 있는 판단입니다."
        )
    else:
        _finding3_text = (
            f"현재 선택한 DAG 기준으로는 추천 가능한 개입 타겟이 없습니다. "
            f"이는 후보 변수에서 {outcome}으로 향하는 인과 경로가 없거나, 효과가 0이라는 뜻입니다. "
            f"정답 구조를 기준으로 보거나 결과 변수를 바꾸면 다른 결과가 나올 수 있습니다."
        )
    st.markdown(
        f"""
        <div class="finding-card">
            <div class="finding-label">이 탭의 핵심</div>
            <div class="finding-body">{_finding3_text}</div>
        </div>
        <div class="nextstep-card">
            <div class="nextstep-label">다음 단계 → 양자적 접근</div>
            <div class="nextstep-body">
            지금까지 고전적 방법(전수 탐색)으로 최적 DAG를 찾았습니다.
            다음 탭에서는 같은 문제를 <b>양자 컴퓨팅(Grover 알고리즘)</b>으로 접근합니다.
            {n_total:,}개 후보를 모두 검사하는 대신, 양자 간섭을 이용해 좋은 구조를 <b>더 높은 확률로</b> 찾아내는 실험입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[4]:
    # ═══ 양자적 접근 ═══
    st.markdown(
        f"""<span class="chapter-num">4</span> <b style="font-size:1.15rem;">양자 컴퓨팅으로 같은 문제에 도전하다</b>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-band">
        앞선 탭에서 <b>{len(valid_dags):,}개</b> DAG를 하나씩 전수 검사하여 최적 구조를 찾았습니다.
        하지만 변수가 늘어나면 이 방법은 현실적으로 불가능해집니다.
        <b>Grover 알고리즘</b>은 이런 비정렬 탐색 문제에서 검색 횟수를 제곱근 수준으로 줄여주는 양자 알고리즘입니다.<br><br>
        <b>작동 원리</b>: {n_edges}개 엣지 후보 각각을 큐비트 1개로 인코딩 → 모든 후보를 양자 중첩 상태로 동시에 준비 →
        BDeu 상위 <b>{top_k_effective}개</b> DAG를 "정답"으로 표시(Oracle) →
        양자 간섭으로 정답의 측정 확률을 증폭(Diffuser) → 측정하면 좋은 구조가 높은 확률로 나옴.
        <br><b>아래 버튼을 눌러 직접 실행해보세요.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    qiskit_ok, qiskit_error = check_qiskit()
    if not qiskit_ok:
        st.error(f"Qiskit 또는 qiskit-aer를 불러올 수 없습니다: {qiskit_error}")
    else:
        run_cols = st.columns([0.15, 0.15, 0.2, 0.5])
        with run_cols[0]:
            run_pressed = st.button("Grover 실행", type="primary", use_container_width=True)
        with run_cols[1]:
            n_runs = int(st.number_input("Multi-run", min_value=1, max_value=10, value=3, help="여러 번 실행해서 가장 좋은 결과를 선택합니다."))
        with run_cols[2]:
            use_penalty = st.toggle("Penalty Oracle", value=False, help="순환 DAG에 부분 위상 패널티를 적용하여 유효 DAG 측정률을 높입니다.")
        with run_cols[3]:
            _penalty_note = " + Penalty Oracle (순환 DAG 억제)" if use_penalty else ""
            st.markdown(
                f"<p class='small-note'>반복 {grover_iteration_count(n_edges, top_k_effective)}회 x {n_runs} runs{_penalty_note}. "
                "여러 번 실행하면 측정 분산을 줄여 더 안정적인 결과를 얻습니다.</p>",
                unsafe_allow_html=True,
            )

        # cyclic bitstrings 계산 (Penalty Oracle용)
        _cyclic_bitstrings = [
            format(i, f"0{n_edges}b")
            for i in range(n_total)
            if format(i, f"0{n_edges}b") not in valid_bitstrings
        ]

        if run_pressed:
            _run_label = "Penalty Oracle Grover" if use_penalty else "Grover"
            with st.spinner(f"Aer 시뮬레이터에서 {_run_label} 회로를 {n_runs}회 실행 중입니다."):
                best_result = None
                best_good_prob = -1.0
                all_run_probs = []
                for run_i in range(n_runs):
                    if use_penalty:
                        result = run_grover_search_with_penalty(
                            n_edges, good_bitstrings, _cyclic_bitstrings, shots=shots,
                        )
                    else:
                        result = run_grover_search(n_edges, good_bitstrings, shots=shots)
                    enriched = enrich_grover_result(result, valid_bitstrings, scored)
                    all_run_probs.append(enriched["good_probability"])
                    if enriched["good_probability"] > best_good_prob:
                        best_good_prob = enriched["good_probability"]
                        best_result = enriched
                best_result["all_run_probs"] = all_run_probs
                best_result["n_runs"] = n_runs
                best_result["used_penalty"] = use_penalty
            st.session_state["grover_result"] = best_result
            st.session_state["grover_run_key"] = run_key

    grover_result = st.session_state.get("grover_result")
    if grover_result is not None:
        if st.session_state.get("grover_run_key") != run_key:
            st.warning("현재 설정이 마지막 Grover 실행 때와 다릅니다. 정확한 비교를 위해 다시 실행하세요.")

        selected_bitstring = grover_result["selected_bitstring"]
        grover_graph = bitstring_to_dag(selected_bitstring, edge_list)
        grover_metrics = graph_metrics(ground_truth if has_ground_truth else None, grover_graph)

        compare_cols = st.columns(3 if has_ground_truth else 2)
        if has_ground_truth:
            with compare_cols[0]:
                st.pyplot(draw_dag(ground_truth, variables, "Ground Truth DAG"), use_container_width=True)
        with compare_cols[-2]:
            subtitle = f"rank 1, {format_edges(best_graph.edges())}"
            if best_metrics:
                subtitle = f"SHD={best_metrics['shd']}, F1={best_metrics['f1']:.2f}"
            st.pyplot(
                draw_dag(best_graph, variables, "Classical Result", reference=ground_truth if has_ground_truth else None, subtitle=subtitle),
                use_container_width=True,
            )
        with compare_cols[-1]:
            subtitle = f"rank {grover_result.get('selected_rank')}, {selected_bitstring}"
            if grover_metrics:
                subtitle = f"SHD={grover_metrics['shd']}, F1={grover_metrics['f1']:.2f}"
            st.pyplot(
                draw_dag(grover_graph, variables, "Grover Result", reference=ground_truth if has_ground_truth else None, subtitle=subtitle),
                use_container_width=True,
            )

        result_cols = st.columns(6)
        result_cols[0].metric("Grover 반복", grover_result["n_iterations"])
        result_cols[1].metric("Oracle 적중률", f"{grover_result['good_probability'] * 100:.1f}%")
        result_cols[2].metric("유효 DAG 측정률", f"{grover_result['valid_probability'] * 100:.1f}%")
        result_cols[3].metric("선택 DAG 순위", grover_result.get("selected_rank") or "N/A")
        result_cols[4].metric("회로 깊이", grover_result["circuit_depth"])
        if grover_result.get("n_runs", 1) > 1:
            avg_prob = np.mean(grover_result.get("all_run_probs", [grover_result["good_probability"]]))
            result_cols[5].metric("Multi-run", f"{grover_result['n_runs']}회", f"avg {avg_prob*100:.1f}%")
        else:
            result_cols[5].metric("실행 시간", f"{grover_result['elapsed_time']:.3f}s")

        st.pyplot(plot_grover_counts(grover_result["counts"], good_bitstrings, valid_bitstrings), use_container_width=True)

        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.markdown("**Oracle 타겟 bitstring**")
            st.dataframe(candidate_table(scored[:top_k_effective], ground_truth if has_ground_truth else None, top_k_effective), use_container_width=True, hide_index=True)
        with detail_cols[1]:
            st.markdown("**회로 및 후처리 요약**")
            uniform_p = top_k_effective / (2**n_edges) * 100
            summary_items = [
                {"item": "Qubits", "value": grover_result["n_qubits"]},
                {"item": "Shots", "value": f"{grover_result['shots']:,}"},
                {"item": "Elapsed", "value": f"{grover_result['elapsed_time']:.4f}s"},
                {"item": "Raw top bitstring", "value": grover_result["top_bitstring"]},
                {"item": "Raw top is valid DAG", "value": grover_result["raw_top_is_valid"]},
                {"item": "Selected (score-weighted)", "value": grover_result["selected_bitstring"]},
                {"item": "Selected BDeu", "value": round(grover_result.get("selected_score") or 0, 3)},
                {"item": "Uniform baseline", "value": f"{uniform_p:.2f}%"},
                {"item": "Amplification", "value": f"{grover_result['good_probability']*100/uniform_p:.1f}x" if uniform_p > 0 else "N/A"},
            ]
            st.dataframe(pd.DataFrame(summary_items), use_container_width=True, hide_index=True)

        with st.expander("양자 회로 보기"):
            try:
                circuit_fig = grover_result["circuit"].draw(output="mpl", fold=60)
                st.pyplot(circuit_fig, use_container_width=True)
                plt.close(circuit_fig)
            except Exception:
                st.code(str(grover_result["circuit"].draw(output="text", fold=100)))

        # ═══ 양자 AI 해석 ═══
        if ai_enabled:
            st.divider()
            _ai_prompt_quantum = (
                f"당신은 양자 컴퓨팅 및 Qiskit 전문가입니다. 아래의 Grover 알고리즘 기반 인과 구조 탐색 결과를 해석해주세요.\n\n"
                f"### 분석 데이터:\n"
                f"- 사용 큐비트 수: {n_edges} (각 엣지 후보당 1큐비트)\n"
                f"- Oracle 타겟 수: {top_k_effective} (BDeu 상위 {top_k_effective}개 DAG)\n"
                f"- Grover 반복 횟수: {grover_result['n_iterations']}\n"
                f"- Oracle 적중률: {grover_result['good_probability']*100:.1f}% (이론적 증폭 성공 여부)\n"
                f"- 선택된 DAG 순위: {grover_result.get('selected_rank', '?')}위\n\n"
                f"### 요청 사항:\n"
                f"1. Qiskit Aer 시뮬레이터에서 Grover 회로가 의도대로 진폭 증폭(Amplitude Amplification)을 수행했는지 평가하세요.\n"
                f"2. 전수조사 대비 Grover 알고리즘의 복잡도 이점($O(\\sqrt{{N}})$)이 이 문제에서 어떻게 나타나는지 설명하세요.\n"
                f"3. 측정 분포 차트에서 나타나는 'Good' 상태와 'Valid' 상태의 차이가 무엇을 의미하는지 비전문가에게 설명하세요.\n"
                f"4. 이 양자적 시도가 인과 추론 분야에서 어떤 방법론적 혁신을 보여주는지 데이터 과학적 관점에서 정리하세요.\n\n"
                f"격식 있고 전문적인 톤으로 작성하세요."
            )
            _cache_key_q = prompt_cache_key("quantum", _ai_prompt_quantum)
            _local_q_summary = (
                f"Qiskit Aer 시뮬레이터를 통해 {n_edges}큐비트 Grover 회로를 실행했습니다. "
                f"Oracle 적중률은 {grover_result['good_probability']*100:.1f}%로, 균등 중첩 상태 대비 진폭이 성공적으로 증폭되었습니다. "
                f"Grover 알고리즘은 가능한 모든 DAG 후보를 하나씩 검사하는 대신, 양자 간섭을 이용해 고득점 후보를 더 높은 확률로 찾아낼 수 있음을 보여줍니다."
            )
            if st.button("양자 결과 AI 해석 생성", key="ai_btn_quantum"):
                with st.spinner("Groq가 양자 실험 결과를 분석 중..."):
                    call_groq(groq_api_key, _ai_prompt_quantum, _cache_key_q, _local_q_summary)
            _cached_q = st.session_state.get(groq_state_key(groq_api_key, _cache_key_q))
            if _cached_q:
                render_ai_box(_cached_q)
    else:
        st.info("위 버튼을 눌러 Grover 실험을 실행하세요. 측정 분포와 고전 결과 비교가 생성됩니다.")

    st.divider()

    st.markdown("#### Grover vs Classical 복잡도")
    st.pyplot(plot_complexity_comparison(n_edges, top_k_effective), use_container_width=True)

    with st.expander("한계점과 구현된 대안"):
        st.markdown(
            """
            | 한계 | 설명 | 이 앱에서 구현한 대안 |
            |---|---|---|
            | **Pre-computed Oracle** | BDeu 점수를 고전적으로 미리 계산해서 Oracle에 하드코딩 | **QAOA 비교 실험** — 점수를 cost Hamiltonian으로 인코딩하여 사전 계산 없이 최적화하는 방향성 제시 (아래 섹션) |
            | **순환 DAG 포함** | Grover가 전체 $2^n$ 공간을 탐색하므로 비유효 DAG도 측정됨 | **Penalty Oracle** — 순환 DAG에 부분 위상 패널티를 적용하여 유효 DAG 측정률 향상 (위 토글로 실행 가능) |
            | **이산화 정보 손실** | 연속형 변수를 이산화하면 구조 식별력 저하 | **BGe 점수** — 연속 데이터에 직접 적용되는 Gaussian 점수 함수 (사이드바에서 선택 가능) |
            | **Markov equivalence** | 관측 데이터만으로는 동일 조건부 독립 DAG를 구분 불가 | 현재 미해결 — 개입 데이터 확보 또는 FCI 알고리즘이 필요 (향후 과제) |

            **Qiskit 프로젝트의 의의:**
            - **정식화**: 인과 구조 탐색을 Grover ($|s\\rangle \\to Oracle \\to Diffuser \\to Measure$)에 성공적으로 매핑
            - **확장성**: $O(N) \\to O(\\sqrt{N})$ 이차 속도 향상, 변수가 늘수록 이점 확대
            - **대안 시연**: Penalty Oracle과 QAOA를 통해 한계를 인식하고 개선 방향을 실제 코드로 시연
            """
        )

    # ── QAOA 비교 실험 ──
    st.divider()
    st.markdown("#### QAOA 비교 실험: Pre-computed Oracle 한계의 대안")
    st.markdown(
        f"""
        <div class="status-band" style="border-left-color: #7c3aed; background: linear-gradient(90deg, #f5f3ff 0%, #f8fafc 100%);">
        Grover는 "좋은 답"을 미리 알아야(Pre-computed Oracle) 하지만,
        <b>QAOA</b>는 점수 함수 자체를 양자 회로에 인코딩합니다.
        BDeu 점수를 cost Hamiltonian의 위상 회전으로 매핑하고,
        mixer 연산과 번갈아 적용하면서 파라미터를 최적화하여 고득점 DAG를 찾습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    _qaoa_cols = st.columns([0.2, 0.2, 0.6])
    with _qaoa_cols[0]:
        qaoa_pressed = st.button("QAOA 실행", type="primary", use_container_width=True)
    with _qaoa_cols[1]:
        qaoa_layers = int(st.number_input("QAOA layers (p)", min_value=1, max_value=5, value=2))
    with _qaoa_cols[2]:
        st.markdown(
            f"<p class='small-note'>scored DAG {len(scored)}개의 점수를 cost operator로 인코딩, "
            f"gamma/beta 파라미터를 grid search로 최적화합니다. p={qaoa_layers} layers.</p>",
            unsafe_allow_html=True,
        )

    if qaoa_pressed:
        with st.spinner(f"QAOA {qaoa_layers}-layer 회로를 실행하고 파라미터를 최적화하는 중..."):
            _qaoa_result = run_qaoa_search(n_edges, scored, p_layers=qaoa_layers, shots=shots)
            _qaoa_enriched = enrich_grover_result(_qaoa_result, valid_bitstrings, scored)
            _qaoa_enriched["qaoa"] = True
            _qaoa_enriched["p_layers"] = _qaoa_result.get("p_layers", qaoa_layers)
            _qaoa_enriched["best_gamma"] = _qaoa_result.get("best_gamma", 0)
            _qaoa_enriched["best_beta"] = _qaoa_result.get("best_beta", 0)
            _qaoa_enriched["optimization_evals"] = _qaoa_result.get("optimization_evals", 0)
            _qaoa_enriched["best_expectation"] = _qaoa_result.get("best_expectation", 0)
        st.session_state["qaoa_result"] = _qaoa_enriched
        st.session_state["qaoa_run_key"] = run_key

    _qaoa_r = st.session_state.get("qaoa_result")
    if _qaoa_r is not None and st.session_state.get("qaoa_run_key") == run_key:
        _qaoa_graph = bitstring_to_dag(_qaoa_r["selected_bitstring"], edge_list)
        _qaoa_metrics = graph_metrics(ground_truth if has_ground_truth else None, _qaoa_graph)

        qaoa_metric_cols = st.columns(6)
        qaoa_metric_cols[0].metric("QAOA Layers", _qaoa_r.get("p_layers", "?"))
        qaoa_metric_cols[1].metric("Oracle 적중률", f"{_qaoa_r['good_probability']*100:.1f}%")
        qaoa_metric_cols[2].metric("유효 DAG 측정률", f"{_qaoa_r['valid_probability']*100:.1f}%")
        qaoa_metric_cols[3].metric("선택 DAG 순위", _qaoa_r.get("selected_rank") or "N/A")
        qaoa_metric_cols[4].metric("회로 깊이", _qaoa_r["circuit_depth"])
        qaoa_metric_cols[5].metric("최적화 평가 횟수", _qaoa_r.get("optimization_evals", "?"))

        st.pyplot(plot_grover_counts(_qaoa_r["counts"], good_bitstrings, valid_bitstrings), use_container_width=True)

        # Grover vs QAOA 비교 테이블
        _grover_for_compare = st.session_state.get("grover_result")
        if _grover_for_compare is not None and st.session_state.get("grover_run_key") == run_key:
            st.markdown("##### Grover vs QAOA 직접 비교")
            _compare_data = {
                "항목": ["알고리즘", "Oracle 적중률", "유효 DAG 측정률", "선택 DAG 순위", "회로 깊이", "실행 시간"],
                "Grover": [
                    "Penalty Oracle" if _grover_for_compare.get("used_penalty") else "Standard",
                    f"{_grover_for_compare['good_probability']*100:.1f}%",
                    f"{_grover_for_compare['valid_probability']*100:.1f}%",
                    str(_grover_for_compare.get("selected_rank", "N/A")),
                    str(_grover_for_compare["circuit_depth"]),
                    f"{_grover_for_compare['elapsed_time']:.3f}s",
                ],
                "QAOA": [
                    f"p={_qaoa_r.get('p_layers', '?')} layers",
                    f"{_qaoa_r['good_probability']*100:.1f}%",
                    f"{_qaoa_r['valid_probability']*100:.1f}%",
                    str(_qaoa_r.get("selected_rank", "N/A")),
                    str(_qaoa_r["circuit_depth"]),
                    f"{_qaoa_r['elapsed_time']:.3f}s",
                ],
            }
            st.dataframe(pd.DataFrame(_compare_data), use_container_width=True, hide_index=True)

    # ── 핵심 발견 & 다음 단계 ──
    _grover_done = st.session_state.get("grover_result") is not None and st.session_state.get("grover_run_key") == run_key
    if _grover_done:
        _gr = st.session_state["grover_result"]
        _uniform_pct = top_k_effective / (2**n_edges) * 100
        _amp_ratio = _gr["good_probability"] * 100 / _uniform_pct if _uniform_pct > 0 else 0
        _finding4_text = (
            f"Grover 회로 실행 결과, Oracle 적중률은 <b>{_gr['good_probability']*100:.1f}%</b>로 "
            f"균등 분포 기준 <b>{_uniform_pct:.1f}%</b> 대비 약 <b>{_amp_ratio:.1f}배</b> 증폭되었습니다. "
            f"이는 양자 간섭을 통해 고득점 DAG가 측정될 확률이 유의미하게 높아졌다는 뜻입니다."
        )
    else:
        _finding4_text = (
            "아직 Grover 회로를 실행하지 않았습니다. "
            "위의 <b>'Grover 실행'</b> 버튼을 눌러 양자 탐색을 직접 실행해보세요. "
            "실행 후 진폭 증폭 효과를 측정 분포 차트에서 확인할 수 있습니다."
        )
    st.markdown(
        f"""
        <div class="finding-card">
            <div class="finding-label">이 탭의 핵심</div>
            <div class="finding-body">{_finding4_text}</div>
        </div>
        <div class="nextstep-card">
            <div class="nextstep-label">다음 단계 → 종합 분석</div>
            <div class="nextstep-body">
            마지막 탭에서 고전 탐색과 양자 탐색의 결과를 나란히 비교하고,
            인과 구조 발견 → 개입 추천 → 양자 가속까지 전체 파이프라인의 종합 판단을 내립니다.
            {f'Groq API를 연결했다면 AI 종합 보고서도 생성할 수 있습니다.' if ai_enabled else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[5]:
    # ═══ 종합 분석 ═══
    st.markdown(
        f"""<span class="chapter-num">5</span> <b style="font-size:1.15rem;">종합 판단 — 무엇을 알았고, 무엇을 해야 하는가</b>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-band" style="border-left-color: #059669; background: linear-gradient(90deg, #ecfdf5 0%, #f8fafc 100%);">
        지금까지의 분석을 정리합니다.
        <b>{dataset_name}</b> 데이터에서 인과 구조를 발견하고(2탭), 개입 타겟을 추천하고(3탭),
        양자 알고리즘으로 탐색 가속을 시연했습니다(4탭).
        아래에서 고전 vs 양자 성능을 나란히 비교하고, 전체 분석의 결론을 확인하세요.
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_grover_r = st.session_state.get("grover_result")
    grover_active = active_grover_r is not None and st.session_state.get("grover_run_key") == run_key

    if grover_active:
        grover_graph_r = bitstring_to_dag(active_grover_r["selected_bitstring"], edge_list)
        grover_metrics_r = graph_metrics(ground_truth if has_ground_truth else None, grover_graph_r)
    else:
        grover_graph_r = None
        grover_metrics_r = None

    # ── Gauges ──
    if has_ground_truth:
        gauge_cols = st.columns(4 if grover_active else 3)
        with gauge_cols[0]:
            st.pyplot(plot_gauge(
                best_metrics["f1"] if best_metrics else 0, "Classical F1",
                color_thresholds=[(0.4, "#ef4444"), (0.7, "#f59e0b"), (1.01, "#059669")],
            ), use_container_width=True)
        with gauge_cols[1]:
            shd_val = best_metrics["shd"] if best_metrics else 0
            n_gt_edges = len(ground_truth.edges())
            st.pyplot(plot_gauge(
                shd_val, "SHD (lower=better)", max_val=max(n_gt_edges * 2, 1),
                color_thresholds=[(0.2, "#059669"), (0.5, "#f59e0b"), (1.01, "#ef4444")],
            ), use_container_width=True)
        if grover_active:
            with gauge_cols[2]:
                st.pyplot(plot_gauge(
                    grover_metrics_r["f1"] if grover_metrics_r else 0, "Grover F1",
                    color_thresholds=[(0.4, "#ef4444"), (0.7, "#f59e0b"), (1.01, "#059669")],
                ), use_container_width=True)
            with gauge_cols[3]:
                uniform_p_r = top_k_effective / (2 ** n_edges)
                amp_r = active_grover_r["good_probability"] / uniform_p_r if uniform_p_r > 0 else 0
                st.pyplot(plot_gauge(
                    amp_r, "Amplification", max_val=max(amp_r * 1.3, 10),
                    color_thresholds=[(0.3, "#ef4444"), (0.6, "#f59e0b"), (1.01, "#059669")],
                ), use_container_width=True)
        else:
            with gauge_cols[2]:
                st.pyplot(plot_gauge(0, "Grover F1\nRun first",
                    color_thresholds=[(1.01, "#e2e8f0")],
                ), use_container_width=True)
    else:
        st.info("정답 구조가 있는 데이터셋에서 구조 정확도 게이지가 표시됩니다.")

    st.divider()

    # ── Radar + Comparison ──
    if grover_active:
        st.markdown("#### 고전 vs 양자 탐색 성능 비교")
        radar_cols = st.columns([1.1, 0.9])
        with radar_cols[0]:
            st.pyplot(plot_radar_comparison(
                best_metrics, grover_metrics_r, active_grover_r, n_edges, top_k_effective,
            ), use_container_width=True)
        with radar_cols[1]:
            compare_rows = []
            dims = [
                ("Structure F1", best_metrics.get("f1", 0) if best_metrics else 0,
                 grover_metrics_r.get("f1", 0) if grover_metrics_r else 0),
                ("SHD Error", best_metrics.get("shd", "-") if best_metrics else "-",
                 grover_metrics_r.get("shd", "-") if grover_metrics_r else "-"),
                ("Search Depth", f"{len(valid_dags):,}", f"{active_grover_r['n_iterations']:,}"),
                ("BDeu Score", round(best_score, 2), round(active_grover_r.get("selected_score", 0) or 0, 2)),
                ("Amplification", "1.0x", f"{active_grover_r['good_probability']/(top_k_effective/2**n_edges):.1f}x"),
            ]
            for name, c_val, g_val in dims:
                compare_rows.append({"Metric": name, "Classical (Exhaustive)": c_val, "Quantum (Grover)": g_val})
            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Qiskit Grover 파이프라인 확률 증폭 현황")
        st.pyplot(plot_amplification_waterfall(active_grover_r, n_edges, top_k_effective), use_container_width=True)
        st.divider()

    # ── Score landscape ──
    st.markdown("#### BDeu Score Landscape 및 데이터 요약")
    explain_cols = st.columns([1.1, 0.9])
    with explain_cols[0]:
        st.pyplot(plot_score_landscape(scored, top_k_effective), use_container_width=True)
    with explain_cols[1]:
        st.markdown(
            f"""
            | 항목 | 값 |
            |---|---|
            | 유효 DAG 후보 수 | **{len(valid_dags):,}**개 |
            | 1위 구조 BDeu 점수 | **{scored[0][2]:.2f}** |
            | 결과 변수 | **{outcome}** |
            | 분석 모델 | **Qiskit-based Grover Search** |
            """
        )

    st.divider()

    # ── AI Synthesis ──
    if ai_enabled:
        st.markdown("#### 인과 추론 및 양자 분석 최종 보고서")
        _ai_edges_syn = format_edges(best_graph.edges())
        _ai_gt_syn = f"정답 구조: {format_edges(ground_truth.edges())}, F1={best_metrics['f1']:.2f}" if has_ground_truth and best_metrics else "정답 구조 없음"
        _ai_grover_syn = "Grover 미실행"
        if grover_active:
            _ai_grover_syn = (
                f"Grover 결과: {n_edges}큐비트 회로, Oracle 적중률 {active_grover_r['good_probability']*100:.1f}%, "
                f"증폭률 {active_grover_r['good_probability']/(top_k_effective/2**n_edges):.1f}배"
            )
        _ai_intv_syn = "개입 효과를 계산할 후보 변수가 없음"
        _decision_instruction = "개입 후보가 없으므로 추천 타겟을 새로 만들지 말고 한계를 설명"
        _intv_preview = intervention_table(data, best_graph, variables, outcome, outcome_higher_is_better)
        if not _intv_preview.empty:
            if has_actionable_intervention(_intv_preview):
                _top = _intv_preview.iloc[0]
                _ai_intv_syn = f"핵심 개입 타겟: {_top['target']} ({_top['recommended_action']}, 효과={_top['effect_high_minus_low']:.4f})"
                _decision_instruction = f"{outcome}을 {'높이기' if outcome_higher_is_better else '낮추기'} 위해 {_ai_intv_syn} 조치가 필요한 이유와 기대 효과"
            else:
                _ai_intv_syn = "선택한 DAG 기준 추천 가능한 개입 없음(effect=0 또는 directed path 없음)"
                _decision_instruction = "추천 가능한 개입이 없다는 판단 근거와, 정답 구조 또는 결과 변수 변경으로 재확인해야 한다는 점"

        _ai_prompt_syn = (
            f"당신은 인과 추론과 양자 컴퓨팅(Qiskit) 전문가입니다. 이 프로젝트의 전체 결과를 종합하여 최종 분석 보고서를 작성하세요.\n\n"
            f"### 보고서 구성 항목:\n"
            f"1. **데이터 기반 인과 구조 분석**: {dataset_name} 데이터에서 발견된 최적의 인과 관계({_ai_edges_syn})가 도메인 관점에서 어떤 의미를 갖는지 설명\n"
            f"2. **양자 알고리즘(Grover)의 역할**: Qiskit을 이용한 Grover 탐색이 인과 구조 탐색이라는 비정렬 탐색 문제에 어떻게 적용되었으며, {_ai_grover_syn} 결과가 갖는 방법론적 의미 설명\n"
            f"3. **의사결정 제언**: {_decision_instruction}\n"
            f"4. **기술적 의의와 향후 전망**: 인과 추론과 양자 알고리즘을 결합한 이 시도가 데이터 과학 분야에서 어떤 가능성을 제시하는지 정리\n\n"
            f"### 참고 데이터:\n"
            f"- 정답 구조 일치도: {(_ai_gt_syn)}\n- 분석 모델: Qiskit Aer Simulator (Grover Search)\n"
            f"- 결과 변수 전략: {'증가' if outcome_higher_is_better else '감소'} 목표\n\n"
            f"격식 있고 전문적인 리포트 문체(~입니다, ~함)를 사용하며, 전문 용어는 쉽게 풀어서 서술하세요."
        )
        _cache_key_syn = prompt_cache_key("synthesis", _ai_prompt_syn)
        _local_syn_summary = (
            f"{dataset_name} 데이터 분석 결과, {outcome}에 대한 DAG 후보를 점수화했습니다. "
            f"특히 Qiskit Grover 알고리즘을 통해 수많은 DAG 후보 중 고득점 구조를 양자적으로 탐색할 수 있음을 확인했습니다. "
            f"개입 판단은 {_ai_intv_syn}입니다."
        )

        if st.button("AI 종합 보고서 생성", key="ai_btn_synthesis"):
            with st.spinner("Groq가 전체 분석을 종합 중..."):
                call_groq(groq_api_key, _ai_prompt_syn, _cache_key_syn, _local_syn_summary)
        _cached_syn = st.session_state.get(groq_state_key(groq_api_key, _cache_key_syn))
        if _cached_syn:
            render_ai_box(_cached_syn)
    else:
        st.markdown("#### Key Findings (Local)")
        insights = generate_interpretation(
            has_gt=has_ground_truth,
            best_metrics=best_metrics,
            grover_result=active_grover_r if grover_active else None,
            grover_metrics=grover_metrics_r,
            n_edges=n_edges,
            top_k=top_k_effective,
            scored=scored,
            variables=variables,
            dataset_name=dataset_name,
        )
        for color, title, body in insights:
            st.markdown(
                f"""
                <div style="border-left: 4px solid {color}; background: {color}08; padding: 0.9rem 1.2rem; border-radius: 0 10px 10px 0; margin-bottom: 0.7rem;">
                    <div style="font-weight: 700; color: {color}; font-size: 0.95rem;">{title}</div>
                    <div style="color: #334155; font-size: 0.88rem;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 전체 결론 ──
    st.divider()
    _concl_intv_text = (
        f"개입 추천: <b>{_intro_top['target']}</b>에 <b>{_intro_top['recommended_action']}</b> 조치가 가장 효과적"
        if _intro_has_intv
        else "현재 구조 기준으로는 명확한 개입 타겟이 도출되지 않았으며, 추가 분석이 필요"
    )
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
            border-radius: 16px;
            padding: 2rem 2.2rem;
            margin: 0.5rem 0;
            color: #e2e8f0;
        ">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: #a5b4fc; margin-bottom: 0.5rem; font-weight: 700;">전체 분석 결론</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.8rem; line-height: 1.4;">
                {dataset_name} — {outcome}을 {'높이' if outcome_higher_is_better else '줄이'}기 위한 인과 기반 의사결정
            </div>
            <div style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.7;">
                <b style="color:#a5b4fc;">1. 인과 구조</b>: {len(valid_dags):,}개 후보 중 <b>{format_edges(best_graph.edges())}</b>가 데이터에 가장 부합{f' (F1={best_metrics["f1"]:.2f})' if best_metrics else ''}.<br>
                <b style="color:#a5b4fc;">2. 개입 전략</b>: {_concl_intv_text}.<br>
                <b style="color:#a5b4fc;">3. 양자 시도</b>: {n_edges}큐비트 Grover 회로로 인과 구조 탐색을 양자 도메인에 정식화. 전수조사 O({n_total:,}) 대비 O({int(math.sqrt(n_total))}) 이차 속도 향상의 가능성을 시연.<br>
                <b style="color:#a5b4fc;">4. 제작 의도</b>: 이 시스템은 "상관관계를 넘어 인과관계를 기반으로 의사결정을 내리는 것"의 가치를 보여주기 위해 제작되었습니다.
                양자 알고리즘을 실제 데이터 과학 문제에 접목한 방법론적 시도이자, 비전문가도 인과 추론 결과를 이해하고 활용할 수 있도록 설계된 의사결정 지원 도구입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
