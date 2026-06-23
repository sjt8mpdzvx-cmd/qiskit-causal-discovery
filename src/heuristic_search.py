"""Scalable score-based DAG search with cached local scores.

The exhaustive search used by the teaching demo becomes impractical beyond
four variables.  This module keeps the same decomposable BDeu/BGe objectives
but limits parent-set size and performs deterministic greedy hill climbing.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx

from .dag_utils import dag_to_bitstring, get_edge_list
from .scoring import compute_local_bdeu, compute_local_bge


def precompute_limited_local_scores(
    data,
    variables,
    max_parents: int = 3,
    scoring_method: str = "bdeu",
    equivalent_sample_size: int = 10,
):
    """Cache every local score allowed by ``max_parents``.

    This evaluates ``|V| * sum(C(|V|-1, k))`` parent sets instead of every
    whole DAG.  The returned cache is also used to score moves in O(1).
    """
    if max_parents < 0:
        raise ValueError("max_parents must be non-negative")
    if scoring_method not in {"bdeu", "bge"}:
        raise ValueError("scoring_method must be 'bdeu' or 'bge'")

    local_scores = {}
    for node in variables:
        candidates = [candidate for candidate in variables if candidate != node]
        for parent_count in range(min(max_parents, len(candidates)) + 1):
            for parents in combinations(candidates, parent_count):
                parent_set = frozenset(parents)
                if scoring_method == "bge":
                    score = compute_local_bge(data, node, list(parent_set))
                else:
                    score = compute_local_bdeu(
                        data,
                        node,
                        list(parent_set),
                        equivalent_sample_size,
                    )
                local_scores[(node, parent_set)] = score
    return local_scores


def hill_climb_search(
    data,
    variables,
    max_parents: int = 3,
    scoring_method: str = "bdeu",
    equivalent_sample_size: int = 10,
):
    """Learn one DAG with deterministic add/remove/reverse hill climbing."""
    variables = list(variables)
    local_scores = precompute_limited_local_scores(
        data,
        variables,
        max_parents=max_parents,
        scoring_method=scoring_method,
        equivalent_sample_size=equivalent_sample_size,
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(variables)

    def local_score(node, parents):
        return local_scores[(node, frozenset(parents))]

    current_score = sum(local_score(node, ()) for node in variables)
    evaluations = 0
    trace = []
    tolerance = 1e-10

    while True:
        best_move = None
        best_delta = tolerance

        def consider(move, delta):
            nonlocal best_move, best_delta, evaluations
            evaluations += 1
            # Stable tie-breaking makes runs reproducible without a random seed.
            if delta > best_delta + tolerance or (
                abs(delta - best_delta) <= tolerance
                and best_move is not None
                and move < best_move
            ):
                best_move, best_delta = move, delta

        for source in variables:
            for target in variables:
                if source == target:
                    continue
                if graph.has_edge(source, target):
                    old_parents = set(graph.predecessors(target))
                    new_parents = old_parents - {source}
                    consider(
                        ("remove", source, target),
                        local_score(target, new_parents) - local_score(target, old_parents),
                    )

                    # Reverse source -> target into target -> source.
                    source_parents = set(graph.predecessors(source))
                    if (
                        target not in source_parents
                        and len(source_parents) < max_parents
                    ):
                        candidate = graph.copy()
                        candidate.remove_edge(source, target)
                        candidate.add_edge(target, source)
                        if nx.is_directed_acyclic_graph(candidate):
                            delta = (
                                local_score(target, new_parents) - local_score(target, old_parents)
                                + local_score(source, source_parents | {target})
                                - local_score(source, source_parents)
                            )
                            consider(("reverse", source, target), delta)
                elif len(list(graph.predecessors(target))) < max_parents:
                    candidate = graph.copy()
                    candidate.add_edge(source, target)
                    if nx.is_directed_acyclic_graph(candidate):
                        old_parents = set(graph.predecessors(target))
                        new_parents = old_parents | {source}
                        consider(
                            ("add", source, target),
                            local_score(target, new_parents) - local_score(target, old_parents),
                        )

        if best_move is None:
            break

        operation, source, target = best_move
        if operation == "add":
            graph.add_edge(source, target)
        elif operation == "remove":
            graph.remove_edge(source, target)
        else:
            graph.remove_edge(source, target)
            graph.add_edge(target, source)
        current_score += best_delta
        trace.append(
            {
                "step": len(trace) + 1,
                "move": operation,
                "edge": f"{source}->{target}",
                "score_gain": best_delta,
                "score": current_score,
            }
        )

    edge_list = get_edge_list(variables)
    return {
        "best_dag": graph,
        "best_bitstring": dag_to_bitstring(graph, edge_list),
        "best_score": current_score,
        "edge_list": edge_list,
        "local_scores": local_scores,
        "max_parents": max_parents,
        "evaluations": evaluations,
        "trace": trace,
    }
