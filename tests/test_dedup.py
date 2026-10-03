"""Dynamic redundancy suppression: dedup -> vote, the project's core claim.

The canonical trace (CLAUDE.md section 2.6) pins the near-duplicate `A'` with a
pre-suppressed edge. Here the same trace runs with a LIVE `0.95` edge and a
similarity function instead: the mechanism itself must zero the edge at query
time, renormalise the shares, and turn the duplicate into a vote - producing
exactly the documented ledger.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from spiyweb import (
    DedupConfig,
    Graph,
    PropagationConfig,
    adaptive_threshold,
    find_survivor,
    propagate,
    propagate_colored,
)
from spiyweb.core.dedup import SimilarityFn
from spiyweb.core.propagate import PropagationResult

# The canonical graph, except A -> A_dup carries its real weight instead of a
# pre-zeroed one; suppression must now happen dynamically.
LIVE_DUP_EDGES = [
    ("A", "A_dup", 0.95),
    ("A", "B", 0.8),
    ("A", "D", 0.4),
    ("C", "D", 0.6),
    ("C", "E", 0.3),
    ("D", "F", 0.5),
]
CANONICAL_SEEDS = {"A": 0.9, "C": 0.7}


def pair_similarity(pairs: dict[frozenset[str], float]) -> SimilarityFn:
    """Similarity backed by an explicit pair table; unlisted pairs score 0."""

    def similarity(node: str, others: Sequence[str]) -> list[float]:
        return [pairs.get(frozenset((node, other)), 0.0) for other in others]

    return similarity


DUP_SIMILARITY = pair_similarity({frozenset(("A", "A_dup")): 0.95})
# min_pairs above any active-set size here -> tau stays at the floor, so the
# canonical assertions do not depend on the adaptive formula.
FLOOR_ONLY = DedupConfig(floor=0.90, min_pairs=100)


@pytest.fixture
def dedup_result() -> PropagationResult:
    graph = Graph.from_edges(LIVE_DUP_EDGES)
    return propagate(
        graph,
        CANONICAL_SEEDS,
        PropagationConfig(),
        similarity=DUP_SIMILARITY,
        dedup=FLOOR_ONLY,
    )


def test_dynamic_dedup_reproduces_the_canonical_ledger(
    dedup_result: PropagationResult,
) -> None:
    """A live duplicate edge ends in the exact energies of the worked example."""
    expected = {"A": 5.625, "C": 4.375, "D": 2.875, "B": 2.25, "F": 1.725}
    assert set(dedup_result.activations) == set(expected)
    for node, energy in expected.items():
        assert dedup_result.energy_of(node) == pytest.approx(energy), node


def test_duplicate_is_suppressed_and_voted(dedup_result: PropagationResult) -> None:
    """`A_dup` never activates; idea `A` counts double."""
    assert "A_dup" not in dedup_result.activations
    assert dedup_result.suppressed == {"A_dup": "A"}
    assert dedup_result.votes == {"A": 2}


def test_suppressed_share_is_redistributed_not_destroyed(
    dedup_result: PropagationResult,
) -> None:
    """`B` receives the renormalised share - energy is conserved, not burned."""
    forwarded = dedup_result.energy_of("A") * 0.60
    assert dedup_result.energy_of("B") == pytest.approx(forwarded * 0.8 / 1.2)


def test_computed_thresholds_are_visible(dedup_result: PropagationResult) -> None:
    """One recorded cut per checking stage, as the design requires.

    Since seed-level dedup (`include_seeds`), the injection check records the
    first entry; the three distributing hops record the rest.
    """
    assert dedup_result.dedup_thresholds == (0.90, 0.90, 0.90, 0.90)


def test_disabled_dedup_dilutes_the_shares_and_votes_nothing() -> None:
    """The ablation switch: the live duplicate edge stays in the denominator.

    `A` forwards over 0.95 + 0.8 + 0.4 = 2.15 instead of 1.2, so `A_dup`
    arrives at 1.491 and `B` at 1.256 - BOTH below the 1.5 floor. Without
    suppression the duplicate not only competes, it starves a genuine
    neighbour out of the web entirely. No votes are recorded.
    """
    graph = Graph.from_edges(LIVE_DUP_EDGES)
    result = propagate(
        graph,
        CANONICAL_SEEDS,
        PropagationConfig(),
        similarity=DUP_SIMILARITY,
        dedup=DedupConfig(enabled=False),
    )
    assert "A_dup" not in result.activations
    assert "B" not in result.activations
    diluted_d = 3.375 * 0.4 / 2.15 + 2.625 * 0.6 / 0.9
    assert result.energy_of("D") == pytest.approx(diluted_d)
    assert result.votes == {}
    assert result.suppressed == {}
    assert result.dedup_thresholds == ()


def test_run_without_similarity_reports_empty_dedup_fields() -> None:
    graph = Graph.from_edges(LIVE_DUP_EDGES)
    result = propagate(graph, CANONICAL_SEEDS)
    assert result.votes == {}
    assert result.suppressed == {}
    assert result.dedup_thresholds == ()


def test_votes_use_source_granularity_when_mapped() -> None:
    """Vote counts land on the document, never on the chunk."""
    graph = Graph.from_edges(LIVE_DUP_EDGES)
    result = propagate(
        graph,
        CANONICAL_SEEDS,
        PropagationConfig(),
        similarity=DUP_SIMILARITY,
        dedup=FLOOR_ONLY,
        source_of={"A": "doc-1", "A_dup": "doc-2"},
    )
    assert result.votes == {"doc-1": 2}


def test_a_duplicate_from_the_same_source_is_suppressed_but_never_votes() -> None:
    """Support is counted per SOURCE, so a source cannot vote for itself.

    On a two-layer index a passage's propositions restate it and each other;
    each one suppressed used to add a vote to its own document, and a
    three-document corpus reported nine votes. Suppression is unchanged - the
    edge still goes to zero and the share is still redistributed - only the
    ledger of corpus support no longer counts the source twice.
    """
    graph = Graph.from_edges(LIVE_DUP_EDGES)
    same = propagate(
        graph,
        CANONICAL_SEEDS,
        PropagationConfig(),
        similarity=DUP_SIMILARITY,
        dedup=FLOOR_ONLY,
        source_of={"A": "doc-1", "A_dup": "doc-1"},
    )
    distinct = propagate(
        graph,
        CANONICAL_SEEDS,
        PropagationConfig(),
        similarity=DUP_SIMILARITY,
        dedup=FLOOR_ONLY,
        source_of={"A": "doc-1", "A_dup": "doc-2"},
    )
    assert same.suppressed == distinct.suppressed == {"A_dup": "A"}
    assert same.ranked() == distinct.ranked(), "the energy ledger is identical"
    assert same.votes == {}
    assert distinct.votes == {"doc-1": 2}


def test_one_other_source_is_one_vote_however_many_copies_it_holds() -> None:
    """Votes count distinct SOURCES, not suppressed nodes.

    Two propositions of one passage restating another document's idea are
    one document agreeing, not two. Both are still suppressed.
    """
    graph = Graph.from_edges(
        [("A", "A_dup1", 0.95), ("A", "A_dup2", 0.95), ("A", "B", 0.8)]
    )
    similarity = pair_similarity(
        {
            frozenset(("A", "A_dup1")): 0.95,
            frozenset(("A", "A_dup2")): 0.95,
            frozenset(("A_dup1", "A_dup2")): 0.95,
        }
    )
    result = propagate(
        graph,
        {"A": 1.0},
        PropagationConfig(),
        similarity=similarity,
        dedup=FLOOR_ONLY,
        source_of={"A": "doc-1", "A_dup1": "doc-2", "A_dup2": "doc-2", "B": "doc-3"},
    )
    assert result.suppressed == {"A_dup1": "A", "A_dup2": "A"}
    assert result.supporters == {"doc-1": frozenset({"doc-2"})}
    assert result.votes == {"doc-1": 2}


def test_without_a_source_mapping_every_copy_is_its_own_source() -> None:
    graph = Graph.from_edges([("A", "A_dup1", 0.95), ("A", "A_dup2", 0.95)])
    similarity = pair_similarity(
        {frozenset(("A", "A_dup1")): 0.95, frozenset(("A", "A_dup2")): 0.95}
    )
    result = propagate(
        graph, {"A": 1.0}, PropagationConfig(), similarity=similarity, dedup=FLOOR_ONLY
    )
    assert result.votes == {"A": 3}


def test_adaptive_threshold_falls_back_to_floor_below_min_pairs() -> None:
    similarity = pair_similarity({frozenset(("x", "y")): 0.99})
    config = DedupConfig(floor=0.85, min_pairs=2)
    assert adaptive_threshold(["x", "y"], similarity, config) == 0.85


def test_adaptive_threshold_uses_mean_plus_sigma_std() -> None:
    # Four nodes, six pairs: sims [0.4, 0.4, 0.4, 0.6, 0.6, 0.6]
    # mean 0.5, std 0.1 -> tau = 0.5 + 2 * 0.1 = 0.7 (above the 0.2 floor).
    pairs = {
        frozenset(("a", "b")): 0.4,
        frozenset(("a", "c")): 0.4,
        frozenset(("a", "d")): 0.6,
        frozenset(("b", "c")): 0.4,
        frozenset(("b", "d")): 0.6,
        frozenset(("c", "d")): 0.6,
    }
    config = DedupConfig(sigma=2.0, floor=0.2, min_pairs=6)
    tau = adaptive_threshold(["a", "b", "c", "d"], pair_similarity(pairs), config)
    assert tau == pytest.approx(0.7)


def test_adaptive_threshold_never_drops_below_the_floor() -> None:
    pairs = {
        frozenset(("a", "b")): 0.1,
        frozenset(("a", "c")): 0.1,
        frozenset(("b", "c")): 0.1,
    }
    config = DedupConfig(sigma=2.0, floor=0.9, min_pairs=3)
    tau = adaptive_threshold(["a", "b", "c"], pair_similarity(pairs), config)
    assert tau == 0.9


def test_find_survivor_returns_none_below_threshold() -> None:
    similarity = pair_similarity({frozenset(("cand", "a")): 0.5})
    assert find_survivor("cand", ["a"], similarity, 0.9) is None


def test_find_survivor_picks_the_most_similar_active_node() -> None:
    similarity = pair_similarity(
        {frozenset(("cand", "a")): 0.92, frozenset(("cand", "b")): 0.97}
    )
    assert find_survivor("cand", ["a", "b"], similarity, 0.9) == "b"


def test_find_survivor_breaks_ties_on_node_id() -> None:
    similarity = pair_similarity(
        {frozenset(("cand", "b")): 0.95, frozenset(("cand", "a")): 0.95}
    )
    assert find_survivor("cand", ["b", "a"], similarity, 0.9) == "a"


# Seed-level dedup (`include_seeds`, 2026-08-14 A1 decision): twin CONTACTS
# must not each hold a seed slot. The graph edge is deliberately unrelated to
# the seeds - isolated seeds legitimately hold their energy.
SEED_TWIN_SIMILARITY = pair_similarity({frozenset(("A", "A2")): 0.96})
UNRELATED_GRAPH_EDGES = [("X", "Y", 0.5)]


def test_duplicate_seed_suppressed_share_renormalised_and_voted() -> None:
    """The twin seed is dropped BEFORE the split; its share is conserved."""
    result = propagate(
        Graph.from_edges(UNRELATED_GRAPH_EDGES),
        {"A": 0.9, "A2": 0.88, "B": 0.45},
        PropagationConfig(),
        similarity=SEED_TWIN_SIMILARITY,
        dedup=FLOOR_ONLY,
    )
    assert result.suppressed == {"A2": "A"}
    assert result.votes == {"A": 2}
    assert "A2" not in result.activations
    # 10.0 split over .9 + .45 (the twin's share redistributed, not burned).
    assert result.energy_of("A") == pytest.approx(10.0 * 0.9 / 1.35)
    assert result.energy_of("B") == pytest.approx(10.0 * 0.45 / 1.35)
    # Injection cut recorded first, then the single distributing hop.
    assert result.dedup_thresholds == (0.90, 0.90)


def test_seed_dedup_keeps_the_strongest_contact() -> None:
    result = propagate(
        Graph.from_edges(UNRELATED_GRAPH_EDGES),
        {"A": 0.9, "A2": 0.95},
        PropagationConfig(),
        similarity=SEED_TWIN_SIMILARITY,
        dedup=FLOOR_ONLY,
    )
    assert result.suppressed == {"A": "A2"}
    assert result.energy_of("A2") == pytest.approx(10.0)


def test_include_seeds_off_restores_injection_blind_behaviour() -> None:
    """The ablation switch: twin seeds both inject and split the energy."""
    result = propagate(
        Graph.from_edges(UNRELATED_GRAPH_EDGES),
        {"A": 0.9, "A2": 0.88},
        PropagationConfig(),
        similarity=SEED_TWIN_SIMILARITY,
        dedup=DedupConfig(floor=0.90, min_pairs=100, include_seeds=False),
    )
    assert result.suppressed == {}
    assert result.votes == {}
    assert result.energy_of("A2") == pytest.approx(10.0 * 0.88 / 1.78)


def test_seed_votes_use_source_granularity_when_mapped() -> None:
    result = propagate(
        Graph.from_edges(UNRELATED_GRAPH_EDGES),
        {"A": 0.9, "A2": 0.88},
        PropagationConfig(),
        similarity=SEED_TWIN_SIMILARITY,
        dedup=FLOOR_ONLY,
        source_of={"A": "doc-1", "A2": "doc-2"},
    )
    assert result.votes == {"doc-1": 2}


def test_a_seed_twin_from_the_same_source_is_dropped_but_never_votes() -> None:
    result = propagate(
        Graph.from_edges(UNRELATED_GRAPH_EDGES),
        {"A": 0.9, "A2": 0.88},
        PropagationConfig(),
        similarity=SEED_TWIN_SIMILARITY,
        dedup=FLOOR_ONLY,
        source_of={"A": "doc-1", "A2": "doc-1"},
    )
    assert result.suppressed == {"A2": "A"}
    assert result.votes == {}


def test_colored_webs_dedup_independently_and_merge_votes() -> None:
    """One suppressed duplicate per colour on the same source -> 3, not 4."""
    graph = Graph.from_edges([("S1", "D1", 0.9), ("S2", "D2", 0.9)])
    similarity = pair_similarity(
        {frozenset(("S1", "D1")): 0.96, frozenset(("S2", "D2")): 0.96}
    )
    result = propagate_colored(
        graph,
        {"c0": {"S1": 1.0}, "c1": {"S2": 1.0}},
        PropagationConfig(),
        similarity=similarity,
        dedup=FLOOR_ONLY,
        source_of={"S1": "doc", "S2": "doc"},
    )
    assert result.per_color["c0"].suppressed == {"D1": "S1"}
    assert result.per_color["c1"].suppressed == {"D2": "S2"}
    assert result.per_color["c0"].votes == {"doc": 2}
    assert result.votes() == {"doc": 3}


def test_one_supporting_source_across_two_colours_is_one_vote() -> None:
    """The colour merge unions supporters; it no longer sums suppressions."""
    graph = Graph.from_edges([("S1", "D1", 0.9), ("S2", "D2", 0.9)])
    similarity = pair_similarity(
        {frozenset(("S1", "D1")): 0.96, frozenset(("S2", "D2")): 0.96}
    )
    result = propagate_colored(
        graph,
        {"c0": {"S1": 1.0}, "c1": {"S2": 1.0}},
        PropagationConfig(),
        similarity=similarity,
        dedup=FLOOR_ONLY,
        source_of={"S1": "doc", "S2": "doc", "D1": "other", "D2": "other"},
    )
    assert result.per_color["c0"].votes == {"doc": 2}
    assert result.per_color["c1"].votes == {"doc": 2}
    assert result.votes() == {"doc": 2}
