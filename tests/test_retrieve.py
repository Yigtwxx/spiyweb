"""Seed injection glue: the retrieve() call is nothing but contact + propagate.

The claim under test: retrieve() adds NO retrieval logic of its own. The seed
similarities the index reports are handed to the core unchanged, so the result
must be bit-identical to calling propagate() with the same weights by hand.
The only behaviour retrieve() owns is contact hygiene (non-positive contacts
never become seeds) and the deliberately partial result contract.
"""

from __future__ import annotations

import warnings
from dataclasses import fields

import pytest

from spiyweb import Graph, RetrievalConfig, RetrievalResult, propagate, retrieve

# A miniature of the canonical §2.6 opening: two contacts at .9/.7 split the
# seed energy 5.625 / 4.375, and both feed the middle node b.
EDGES = [
    ("a", "b", 0.9),
    ("b", "c", 0.5),
]

CONTACTS = [
    ("a", 0.9),
    ("c", 0.7),
    ("dead", 0.0),
    ("anti", -0.3),
]


def make_graph() -> Graph:
    return Graph.from_edges(EDGES)


class FakeSeedSource:
    """Dict-backed stand-in for VectorStore.search; records what it was asked."""

    def __init__(self, contacts: list[tuple[str, float]]) -> None:
        self.contacts = contacts
        self.calls: list[tuple[tuple[float, ...], int]] = []

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        self.calls.append((tuple(query), k))
        return self.contacts[:k]


def test_retrieve_equals_hand_called_propagate_on_the_same_seeds() -> None:
    graph = make_graph()
    result = retrieve([1.0, 0.0], FakeSeedSource(CONTACTS), graph)
    by_hand = propagate(graph, {"a": 0.9, "c": 0.7})

    assert result.ranked() == by_hand.ranked(), (
        "retrieve() must add no retrieval logic of its own - same contacts, "
        "same propagation, same ranking"
    )
    assert result.propagation.stop_reason == by_hand.stop_reason


def test_seed_energy_splits_proportionally_to_contact_similarity() -> None:
    result = retrieve([1.0, 0.0], FakeSeedSource(CONTACTS), make_graph())

    # Hand-traced: 10.0 * .9/1.6 = 5.625 and 10.0 * .7/1.6 = 4.375 at hop 0;
    # b receives 5.625*.6 + 4.375*.6 = 6.0 - converging evidence at hop 1.
    assert result.propagation.energy_of("a") == pytest.approx(5.625)
    assert result.propagation.energy_of("c") == pytest.approx(4.375)
    assert result.propagation.energy_of("b") == pytest.approx(6.0)


def test_seed_width_is_what_the_index_is_asked_for() -> None:
    index = FakeSeedSource(CONTACTS)
    retrieve([0.5, 0.5], index, make_graph(), RetrievalConfig(seed_width=1))

    assert index.calls == [((0.5, 0.5), 1)], (
        "the index must be asked for exactly seed_width contacts - the width "
        "is a config decision, not an index property"
    )


def test_non_positive_contacts_never_become_seeds() -> None:
    result = retrieve([1.0, 0.0], FakeSeedSource(CONTACTS), make_graph())

    assert set(result.seeds) == {"a", "c"}, (
        "a cosine of 0.0 or below is no evidence of contact and would poison "
        "the proportional split - it must be dropped before injection"
    )


def test_no_positive_contact_at_all_is_a_hard_error() -> None:
    index = FakeSeedSource([("dead", 0.0), ("anti", -0.3)])
    with pytest.raises(ValueError, match="no seed contact"):
        retrieve([1.0, 0.0], index, make_graph())


def test_a_seed_missing_from_the_graph_holds_its_energy() -> None:
    result = retrieve([1.0, 0.0], FakeSeedSource([("ghost", 0.5)]), make_graph())

    assert result.propagation.energy_of("ghost") == pytest.approx(10.0), (
        "an isolated chunk is a legitimate contact: it receives its share and "
        "simply has nowhere to forward it"
    )
    assert result.propagation.stop_reason == "threshold"


def test_confidence_reports_the_activation_triple() -> None:
    result = retrieve([1.0, 0.0], FakeSeedSource(CONTACTS), make_graph())

    confidence = result.confidence
    # 5.625 + 4.375 + 6.0 over three nodes, deepest activation at hop 1.
    assert confidence.total_energy == pytest.approx(16.0)
    assert confidence.node_count == 3
    assert confidence.hop_depth == 1


def test_ranked_delegates_to_the_propagation_result() -> None:
    result = retrieve([1.0, 0.0], FakeSeedSource(CONTACTS), make_graph())
    assert result.ranked() == result.propagation.ranked()


def test_the_result_contract_is_deliberately_partial() -> None:
    assert {spec.name for spec in fields(RetrievalResult)} == {
        "seeds",
        "propagation",
        "contact_suppressed",
        "contact_votes",
        "contact_supporters",
        "contact_tau",
        "dedup_mode",
    }, (
        "the full §2.5 contract (paths, clusters, gaps, refusal) grows "
        "additively and on purpose - the contact_* fields are the elastic "
        "refill's ledger (2026-08-14 A1 decision), contact_supporters the "
        "sources behind its votes (2026-10-03); if you extended this again, "
        "update this test alongside the contract"
    )


def test_retrieval_config_rejects_a_zero_seed_width() -> None:
    with pytest.raises(ValueError, match="seed_width"):
        RetrievalConfig(seed_width=0)


# --- Elastic contact refill (2026-08-14 A1 decision) ---------------------

DUP_CONTACTS = [
    ("a", 0.9),
    ("a_dup", 0.89),
    ("c", 0.7),
    ("d", 0.5),
]


def _no_twins(node: str, others: list[str]) -> list[float]:
    """A live similarity backend that finds no duplicate anywhere.

    Two tests below ablate the SOURCE rule; their subject is
    `distinct_sources=False`, not the absence of a cosine backend. The
    guard refuses a config no rule can run, so they supply the
    emptiest legal backend instead of none.

    The SEEDS do not move - every pair scores 0.0, the adaptive cut
    clamps to `floor`, `find_survivor` never fires. The RUN does: the
    contact scan goes `width * overfetch` deep, `contact_tau` becomes a
    number instead of None, and neighbour-level suppression now runs at
    every hop (`core/propagate.py` gates that on `similarity` alone).
    Writing "nothing changed" here would be the same kind of claim that
    cost this project a measurement campaign, so the tests assert the
    new reality instead of papering over it.
    """
    return [0.0] * len(others)


def _twin_similarity(node: str, others: list[str]) -> list[float]:
    twins = {frozenset(("a", "a_dup")): 0.96}
    return [twins.get(frozenset((node, other)), 0.0) for other in others]


def test_elastic_refill_gives_the_freed_slot_to_the_next_distinct_idea() -> None:
    from spiyweb import DedupConfig

    index = FakeSeedSource(DUP_CONTACTS)
    result = retrieve(
        [1.0, 0.0],
        index,
        make_graph(),
        RetrievalConfig(seed_width=2, contact_overfetch=2),
        similarity=_twin_similarity,
        dedup=DedupConfig(floor=0.90, min_pairs=100),
    )
    assert index.calls[0][1] == 4, "dedup active -> width * overfetch deep"
    assert set(result.seeds) == {"a", "c"}, (
        "the twin is skipped and its slot goes to the next distinct contact"
    )
    assert result.contact_suppressed == {"a_dup": "a"}
    assert result.contact_tau == 0.90
    assert result.votes()["a"] == 2, "a skipped twin is corpus support"


def test_overfetch_is_inert_while_dedup_is_off() -> None:
    index = FakeSeedSource(DUP_CONTACTS)
    result = retrieve(
        [1.0, 0.0],
        index,
        make_graph(),
        RetrievalConfig(seed_width=2, contact_overfetch=3),
    )
    assert index.calls == [((1.0, 0.0), 2)], "no dedup -> plain seed_width"
    assert set(result.seeds) == {"a", "a_dup"}
    assert result.contact_suppressed == {}
    assert result.contact_tau is None


def test_colored_elastic_refill_records_per_colour_and_merges_votes() -> None:
    from spiyweb import ColoredRetrievalConfig, DedupConfig, retrieve_colored

    index = FakeSeedSource(DUP_CONTACTS)
    result = retrieve_colored(
        {"c0": [1.0, 0.0]},
        index,
        make_graph(),
        ColoredRetrievalConfig(seed_width=2, contact_overfetch=2),
        similarity=_twin_similarity,
        dedup=DedupConfig(floor=0.90, min_pairs=100),
    )
    assert result.seeds_by_color == {"c0": {"a": 0.9, "c": 0.7}}
    assert result.contact_suppressed == {"c0": {"a_dup": "a"}}
    assert result.contact_taus == {"c0": 0.90}
    assert result.votes()["a"] == 2


# --- Distinct sources at contact selection (2026-08-16) -------------------
#
# The cosine twin test cannot see this failure: two propositions of one
# passage are different sentences, so they are never near-duplicates, yet a
# query part that seeds both explores one passage instead of two. Measured on
# `musique_prop200`: 287 of 534 colours had both seeds on one passage (0 of
# 534 on the chunk-only control), costing the coloured web -.0524 P=.001.

SOURCE_CONTACTS = [
    ("p1#p0", 0.9),
    ("p1#p3", 0.88),  # same passage, different sentence
    ("p2", 0.7),
    ("p3", 0.5),
]


def _two_layer_graph() -> Graph:
    from spiyweb.core.graph import Node

    nodes = [
        Node(id="p1", layer="chunk", source_id="d1", length=400),
        Node(id="p1#p0", layer="proposition", source_id="d1", length=40),
        Node(id="p1#p3", layer="proposition", source_id="d1", length=40),
        Node(id="p2", layer="chunk", source_id="d2", length=400),
        Node(id="p3", layer="chunk", source_id="d3", length=400),
    ]
    layers = {"semantic": [("p1#p0", "p2", 0.5), ("p2", "p3", 0.5)]}
    return Graph.from_layers(layers, nodes=nodes)  # type: ignore[arg-type]


def test_a_second_seed_on_the_same_passage_is_a_twin_and_refills() -> None:
    from spiyweb import DedupConfig

    index = FakeSeedSource(SOURCE_CONTACTS)
    # Only the source rule can run without a similarity backend, and the
    # library now says so out loud - that warning is part of the contract.
    with pytest.warns(UserWarning, match="only the distinct-source"):
        result = retrieve(
            [1.0, 0.0],
            index,
            _two_layer_graph(),
            RetrievalConfig(seed_width=2, contact_overfetch=2),
            dedup=DedupConfig(),
        )
    assert result.dedup_mode == "sources_only"
    assert index.calls[0][1] == 4, "the source rule needs the overfetch too"
    assert set(result.seeds) == {"p1#p0", "p2"}, (
        "the colour must reach a SECOND passage, not a second sentence of the first one"
    )
    assert result.contact_suppressed == {"p1#p3": "p1#p0"}
    assert "p1#p0" not in result.votes(), (
        "a second sentence of the same passage is not a second source - "
        "skipped and refilled, but never a vote (2026-10-03)"
    )
    assert result.contact_tau is None, "no cosine test ran, so no cut to report"


def test_the_source_rule_is_individually_disableable() -> None:
    from spiyweb import DedupConfig

    result = retrieve(
        [1.0, 0.0],
        FakeSeedSource(SOURCE_CONTACTS),
        _two_layer_graph(),
        RetrievalConfig(seed_width=2, contact_overfetch=2),
        similarity=_no_twins,
        dedup=DedupConfig(distinct_sources=False),
    )
    assert result.dedup_mode == "cosine_only", (
        "source rule off, cosine test live - that IS the ablation"
    )
    assert set(result.seeds) == {"p1#p0", "p1#p3"}, (
        "without the source rule both propositions of one passage keep "
        "their slots - the behaviour the rule was added to change"
    )
    assert result.contact_suppressed == {}, "0.0 everywhere suppresses nothing"


def test_the_source_rule_is_a_no_op_on_a_single_layer_index() -> None:
    """Every chunk is its own source, so no measured number can move."""
    from spiyweb import DedupConfig
    from spiyweb.core.graph import Node

    nodes = [
        Node(id=name, layer="chunk", source_id=f"doc-{name}", length=400)
        for name in ("a", "b", "c")
    ]
    graph = Graph.from_layers({"semantic": list(EDGES)}, nodes=nodes)  # type: ignore[arg-type]
    on, off = (
        retrieve(
            [1.0, 0.0],
            FakeSeedSource(CONTACTS),
            graph,
            RetrievalConfig(seed_width=2, contact_overfetch=2),
            similarity=_no_twins,
            dedup=DedupConfig(distinct_sources=flag),
        )
        for flag in (True, False)
    )
    assert on.dedup_mode == "full" and off.dedup_mode == "cosine_only", (
        "the flag really is the only difference between the two runs"
    )
    assert dict(on.seeds) == dict(off.seeds)
    assert on.contact_suppressed == off.contact_suppressed == {}


def test_the_source_rule_binds_inside_a_colour_not_across_colours() -> None:
    """Two sub-questions may legitimately land on the same passage."""
    from spiyweb import ColoredRetrievalConfig, DedupConfig, retrieve_colored

    with pytest.warns(UserWarning, match="only the distinct-source"):
        result = retrieve_colored(
            {"c0": [1.0, 0.0], "c1": [0.0, 1.0]},
            FakeSeedSource(SOURCE_CONTACTS),
            _two_layer_graph(),
            ColoredRetrievalConfig(seed_width=2, contact_overfetch=2),
            dedup=DedupConfig(),
        )
    assert result.dedup_mode == "sources_only"
    assert result.seeds_by_color["c0"] == {"p1#p0": 0.9, "p2": 0.7}
    assert result.seeds_by_color["c1"] == {"p1#p0": 0.9, "p2": 0.7}, (
        "the rule is per colour - it must not stop a second colour from "
        "touching a passage the first one already used"
    )
    assert result.contact_suppressed["c0"] == {"p1#p3": "p1#p0"}


# --- The dedup contract (2026-08-25) -------------------------------------


def test_a_dedup_config_no_rule_can_run_is_refused() -> None:
    """The trap that cost the 2026-08 campaign, now a hard error.

    `DedupConfig` says what the caller wanted; the cosine twin test also needs
    a `similarity` backend. With `distinct_sources=False` switching off the
    only rule that runs without one, the config asks for suppression and gets
    an unsuppressed web. Silence there is the failure, not the permissiveness.
    """
    from spiyweb import DedupConfig

    with pytest.raises(ValueError, match="nothing can run it"):
        retrieve(
            [1.0, 0.0],
            FakeSeedSource(CONTACTS),
            make_graph(),
            RetrievalConfig(seed_width=2),
            dedup=DedupConfig(distinct_sources=False),
        )


def test_the_coloured_path_refuses_the_same_config() -> None:
    from spiyweb import ColoredRetrievalConfig, DedupConfig, retrieve_colored

    with pytest.raises(ValueError, match="nothing can run it"):
        retrieve_colored(
            {"c0": [1.0, 0.0]},
            FakeSeedSource(CONTACTS),
            make_graph(),
            ColoredRetrievalConfig(seed_width=2),
            dedup=DedupConfig(distinct_sources=False),
        )


def test_the_result_reports_which_dedup_rules_actually_ran() -> None:
    """`dedup_mode` is the receipt. Without it, "the mechanism was on" is a
    belief about a config rather than a statement about a run."""
    from spiyweb import DedupConfig

    plain = retrieve([1.0, 0.0], FakeSeedSource(CONTACTS), make_graph())
    assert plain.dedup_mode == "off"

    full = retrieve(
        [1.0, 0.0],
        FakeSeedSource(DUP_CONTACTS),
        make_graph(),
        RetrievalConfig(seed_width=2, contact_overfetch=2),
        similarity=_twin_similarity,
        dedup=DedupConfig(floor=0.90, min_pairs=100),
    )
    assert full.dedup_mode == "full"

    cosine_only = retrieve(
        [1.0, 0.0],
        FakeSeedSource(DUP_CONTACTS),
        make_graph(),
        RetrievalConfig(seed_width=2, contact_overfetch=2),
        similarity=_twin_similarity,
        dedup=DedupConfig(floor=0.90, min_pairs=100, distinct_sources=False),
    )
    assert cosine_only.dedup_mode == "cosine_only"


def test_a_configuration_that_cannot_spread_says_so() -> None:
    """The shipped defaults returned hop-0-only webs on a real index.

    Five seeds share 10.0 energy, so the strongest holds ~2.05 and forwards
    at most 1.23 - against a threshold of 1.5. Nothing can ever clear it, so
    the web stops at first contact: `top-k` with extra steps, which is the
    one thing this library exists to not be. CLAUDE.md sizes the threshold
    for the TWO-seed worked example (§2.1) and the seed width at 5 (§4);
    together they cannot propagate. Measured 2026-08-26.
    """
    from spiyweb.config import PropagationConfig
    from spiyweb.retrieve import _check_propagation_can_spread

    five = {f"n{i}": 0.78 for i in range(5)}
    with pytest.warns(UserWarning, match="cannot spread past the seed"):
        _check_propagation_can_spread(five, PropagationConfig())


def test_the_canonical_two_seed_example_is_never_warned_about() -> None:
    """§2.6's trace is the documented case; warning on it would be noise."""
    from spiyweb.config import PropagationConfig
    from spiyweb.retrieve import _check_propagation_can_spread

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_propagation_can_spread({"A": 0.9, "C": 0.7}, PropagationConfig())


def test_a_profile_clears_the_bar() -> None:
    """Every named profile propagates; that is why they exist."""
    from spiyweb.config import PropagationConfig, RetrievalConfig
    from spiyweb.profiles import PROFILES
    from spiyweb.retrieve import _check_propagation_can_spread

    five = {f"n{i}": 0.78 for i in range(5)}
    for name, profile in PROFILES.items():
        settings = profile.as_retrieval(RetrievalConfig()).propagation
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_propagation_can_spread(five, settings)
        assert isinstance(settings, PropagationConfig), name


def test_a_source_supporting_at_contact_and_in_the_web_votes_once() -> None:
    """The two suppression stages merge supporter sets, not increments."""
    from spiyweb.core.propagate import PropagationResult
    from spiyweb.retrieve import RetrievalResult

    propagation = PropagationResult(
        activations={},
        injected_energy=10.0,
        threshold=1.5,
        hops_used=0,
        stop_reason="threshold",
        votes={"doc": 2},
        supporters={"doc": frozenset({"other"})},
    )
    result = RetrievalResult(
        seeds={},
        propagation=propagation,
        contact_votes={"doc": 2},
        contact_supporters={"doc": frozenset({"other"})},
    )
    assert result.votes() == {"doc": 2}
