"""Coloured retrieval glue: retrieve_colored() is contact + propagate_colored.

The claim under test mirrors test_retrieve.py: the function adds NO retrieval
logic of its own beyond contact hygiene (positive-similarity filter, the
primary-colour hard error, later-colour skipping) and hands the surviving
seeds to the core unchanged. Also pinned here: the ColoredRetrievalConfig
defaults ARE the measured winner of the 2026-08-14 grid campaign - changing
them changes what `python -m spiyweb.evaluation.run` measures.
"""

from __future__ import annotations

import pytest

from spiyweb import (
    ColoredRetrievalConfig,
    Graph,
    PropagationConfig,
    propagate,
    propagate_colored,
    retrieve_colored,
)

# A chain a - x - b: colour 0 lands on a, colour 1 lands on b, and with mild
# decay both webs reach the middle node x - the bridge.
EDGES = [
    ("a", "x", 0.9),
    ("x", "b", 0.9),
]


def make_graph() -> Graph:
    return Graph.from_edges(EDGES)


class RoutedSeedSource:
    """Maps each query embedding to its own contact list; records the ks."""

    def __init__(
        self, routes: dict[tuple[float, ...], list[tuple[str, float]]]
    ) -> None:
        self.routes = routes
        self.calls: list[tuple[tuple[float, ...], int]] = []

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        self.calls.append((tuple(query), k))
        return self.routes[tuple(query)][:k]


QUERY_A = (1.0, 0.0)
QUERY_B = (0.0, 1.0)
CONFIG = ColoredRetrievalConfig(
    propagation=PropagationConfig(damping=0.6, threshold_ratio=0.01)
)


def make_index() -> RoutedSeedSource:
    return RoutedSeedSource(
        {
            QUERY_A: [("a", 0.9), ("dead", 0.0)],
            QUERY_B: [("b", 0.7), ("anti", -0.3)],
        }
    )


def test_retrieve_colored_equals_hand_called_propagate_colored() -> None:
    graph = make_graph()
    result = retrieve_colored(
        {"c0": QUERY_A, "c1": QUERY_B}, make_index(), graph, CONFIG
    )
    by_hand = propagate_colored(
        graph, {"c0": {"a": 0.9}, "c1": {"b": 0.7}}, CONFIG.propagation
    )

    assert result.ranked() == by_hand.ranked(), (
        "retrieve_colored() must add no retrieval logic of its own - same "
        "surviving contacts, same coloured propagation, same ranking"
    )
    assert result.bridges == by_hand.bridges


def test_both_colours_meet_at_the_middle_node() -> None:
    result = retrieve_colored(
        {"c0": QUERY_A, "c1": QUERY_B}, make_index(), make_graph(), CONFIG
    )

    assert "x" in result.bridges, (
        "both colours must reach the middle node of the chain - that meeting "
        "IS the multi-hop signal the coloured web exists for"
    )
    assert result.bridges["x"] == ("c0", "c1")


def test_non_positive_contacts_never_become_coloured_seeds() -> None:
    result = retrieve_colored(
        {"c0": QUERY_A, "c1": QUERY_B}, make_index(), make_graph(), CONFIG
    )

    assert result.seeds_by_color == {"c0": {"a": 0.9}, "c1": {"b": 0.7}}, (
        "a cosine of 0.0 or below is no evidence of contact, exactly as in "
        "the plain retrieve()"
    )


def test_a_later_colour_without_positive_contact_is_skipped() -> None:
    index = RoutedSeedSource(
        {
            QUERY_A: [("a", 0.9)],
            QUERY_B: [("dead", 0.0), ("anti", -0.3)],
        }
    )
    result = retrieve_colored(
        {"c0": QUERY_A, "c1": QUERY_B}, index, make_graph(), CONFIG
    )

    assert set(result.seeds_by_color) == {"c0"}, (
        "a decomposed fragment the index cannot touch would only seed noise "
        "into a wrong region - it must be dropped, not injected raw"
    )
    assert result.bridges == {}, "one colour alone can never form a bridge"


def test_the_primary_colour_without_positive_contact_is_a_hard_error() -> None:
    index = RoutedSeedSource(
        {
            QUERY_A: [("dead", 0.0), ("anti", -0.3)],
            QUERY_B: [("b", 0.7)],
        }
    )
    with pytest.raises(ValueError, match="primary colour"):
        retrieve_colored({"c0": QUERY_A, "c1": QUERY_B}, index, make_graph(), CONFIG)


def test_no_coloured_query_at_all_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="at least one coloured query"):
        retrieve_colored({}, make_index(), make_graph(), CONFIG)


def test_a_single_colour_runs_under_the_full_seed_energy() -> None:
    graph = make_graph()
    result = retrieve_colored({"c0": QUERY_A}, make_index(), graph, CONFIG)
    by_hand = propagate(graph, {"a": 0.9}, CONFIG.propagation)

    assert result.ranked() == by_hand.ranked(), (
        "one colour means no split: the coloured call must degrade to the "
        "plain propagation bit for bit"
    )


def test_seed_width_is_asked_per_colour() -> None:
    index = make_index()
    config = ColoredRetrievalConfig(seed_width=1)
    retrieve_colored({"c0": QUERY_A, "c1": QUERY_B}, index, make_graph(), config)

    assert index.calls == [(QUERY_A, 1), (QUERY_B, 1)], (
        "every colour must be granted exactly seed_width contacts - the "
        "width is per colour, not shared across the palette"
    )


def test_confidence_aggregates_across_colours() -> None:
    result = retrieve_colored(
        {"c0": QUERY_A, "c1": QUERY_B}, make_index(), make_graph(), CONFIG
    )
    confidence = result.confidence

    per_color = result.colored.per_color.values()
    assert confidence.total_energy == pytest.approx(
        sum(a.energy for r in per_color for a in r.activations.values())
    )
    assert confidence.node_count == 3, "a, b and x each activated at least once"
    assert confidence.hop_depth == max(r.hops_used for r in per_color)


def test_the_defaults_are_the_measured_campaign_winner() -> None:
    config = ColoredRetrievalConfig()
    # 2026-08-14 MuSiQue campaign, tour 12: S@5 .512 at ~2.6 LLM
    # calls/question - qwen3.5:9b decomposition, sequential chaining.
    # Changing any of these changes what the evaluation harness measures.
    assert config.seed_width == 2
    assert config.propagation.threshold_ratio == pytest.approx(0.01)
    assert config.propagation.split_alpha == pytest.approx(3.0)
    assert config.max_colors == 4
    assert config.chain_mode == "sequential"
    assert config.decomposition_model == "qwen3.5:9b"
    assert config.decomposition_no_think is True
    assert config.max_answer_words == 10


def test_colored_config_rejects_unknown_chain_mode() -> None:
    with pytest.raises(ValueError, match="chain_mode"):
        ColoredRetrievalConfig(chain_mode="both")


def test_colored_config_rejects_empty_decomposition_model() -> None:
    with pytest.raises(ValueError, match="decomposition_model"):
        ColoredRetrievalConfig(decomposition_model="")


def test_the_core_propagation_defaults_stay_canonical() -> None:
    # The winning operating point lives in ColoredRetrievalConfig ONLY; the
    # core defaults keep carrying the canonical §2.6 worked example.
    config = PropagationConfig()
    assert config.threshold_ratio == pytest.approx(0.15)
    assert config.split_alpha == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("seed_width", 0),
        ("max_colors", 0),
        ("max_answer_words", 0),
    ],
)
def test_colored_config_rejects_non_positive_counts(
    field_name: str, value: int
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ColoredRetrievalConfig(**{field_name: value})
