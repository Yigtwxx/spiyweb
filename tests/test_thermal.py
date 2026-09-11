"""Thermal conversation memory (D22/D32): follow-ups land on warm ground.

The claim under test: residue energy is injected on top of the seed split,
the relative stop threshold scales with the injected TOTAL (the reason it is
relative at all, D5/D27), the session keeps `residue_ratio` of each turn
warm, `reset()` cools it, and the contact-overlap auto-reset fires only on a
real topic change.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from spiyweb import (
    Graph,
    PropagationConfig,
    RetrievalConfig,
    ThermalConfig,
    ThermalSession,
    propagate,
)

# A -> B: the canonical single-edge run (A 10.0, B 6.0), plus an isolated
# far-away region X for the topic-change tests.
CHAIN = Graph.from_edges([("A", "B", 0.5)])

QUERY_AB = (1.0, 0.0)
QUERY_X = (0.0, 1.0)


class FakeIndex:
    """Two disjoint neighbourhoods: one query warms A, the other touches X."""

    def search(self, query: Sequence[float], k: int) -> list[tuple[str, float]]:
        contacts = [("A", 0.9)] if query[0] >= query[1] else [("X", 0.9)]
        return contacts[:k]


def test_residue_raises_the_injected_total_and_the_threshold() -> None:
    result = propagate(CHAIN, {"A": 1.0}, PropagationConfig(), residue={"X": 2.0})
    assert result.injected_energy == pytest.approx(12.0)
    assert result.threshold == pytest.approx(0.15 * 12.0), (
        "the stop rule scales with the injected TOTAL - D5/D27"
    )
    assert result.energy_of("X") == pytest.approx(2.0), "warm ground holds"
    assert result.energy_of("A") == pytest.approx(10.0)


def test_a_warm_seed_simply_starts_hotter() -> None:
    result = propagate(CHAIN, {"A": 1.0}, PropagationConfig(), residue={"A": 2.5})
    assert result.energy_of("A") == pytest.approx(12.5)


def test_non_positive_residue_entries_are_ignored() -> None:
    result = propagate(
        CHAIN, {"A": 1.0}, PropagationConfig(), residue={"X": 0.0, "Y": -1.0}
    )
    assert result.injected_energy == pytest.approx(10.0)
    assert "X" not in result.activations


def test_session_injects_the_residue_ratio_of_the_previous_turn() -> None:
    session = ThermalSession(FakeIndex(), CHAIN, RetrievalConfig())
    first = session.retrieve(QUERY_AB)
    assert first.propagation.injected_energy == pytest.approx(10.0), "cold start"
    # Warm ground: A 10.0 * .25 = 2.5, B 6.0 * .25 = 1.5 -> injected 14.0;
    # the repeated contact A starts at 10.0 + 2.5.
    second = session.retrieve(QUERY_AB)
    assert second.propagation.injected_energy == pytest.approx(14.0)
    assert second.propagation.energy_of("A") == pytest.approx(12.5)


def test_reset_cools_the_ground() -> None:
    session = ThermalSession(FakeIndex(), CHAIN, RetrievalConfig())
    session.retrieve(QUERY_AB)
    assert session.warm
    session.reset()
    assert not session.warm
    again = session.retrieve(QUERY_AB)
    assert again.propagation.injected_energy == pytest.approx(10.0)


def test_disabled_thermal_is_a_stateless_retrieve() -> None:
    session = ThermalSession(
        FakeIndex(), CHAIN, RetrievalConfig(), thermal=ThermalConfig(enabled=False)
    )
    session.retrieve(QUERY_AB)
    assert not session.warm, "the ablation switch stores nothing"
    second = session.retrieve(QUERY_AB)
    assert second.propagation.injected_energy == pytest.approx(10.0)


def test_auto_reset_fires_on_zero_contact_overlap() -> None:
    session = ThermalSession(
        FakeIndex(),
        CHAIN,
        RetrievalConfig(),
        thermal=ThermalConfig(auto_reset=True),
    )
    session.retrieve(QUERY_AB)
    # QUERY_X touches only X, which the warm set {A, B} does not contain -
    # the residue would warm the WRONG region, so the session must cool.
    switched = session.retrieve(QUERY_X)
    assert switched.propagation.injected_energy == pytest.approx(10.0), (
        "a topic change must not carry the old topic's warmth"
    )


def test_auto_reset_spares_a_follow_up_in_the_same_region() -> None:
    session = ThermalSession(
        FakeIndex(),
        CHAIN,
        RetrievalConfig(),
        thermal=ThermalConfig(auto_reset=True),
    )
    session.retrieve(QUERY_AB)
    follow_up = session.retrieve(QUERY_AB)
    assert follow_up.propagation.injected_energy == pytest.approx(14.0)


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="residue_ratio"):
        ThermalConfig(residue_ratio=0.0)
    with pytest.raises(ValueError, match="residue_ratio"):
        ThermalConfig(residue_ratio=1.0)
    with pytest.raises(ValueError, match="min_overlap"):
        ThermalConfig(min_overlap=1.5)


def test_a_session_with_no_preference_runs_the_default_profile() -> None:
    """Same rule as `SpiywebIndex.retrieve()`: silence means `explore`."""
    from spiyweb import DEFAULT_PROFILE, PROFILES

    session = ThermalSession(FakeIndex(), CHAIN)
    assert session.profile == DEFAULT_PROFILE
    assert session.config.propagation.damping == PROFILES[DEFAULT_PROFILE].damping


def test_a_session_given_a_config_runs_it_as_given() -> None:
    mine = RetrievalConfig(seed_width=3)
    session = ThermalSession(FakeIndex(), CHAIN, mine)
    assert session.profile == ""
    assert session.config == mine


def test_a_session_profile_overlays_the_given_config() -> None:
    from spiyweb import PRECISE

    mine = RetrievalConfig(seed_width=3, contact_overfetch=7)
    session = ThermalSession(FakeIndex(), CHAIN, mine, profile="precise")
    assert session.profile == "precise"
    assert session.config.seed_width == PRECISE.seed_width
    assert session.config.contact_overfetch == 7


def test_a_session_refuses_an_unknown_profile() -> None:
    with pytest.raises(ValueError, match="compare"):
        ThermalSession(FakeIndex(), CHAIN, profile="fast")
