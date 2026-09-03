"""Hop rings: the decay made spatial, under the same determinism contract."""

from __future__ import annotations

import math

import pytest

from spiyweb.scene import hop_ring_layout

_HOPS = {"a": 0, "b": 1, "c": 1, "d": 2, "e": 2, "f": 2}
_ENERGY = {"a": 5.0, "b": 3.0, "c": 2.0, "d": 1.5, "e": 1.2, "f": 1.0}
_NODES = list(_HOPS)


def test_layout_is_deterministic() -> None:
    assert hop_ring_layout(_NODES, _HOPS, _ENERGY) == hop_ring_layout(
        _NODES, _HOPS, _ENERGY
    )


def test_input_order_does_not_change_the_picture() -> None:
    assert hop_ring_layout(_NODES, _HOPS, _ENERGY) == hop_ring_layout(
        list(reversed(_NODES)), _HOPS, _ENERGY
    )


def test_a_lone_seed_sits_at_the_centre() -> None:
    placed = hop_ring_layout(_NODES, _HOPS, _ENERGY)
    assert placed["a"] == (0.5, 0.5)


def test_radius_grows_with_hop() -> None:
    placed = hop_ring_layout(_NODES, _HOPS, _ENERGY)

    def radius(node: str) -> float:
        x, y = placed[node]
        return math.hypot(x - 0.5, y - 0.5)

    assert radius("b") < radius("d")
    assert (
        radius("b") == round(radius("c"), 12) or abs(radius("b") - radius("c")) < 1e-9
    )


def test_every_point_stays_inside_the_unit_square() -> None:
    for x, y in hop_ring_layout(_NODES, _HOPS, _ENERGY).values():
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_different_query_rotates_the_rings() -> None:
    assert hop_ring_layout(_NODES, _HOPS, _ENERGY, salt="q1") != hop_ring_layout(
        _NODES, _HOPS, _ENERGY, salt="q2"
    )


def test_degenerate_inputs_are_not_errors() -> None:
    assert hop_ring_layout([], {}, {}) == {}
    assert hop_ring_layout(["only"], {"only": 0}, {"only": 1.0}) == {"only": (0.5, 0.5)}


def test_unknown_hops_default_to_the_centre_ring() -> None:
    placed = hop_ring_layout(["x", "y"], {}, {})
    assert len(placed) == 2
    assert all(math.isfinite(value) for point in placed.values() for value in point)


def test_a_single_hop_web_uses_the_whole_ring_rather_than_a_knot() -> None:
    """A web that stopped at first contact IS the drawing.

    The old rule spaced rings by `hop / deepest`, so one level put every atom
    on the innermost ring - a blob six percent of the plate wide with the
    labels inside each other. That is what a reader saw for any query whose
    energy never left the seeds, which on the shipped defaults was most of
    them.
    """
    from spiyweb.scene import RING_SINGLE, hop_ring_layout

    ids = [f"n{i}" for i in range(6)]
    placed = hop_ring_layout(ids, dict.fromkeys(ids, 0), dict.fromkeys(ids, 1.0))
    spans = [
        max(p[axis] for p in placed.values()) - min(p[axis] for p in placed.values())
        for axis in (0, 1)
    ]
    assert max(spans) > RING_SINGLE, spans


def test_a_suppressed_atom_sits_outside_the_frontier() -> None:
    """It was cut OUT of the web; a negative radius placed it by accident."""
    import math

    from spiyweb.scene import RING_OUTER, hop_ring_layout

    hops = {"seed": 0, "far": 1, "ghost": -1}
    energies = {"seed": 3.0, "far": 1.0, "ghost": 0.0}
    placed = hop_ring_layout(list(hops), hops, energies)
    distance = {node: math.hypot(x - 0.5, y - 0.5) for node, (x, y) in placed.items()}
    assert distance["ghost"] > distance["far"] >= RING_OUTER - 1e-9
    assert distance["ghost"] > 0.0


def test_rings_are_spaced_by_position_not_by_hop_number() -> None:
    """A web whose hops skip a level must still space its rings evenly."""
    import math

    from spiyweb.scene import RING_INNER, RING_OUTER, hop_ring_layout

    hops = {"a": 0, "b": 0, "c": 5, "d": 5}
    placed = hop_ring_layout(list(hops), hops, dict.fromkeys(hops, 1.0))
    radius = {n: math.hypot(x - 0.5, y - 0.5) for n, (x, y) in placed.items()}
    assert radius["a"] == pytest.approx(RING_INNER, abs=1e-9)
    assert radius["c"] == pytest.approx(RING_OUTER, abs=1e-9)
