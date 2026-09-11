"""The ring rule lives outside numpy now; the scene still sees it."""

from __future__ import annotations

import math
import subprocess
import sys

from spiyweb.rings import RING_INNER, RING_OUTER, hop_ring_layout, ring_radii

_NODES = ["a", "b", "c", "d", "e"]
_HOPS = {"a": 0, "b": 1, "c": 1, "d": 2, "e": 2}
_ENERGY = {"a": 5.0, "b": 3.0, "c": 2.0, "d": 1.5, "e": 1.0}


def test_layout_is_deterministic() -> None:
    assert hop_ring_layout(_NODES, _HOPS, _ENERGY) == hop_ring_layout(
        _NODES, _HOPS, _ENERGY
    )


def test_input_order_does_not_change_the_picture() -> None:
    assert hop_ring_layout(_NODES, _HOPS, _ENERGY) == hop_ring_layout(
        list(reversed(_NODES)), _HOPS, _ENERGY
    )


def test_a_lone_seed_sits_at_the_centre_and_radius_grows_with_hop() -> None:
    placed = hop_ring_layout(_NODES, _HOPS, _ENERGY)
    assert placed["a"] == (0.5, 0.5)

    def radius(node: str) -> float:
        x, y = placed[node]
        return math.hypot(x - 0.5, y - 0.5)

    assert radius("b") < radius("d")
    assert abs(radius("b") - radius("c")) < 1e-9


def test_the_layout_agrees_with_the_guide_rings() -> None:
    placed = hop_ring_layout(_NODES, _HOPS, _ENERGY)
    radii = ring_radii(_HOPS.values())
    for node, hop in _HOPS.items():
        if node == "a":
            continue
        x, y = placed[node]
        assert math.hypot(x - 0.5, y - 0.5) == pytest_approx(radii[hop])
    assert radii[0] == pytest_approx(RING_INNER)
    assert radii[2] == pytest_approx(RING_OUTER)
    assert RING_INNER < radii[1] < RING_OUTER


def pytest_approx(value: float) -> object:
    import pytest

    return pytest.approx(value, abs=1e-9)


def test_a_different_query_rotates_the_rings() -> None:
    assert hop_ring_layout(_NODES, _HOPS, _ENERGY, salt="q1") != hop_ring_layout(
        _NODES, _HOPS, _ENERGY, salt="q2"
    )


def test_the_scene_module_re_exports_the_same_functions() -> None:
    import pytest

    scene = pytest.importorskip("spiyweb.scene")
    assert scene.hop_ring_layout is hop_ring_layout
    assert scene.ring_radii is ring_radii


_PROBE = """
import sys
BANNED = ("numpy", "faiss", "torch", "spacy", "transformers")
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BANNED:
            raise ImportError(name + " is blocked")
        return None
sys.meta_path.insert(0, _Blocker())
from spiyweb.rings import hop_ring_layout
placed = hop_ring_layout(["a", "b"], {"a": 0, "b": 1}, {"a": 2.0, "b": 1.0})
assert set(placed) == {"a", "b"}
print("ok")
"""


def test_rings_import_without_numpy() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout
