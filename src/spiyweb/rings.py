"""The one ring rule: concentric hops, importable with nothing installed.

`scene.py` places atoms on rings by hop and the terminal monitor draws the
same rings from the same rule - one mechanism, one implementation, which is
the discipline the scene builder itself exists for (CLAUDE.md §3, rule 4).
The rule used to live inside `scene.py`, next to numpy, and the monitor may
not import numpy; so the rule moved here and the scene imports it back.

Determinism is the contract: the same query over the same index must place
the same atoms at the same angles, on any machine. That is why the starting
angle comes from `blake2b`, never `hash()` (salted per process), and from a
`random.Random` seeded by it rather than the system generator.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "RING_GHOST_GAP",
    "RING_INNER",
    "RING_OUTER",
    "RING_SINGLE",
    "hop_ring_layout",
    "layout_seed",
    "ring_radii",
]


def layout_seed(node_ids: Sequence[str], salt: str, base: int) -> int:
    """Stable RNG seed from the drawn id set and the query - never `hash()`.

    `hash()` is salted per process (PYTHONHASHSEED), so a layout keyed on it
    would silently differ between runs on the same machine. `blake2b` is
    deterministic everywhere, which is what the "same query, same picture"
    contract needs.
    """
    payload = "\n".join(sorted(node_ids)) + "\x00" + salt
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "big") ^ (base & 0xFFFFFFFFFFFFFFFF)) & 0x7FFFFFFF


def hop_ring_layout(
    node_ids: Sequence[str],
    hops: Mapping[str, int],
    energies: Mapping[str, float],
    *,
    salt: str = "",
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Concentric rings by hop: the decay made spatial.

    The force layout answers "what is connected to what"; this one answers
    "how far did the energy get", which is the question the whole project is
    about. Radius is the hop number, so hop 0 sits at the centre and the last
    ring is the frontier where the web died.

    Within a ring, atoms are ordered by energy (strongest first) and spread
    evenly around the circle, so the eye reads the ring as a ranking. The
    starting angle comes from `layout_seed`, so this shares the spring
    layout's determinism contract: same query, same picture.
    """
    ids = sorted(set(node_ids))
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: (0.5, 0.5)}

    rings: dict[int, list[str]] = {}
    for node_id in ids:
        rings.setdefault(int(hops.get(node_id, 0)), []).append(node_id)

    # Rings are spaced by their POSITION in the sequence of hop levels, not by
    # the hop number, and only activated levels set the scale.
    #
    # The old rule was `0.06 + 0.44 * (hop / deepest)`, and it had two faults
    # that only showed on real data. A web that stopped at first contact has
    # one level, so every atom landed on the innermost ring - a blob six
    # percent of the plate wide, with the labels inside each other, which is
    # what a reader saw for any single-hop query. And a suppressed atom
    # carries hop -1, which made its radius NEGATIVE; it still drew, mirrored
    # through the centre, but by accident rather than by decision.
    levels = sorted(hop for hop in rings if hop >= 0)
    order = {hop: index for index, hop in enumerate(levels)}

    rng = random.Random(layout_seed(ids, salt, seed))
    placed: dict[str, tuple[float, float]] = {}
    for hop in sorted(rings):
        members = sorted(
            rings[hop], key=lambda node: (-float(energies.get(node, 0.0)), node)
        )
        if hop == 0 and len(members) == 1 and len(levels) > 1:
            # The single seed of a web that DID travel belongs at the centre.
            placed[members[0]] = (0.5, 0.5)
            continue
        radius = _ring_radius(hop, order, len(levels))
        offset = rng.random() * 2.0 * math.pi
        for position, node_id in enumerate(members):
            angle = offset + 2.0 * math.pi * position / len(members)
            placed[node_id] = (
                0.5 + radius * math.cos(angle),
                0.5 + radius * math.sin(angle),
            )
    return placed


RING_INNER = 0.10
"""Radius of the first ring when the web travelled. Not zero: hop 0 is a set
of atoms, and stacking them on the centre point would hide all but one."""

RING_OUTER = 0.44
"""Radius of the frontier ring, leaving the corners for labels."""

RING_SINGLE = 0.32
"""The one ring of a web that stopped at first contact. It IS the drawing, so
it takes the room a drawing needs - the old rule put it at 0.10 and the whole
picture became a knot."""

RING_GHOST_GAP = 0.06
"""How far outside the frontier a suppressed atom sits.

Relative to the outermost activated ring rather than an absolute radius: a
fixed 0.50 put ghosts hard against the edge of the unit square, and on a
single-ring web - where the frontier is at 0.32 - that stranded them halfway
across the plate from anything they were cut from. A suppressed atom was cut
OUT of the web, so just beyond the last ring is the honest place for it, and
the dashed line back to its survivor then reads as the cut it is."""


def _ring_radius(hop: int, order: Mapping[int, int], levels: int) -> float:
    """Where this hop's ring sits, in normalised radius."""
    if hop < 0:
        return (RING_SINGLE if levels <= 1 else RING_OUTER) + RING_GHOST_GAP
    if levels <= 1:
        return RING_SINGLE
    step = order.get(hop, 0) / (levels - 1)
    return RING_INNER + (RING_OUTER - RING_INNER) * step


def ring_radii(hops: Iterable[int]) -> dict[int, float]:
    """The radius of every ring the layout will draw, keyed by hop.

    Exposed so a drawing surface can draw its guide circles from the SAME
    rule that placed the atoms. The browser canvas once carried its own copy
    of the spacing formula; the moment this one changed, the guides described
    a layout that no longer existed and the rings sat where nothing was.
    """
    present = sorted({int(hop) for hop in hops})
    levels = sorted(hop for hop in present if hop >= 0)
    order = {hop: index for index, hop in enumerate(levels)}
    return {hop: _ring_radius(hop, order, len(levels)) for hop in present}
