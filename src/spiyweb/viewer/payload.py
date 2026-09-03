"""One shape for a run on the wire, written once.

Faz 2.5 left a carry-over: the measurement rig and the shipped viewer each
turned a finished run into JSON, in their own copy of the same ninety lines.
The scene builder was already shared (Faz 2.2 promoted `scene.py` for exactly
that reason), but the *serialisation* was not - and a duplicated shape drifts
the same way a duplicated mechanism does, only more quietly: nothing fails, a
field just stops meaning the same thing on one of the two pages.

So the shaping lives here, and both callers hand it their own extras:

- the rig adds the benchmark's titles, entity-derived edge labels and the
  top-k comparison, none of which exist outside a benchmark corpus;
- the viewer adds nothing, because a recorded call already carries what it
  needs.

Plain dicts and not pydantic models: this module is inside the package, and
the package does not depend on pydantic. `server/schemas.py` builds its
`SceneDto` straight from these keys, which is also a test that the two agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spiyweb.scene import ring_radii

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from spiyweb.ledger import Ledger
    from spiyweb.output import ActivationPath, RefusalReport, ThemeCluster
    from spiyweb.scene import GraphScene

__all__ = [
    "clusters_payload",
    "ledger_payload",
    "paths_payload",
    "refusal_payload",
    "scene_payload_of",
]

DEFAULT_ORIGIN = (0.5, 0.5)
"""Where an atom the ring layout never placed is drawn: dead centre. It
cannot be dropped - an edge would then end in empty space."""


def scene_payload_of(
    scene: GraphScene,
    rings: Mapping[str, tuple[float, float]],
    *,
    titles: Mapping[str, str] | None = None,
    legend: Mapping[str, str] | None = None,
    layer_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """A drawn scene as the canvas expects it, force AND ring coordinates.

    Both layouts ride along in every node and edge. The force layout answers
    "what is connected to what"; the rings answer "how far did the energy
    get", which is the question this project is about - and shipping only one
    would make the toggle a second request instead of a repaint.
    """
    names = titles or {}
    return {
        "nodes": [
            {
                "id": node.id,
                "x": node.x,
                "y": node.y,
                "rx": rings.get(node.id, DEFAULT_ORIGIN)[0],
                "ry": rings.get(node.id, DEFAULT_ORIGIN)[1],
                "energy": node.energy,
                "hop": node.hop,
                "votes": node.votes,
                "kind": node.kind,
                "node_layer": node.node_layer,
                "source_id": node.source_id,
                "polarity": node.polarity,
                "disputed": node.disputed,
                "label": node.label,
                "title": names.get(node.id, node.source_id or node.id),
                "tooltip": node.tooltip,
            }
            for node in scene.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "x1": edge.x1,
                "y1": edge.y1,
                "x2": edge.x2,
                "y2": edge.y2,
                "rx1": rings.get(edge.source, DEFAULT_ORIGIN)[0],
                "ry1": rings.get(edge.source, DEFAULT_ORIGIN)[1],
                "rx2": rings.get(edge.target, DEFAULT_ORIGIN)[0],
                "ry2": rings.get(edge.target, DEFAULT_ORIGIN)[1],
                "weight": edge.weight,
                "layer": edge.layer,
                "layers": list(edge.layers),
                "kind": edge.kind,
                "tooltip": edge.tooltip,
            }
            for edge in scene.edges
        ],
        "legend": dict(legend) if legend is not None else dict(scene.legend),
        "layer_order": list(layer_order) if layer_order is not None else [],
        "dropped_nodes": scene.dropped_nodes,
        "dropped_edges": scene.dropped_edges,
        "caption": scene.caption,
        "max_hop": max((node.hop for node in scene.nodes), default=0),
        # The ring radii travel with the scene so the browser draws its guide
        # circles from the SAME rule that placed the atoms. The canvas used to
        # carry a copy of the spacing formula, and the moment the rule changed
        # the guides described a layout that no longer existed.
        "rings": [
            {"hop": hop, "radius": radius}
            for hop, radius in sorted(
                ring_radii(node.hop for node in scene.nodes).items()
            )
        ],
    }


def ledger_payload(
    book: Ledger,
    *,
    dedup_cuts: int | None = None,
    contact_cuts: int = 0,
    contact_tau: float | None = None,
) -> dict[str, Any]:
    """The energy ledger on the wire, including how badly it failed to add up.

    `residual` and `mismatch` are carried and never rounded away: CLAUDE.md
    §2.1 makes an auditable claim, and a reconstruction that quietly hides
    its own disagreement would make the audit worthless.
    """
    return {
        "injected": book.injected,
        "held": book.held,
        "dissipated": book.dissipated,
        "destroyed": {
            "conflict": book.destroyed.conflict,
            "conflict_events": book.destroyed.conflict_events,
            "negative_seed": book.destroyed.negative_seed,
            "negative_seed_events": book.destroyed.negative_seed_events,
            "polarity": book.destroyed.polarity,
            "polarity_events": book.destroyed.polarity_events,
            "total": book.destroyed.total,
        },
        "residual": book.residual,
        "residual_share": book.residual / book.injected if book.injected else 0.0,
        "mismatch": book.mismatch,
        "tolerance": book.tolerance,
        "balanced": book.balanced,
        "exact": book.exact,
        "notes": list(book.notes),
        "dedup_cuts": book.dedup_cuts if dedup_cuts is None else dedup_cuts,
        "contact_cuts": contact_cuts,
        "dedup_taus": list(book.dedup_taus),
        "contact_tau": contact_tau,
    }


def paths_payload(
    paths: Sequence[ActivationPath],
    *,
    labels: Mapping[tuple[str, str], str] | None = None,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Activation paths, rendered. D19 says these go to the LLM, not a log."""
    return [
        {
            "node": path.node,
            "steps": list(path.steps),
            "hop": path.hop,
            "energy": path.energy,
            "converging": path.converging,
            "rendered": path.rendered(labels),
        }
        for path in paths[:limit]
    ]


def clusters_payload(
    clusters: Sequence[ThemeCluster], *, limit: int = 12
) -> list[dict[str, Any]]:
    return [
        {
            "nodes": list(cluster.nodes),
            "energy": cluster.energy,
            "energy_share": cluster.energy_share,
            "top_node": cluster.top_node,
        }
        for cluster in clusters[:limit]
    ]


def refusal_payload(report: RefusalReport, *, limit: int = 8) -> dict[str, Any]:
    return {
        "stop_reason": report.stop_reason,
        "hop_depth": report.hop_depth,
        "deepest_nodes": list(report.deepest_nodes[:limit]),
        "text": report.text,
    }
