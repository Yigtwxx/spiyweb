"""Render-agnostic view layer of the activated web: layout, scene, renderers.

No I/O and no rendering library live here on purpose: everything worth
testing must be importable with numpy alone. That is what lets one query
produce one picture no matter which front end asked - the browser face reads
this module rather than reimplementing any of it in TypeScript. The split is
"pure / shell", not "graph / page".

numpy is the reason this is not imported from `spiyweb/__init__.py`: the
package promises `import spiyweb` with nothing installed, so this module
arrives through `spiyweb[view]` and an explicit `import spiyweb.scene`.

Determinism is this module's contract, not the drawing library's: the same
query over the same index must produce a bit-identical picture. That is why
the spring layout lives here, seeded from a hash of the drawn id set, and why
`hash()` is never used (it is salted per process).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

    from numpy.typing import NDArray

    from spiyweb.config import LayerWeights
    from spiyweb.core.dedup import SimilarityFn
    from spiyweb.core.graph import Graph
    from spiyweb.core.propagate import Activation

EDGE_LAYER_ORDER: tuple[str, ...] = (
    "semantic",
    "entity",
    "structural",
    "derivation",
    "learned",
)
"""Fixed layer order: the colour scale's domain and every tie-break.

Held constant so a node's edge keeps its colour from one query to the next -
a legend that reshuffles per query cannot be read.
"""

LAYER_COLORS: dict[str, str] = {
    "entity": "#7fb3ff",
    "semantic": "#3f7fd8",
    "derivation": "#c3d8ff",
    "structural": "#8fa0bc",
    "learned": "#ff4d5e",
}
"""One stable colour per edge layer, keyed by `EDGE_LAYER_ORDER`.

Within the palette the interface uses (navy, blue, red, black, white), so the
two front ends and the page around them read as one drawing. Four blues plus
one red, and the red is deliberate: `learned` is the only layer that writes
back from usage, so the layer that can drift is the one that stands out. The
brightest blue goes to `entity`, the main hop fuel.

Colour is never the only carrier - the legend names every layer and each edge
tooltip spells out which layers it merges.
"""


@dataclass(frozen=True)
class LayoutConfig:
    """Fruchterman-Reingold knobs; every default is a documented tunable.

    Attributes:
        iterations: Fixed number of relaxation rounds. Fixed, never
            time-bounded: an early exit on elapsed time would make the picture
            depend on machine load.
        seed: Base RNG seed, combined with a hash of the drawn ids so that
            two different node sets do not start from the same cloud.
        gravity: Pull of every node toward the centroid. Mandatory, not
            cosmetic: the activated subgraph splits into connected components
            (that is what `theme_clusters` reports), and plain FR lets
            disconnected components drift apart without bound.
        initial_temperature: Maximum displacement per round, cooling linearly
            to zero over `iterations`.
        min_distance: Floor on pairwise distance, so two nodes landing on the
            same point cannot divide by zero.
    """

    iterations: int = 200
    seed: int = 0
    gravity: float = 0.05
    initial_temperature: float = 0.10
    min_distance: float = 1e-4

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if not 0.0 <= self.gravity <= 1.0:
            raise ValueError("gravity must lie in [0, 1]")
        if self.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")
        if self.min_distance <= 0.0:
            raise ValueError("min_distance must be positive")


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


def _edge_arrays(
    edges: Sequence[tuple[str, str, float]],
    position: Mapping[str, int],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Canonical `(lo, hi, weight)` index arrays over the drawn nodes.

    Edges touching a node that is not drawn are dropped, self-edges are
    dropped, and each pair is emitted with `lo < hi` then sorted - so the
    arrays depend on the edge SET, never on the order the caller listed it in.
    """
    collected: dict[tuple[int, int], float] = {}
    for source, target, weight in edges:
        if source not in position or target not in position:
            continue
        a, b = position[source], position[target]
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        collected[key] = max(collected.get(key, 0.0), float(weight))
    if not collected:
        empty_index: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        return empty_index, empty_index.copy(), np.empty(0, dtype=np.float64)
    ordered = sorted(collected.items())
    sources = np.array([pair[0] for pair, _ in ordered], dtype=np.int64)
    targets = np.array([pair[1] for pair, _ in ordered], dtype=np.int64)
    weights = np.array([weight for _, weight in ordered], dtype=np.float64)
    return sources, targets, weights


def _repulsion(
    positions: NDArray[np.float64], k: float, min_distance: float
) -> NDArray[np.float64]:
    """All-pairs repulsive displacement, `k**2 / d` along each delta."""
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    distance = np.maximum(distance, min_distance)
    scale = (k * k) / (distance * distance)
    return np.einsum("ijd,ij->id", delta, scale)


def _attraction(
    positions: NDArray[np.float64],
    sources: NDArray[np.int64],
    targets: NDArray[np.int64],
    weights: NDArray[np.float64],
    k: float,
    min_distance: float,
) -> NDArray[np.float64]:
    """Edge-wise attractive displacement, `w * d**2 / k`, scattered in place."""
    displacement = np.zeros_like(positions)
    if sources.size == 0:
        return displacement
    delta = positions[sources] - positions[targets]
    distance = np.maximum(np.linalg.norm(delta, axis=-1), min_distance)
    magnitude = weights * (distance * distance) / k
    pull = delta / distance[:, None] * magnitude[:, None]
    np.add.at(displacement, sources, -pull)
    np.add.at(displacement, targets, pull)
    return displacement


def _rescale(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit the cloud into `[0, 1]^2`; a degenerate axis lands at the centre."""
    rescaled = np.empty_like(positions)
    for axis in range(positions.shape[1]):
        column = positions[:, axis]
        low, high = float(column.min()), float(column.max())
        span = high - low
        if span <= 0.0:
            rescaled[:, axis] = 0.5
        else:
            rescaled[:, axis] = (column - low) / span
    return rescaled


def _components(
    ids: Sequence[str], edges: Sequence[tuple[str, str, float]]
) -> list[list[str]]:
    """Connected components, largest first, ties broken by first id.

    An isolated atom is its own component. Order is fully determined by the
    ids, never by the caller's listing order.
    """
    parent = {node_id: node_id for node_id in ids}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for source, target, _ in edges:
        if source in parent and target in parent:
            a, b = find(source), find(target)
            if a != b:
                parent[max(a, b)] = min(a, b)

    grouped: dict[str, list[str]] = {}
    for node_id in ids:
        grouped.setdefault(find(node_id), []).append(node_id)
    return sorted(grouped.values(), key=lambda group: (-len(group), group[0]))


def _shelf_boxes(sizes: Sequence[int]) -> list[tuple[float, float, float, float]]:
    """Pack one box per component into the unit square, biggest first.

    Each box gets a side proportional to `sqrt(size)`, so a 30-atom component
    is not handed the same room as a stray pair, and rows are broken once they
    reach the width of a square of the same total area. Returns
    `(x, y, width, height)` per component, already fitted to `[0, 1]^2`.
    """
    sides = [float(np.sqrt(size)) for size in sizes]
    target = float(np.sqrt(sum(size for size in sizes))) or 1.0
    rows: list[list[float]] = [[]]
    row_width = 0.0
    for side in sides:
        if rows[-1] and row_width + side > target:
            rows.append([])
            row_width = 0.0
        rows[-1].append(side)
        row_width += side

    # Each row is stretched to the full width afterwards. Without it the last
    # row sets the scale for every other, and the biggest component - the one
    # the reader actually studies - can end up in 70% of the frame while the
    # right edge sits empty.
    heights = [max(row) for row in rows]
    total_height = sum(heights) or 1.0
    boxes: list[tuple[float, float, float, float]] = []
    cursor_y = 0.0
    for row, height in zip(rows, heights, strict=True):
        stretch = 1.0 / (sum(row) or 1.0)
        cursor_x = 0.0
        for side in row:
            boxes.append(
                (
                    cursor_x,
                    cursor_y / total_height,
                    side * stretch,
                    height / total_height,
                )
            )
            cursor_x += side * stretch
        cursor_y += height
    return boxes


def spring_layout(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
    config: LayoutConfig | None = None,
    *,
    initial: Mapping[str, tuple[float, float]] | None = None,
    salt: str = "",
) -> dict[str, tuple[float, float]]:
    """Deterministic Fruchterman-Reingold positions in the unit square.

    Ids are sorted internally, so the caller's listing order never reaches the
    picture - the same rule the rest of the project uses to break ties. The
    RNG is a local `default_rng`, never the global `np.random` state, so an
    unrelated caller cannot shift the layout.

    `initial` is an optional warm start (the previous frame's positions) that
    buys visual continuity while a slider moves, at the price of making the
    result path-dependent; the cold start is the canonical behaviour.
    """
    cfg = config if config is not None else LayoutConfig()
    ids = sorted(set(node_ids))
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: (0.5, 0.5)}

    # Disconnected pieces have nothing pulling them together, so plain
    # repulsion throws them into opposite corners and the middle of the frame
    # goes empty - a 36-atom web was using about a quarter of the canvas. Each
    # component is solved on its own and the results are packed side by side,
    # which uses the room and keeps the separation legible instead of extreme.
    # A single connected web takes the original path untouched.
    groups = _components(ids, edges)
    if len(groups) > 1:
        by_group = {
            node_id: index for index, group in enumerate(groups) for node_id in group
        }
        placed: dict[str, tuple[float, float]] = {}
        boxes = _shelf_boxes([len(group) for group in groups])
        for index, (group, (box_x, box_y, box_w, box_h)) in enumerate(
            zip(groups, boxes, strict=True)
        ):
            inner = spring_layout(
                group,
                [
                    edge
                    for edge in edges
                    if by_group.get(edge[0]) == index and by_group.get(edge[1]) == index
                ],
                cfg,
                initial=initial,
                salt=f"{salt}#component{index}",
            )
            # A margin inside each box so neighbouring components read as
            # separate clouds rather than one smear.
            pad = 0.06
            for node_id, (x, y) in inner.items():
                placed[node_id] = (
                    box_x + box_w * (pad + (1.0 - 2 * pad) * x),
                    box_y + box_h * (pad + (1.0 - 2 * pad) * y),
                )
        return placed

    position = {node_id: index for index, node_id in enumerate(ids)}
    rng = np.random.default_rng(layout_seed(ids, salt, cfg.seed))
    points = rng.random((len(ids), 2))
    if initial is not None:
        for node_id, index in position.items():
            start = initial.get(node_id)
            if start is not None:
                points[index] = start

    sources, targets, weights = _edge_arrays(edges, position)
    k = float(np.sqrt(1.0 / len(ids)))
    for step in range(cfg.iterations):
        temperature = cfg.initial_temperature * (1.0 - step / cfg.iterations)
        displacement = _repulsion(points, k, cfg.min_distance)
        displacement += _attraction(
            points, sources, targets, weights, k, cfg.min_distance
        )
        displacement -= cfg.gravity * (points - points.mean(axis=0))
        length = np.maximum(np.linalg.norm(displacement, axis=-1), cfg.min_distance)
        limited = np.minimum(length, temperature)
        points += displacement / length[:, None] * limited[:, None]
        # No clipping to the unit square inside the loop. Clamping every round
        # pins nodes against the walls - repulsion keeps pushing outward and
        # the frame keeps them there, which is exactly what the first drawn
        # graph looked like. The cloud is free to expand and `_rescale` fits
        # whatever shape it settles into afterwards.

    final = _rescale(points)
    return {
        node_id: (float(final[index, 0]), float(final[index, 1]))
        for node_id, index in position.items()
    }


def _encode(lo: int, hi: int, stride: int) -> int:
    """`min * stride + max` - a canonical, undirected, order-free pair code."""
    return lo * stride + hi


@dataclass(frozen=True)
class EdgeLayerIndex:
    """Per-layer edge weights, keyed by a sorted pair code.

    The merged adjacency deliberately forgets which layer an edge came from
    (`Graph.from_layers` sums them), but the inspector has to colour edges by
    layer, so the layer memory is rebuilt here from the raw artifacts.

    The encoding is not a micro-optimisation: the MuSiQue index carries 632838
    entity edges and 43405 semantic ones, and a plain `dict[(u, v)] -> layer`
    over that runs to roughly 100 MB and pickles slowly. Two sorted numpy
    arrays per layer cost about 12 MB in total and answer a lookup with one
    `searchsorted`.

    Attributes:
        node_ids: Every node the index can talk about, in sorted order.
        position: Node id -> its row, the encoding's alphabet.
        codes: Layer -> its pair codes, ASCENDING (searchsorted needs it).
        layer_weights: Layer -> raw within-layer weight, aligned with `codes`.
        counts: Layer -> edge count, so the UI can grey out an empty layer
            instead of offering a slider that does nothing.
    """

    node_ids: tuple[str, ...]
    position: Mapping[str, int]
    codes: Mapping[str, NDArray[np.int64]]
    layer_weights: Mapping[str, NDArray[np.float64]]
    counts: Mapping[str, int]

    def raw_weight(self, layer: str, source: str, target: str) -> float | None:
        """Within-layer weight of the undirected pair, or `None` if absent."""
        codes = self.codes.get(layer)
        if codes is None or codes.size == 0:
            return None
        a, b = self.position.get(source), self.position.get(target)
        if a is None or b is None or a == b:
            return None
        code = _encode(min(a, b), max(a, b), len(self.node_ids))
        found = int(np.searchsorted(codes, code))
        if found >= codes.size or int(codes[found]) != code:
            return None
        return float(self.layer_weights[layer][found])

    def contributions(
        self, source: str, target: str, weights: LayerWeights
    ) -> dict[str, float]:
        """Per-layer contribution `weight_of(L) * w_L(u, v)` - the tooltip.

        A layer whose weight is `0.0` is omitted entirely, matching
        `Graph.from_layers`: a disabled layer is not in the graph at all, so
        claiming it holds the edge would be a lie the picture tells.
        """
        found: dict[str, float] = {}
        for layer in EDGE_LAYER_ORDER:
            layer_weight = weights.weight_of(layer)  # type: ignore[arg-type]
            if layer_weight == 0.0:
                continue
            raw = self.raw_weight(layer, source, target)
            if raw is None:
                continue
            found[layer] = layer_weight * raw
        return found

    def layer_of(
        self, source: str, target: str, weights: LayerWeights
    ) -> tuple[str | None, tuple[str, ...]]:
        """Dominant layer of the edge under `weights`, plus every layer holding it.

        "Dominant" is the largest weighted contribution - the layer actually
        dragging this edge into the merged graph - so turning a layer weight
        down recolours the edges it stops driving. Ties break on
        `EDGE_LAYER_ORDER`, never on dict order.
        """
        found = self.contributions(source, target, weights)
        if not found:
            return None, ()
        best: str | None = None
        best_value = -1.0
        for layer in EDGE_LAYER_ORDER:
            value = found.get(layer)
            if value is not None and value > best_value:
                best, best_value = layer, value
        return best, tuple(layer for layer in EDGE_LAYER_ORDER if layer in found)


def build_layer_index(
    node_ids: Sequence[str],
    layers: Mapping[str, Sequence[tuple[str, str, float]]],
) -> EdgeLayerIndex:
    """Encode every layer's canonical `(u < v)` pairs as sorted codes.

    Edges naming a node outside `node_ids` are dropped rather than raising:
    an index built before a later artifact landed is a normal state here, and
    a developer tool must open it, not refuse it.
    """
    ordered_ids = tuple(sorted(set(node_ids)))
    position = {node_id: index for index, node_id in enumerate(ordered_ids)}
    stride = len(ordered_ids)
    codes: dict[str, NDArray[np.int64]] = {}
    layer_weights: dict[str, NDArray[np.float64]] = {}
    counts: dict[str, int] = {}
    for layer in EDGE_LAYER_ORDER:
        edges = layers.get(layer, ())
        collected: dict[int, float] = {}
        for source, target, weight in edges:
            a, b = position.get(source), position.get(target)
            if a is None or b is None or a == b:
                continue
            code = _encode(min(a, b), max(a, b), stride)
            collected[code] = max(collected.get(code, 0.0), float(weight))
        counts[layer] = len(collected)
        if not collected:
            codes[layer] = np.empty(0, dtype=np.int64)
            layer_weights[layer] = np.empty(0, dtype=np.float64)
            continue
        keys = np.fromiter(sorted(collected), dtype=np.int64, count=len(collected))
        codes[layer] = keys
        layer_weights[layer] = np.array(
            [collected[int(code)] for code in keys], dtype=np.float64
        )
    return EdgeLayerIndex(
        node_ids=ordered_ids,
        position=position,
        codes=codes,
        layer_weights=layer_weights,
        counts=counts,
    )


NODE_KIND_COLORS: dict[str, str] = {
    "seed": "#d62728",
    "bridge": "#9467bd",
    "activated": "#1f77b4",
    "suppressed": "#bab0ac",
}
"""Node role -> colour. `suppressed` is the ghost a dashed edge points at."""

NODE_KIND_ORDER: tuple[str, ...] = ("bridge", "seed", "activated", "suppressed")
"""Precedence when a node qualifies as more than one role."""

EDGE_KINDS: tuple[str, ...] = ("active", "suppressed")
"""`suppressed` is drawn DASHED - the redundancy link dedup cut."""


@dataclass(frozen=True)
class ViewConfig:
    """What the picture is allowed to contain - the performance guardrails.

    Attributes:
        max_nodes: Drawing cap, deliberately BELOW `PropagationConfig.
            max_nodes` (512): the propagation may legitimately light up more
            atoms than a readable picture can hold, and the honest answer is
            to draw the strongest ones and say how many were dropped.
        max_edges: Cap on ordinary edges. Dedup links are exempt - they are a
            required element of this view, never silently trimmed.
        edge_mode: `"contributors"` draws only the links energy actually
            crossed (the causal skeleton, about one edge per node);
            `"induced"` draws the full induced subgraph, which on the entity
            layer (average degree around 107) can reach tens of thousands of
            edges before the cap bites.
        label_top_n: How many of the strongest nodes carry a visible label.
        height: Chart height in pixels.
    """

    max_nodes: int = 300
    max_edges: int = 1500
    edge_mode: str = "contributors"
    label_top_n: int = 15
    height: int = 620

    def __post_init__(self) -> None:
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")
        if self.max_edges < 0:
            raise ValueError("max_edges must not be negative")
        if self.edge_mode not in ("contributors", "induced"):
            raise ValueError(
                f"edge_mode {self.edge_mode!r} must be 'contributors' or 'induced'"
            )
        if self.label_top_n < 0:
            raise ValueError("label_top_n must not be negative")
        if self.height < 1:
            raise ValueError("height must be at least 1")


@dataclass(frozen=True)
class SceneNode:
    """One drawable atom, with everything a mark or a tooltip needs."""

    id: str
    x: float
    y: float
    energy: float
    hop: int
    votes: int
    kind: str
    node_layer: str
    source_id: str
    polarity: int
    disputed: bool
    label: str
    tooltip: str


@dataclass(frozen=True)
class SceneEdge:
    """One drawable link; `kind == "suppressed"` means DASHED (a cut duplicate)."""

    source: str
    target: str
    x1: float
    y1: float
    x2: float
    y2: float
    weight: float
    layer: str
    layers: tuple[str, ...]
    kind: str
    tooltip: str


@dataclass(frozen=True)
class GraphScene:
    """A complete, render-agnostic picture plus what it had to leave out."""

    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]
    legend: dict[str, str]
    dropped_nodes: int
    dropped_edges: int
    caption: str


def select_subgraph(
    activations: Mapping[str, object], energies: Mapping[str, float], *, limit: int
) -> tuple[tuple[str, ...], int]:
    """The `limit` strongest activated nodes (ties on id) and how many fell out."""
    ranked = sorted(activations, key=lambda node: (-energies.get(node, 0.0), node))
    kept = tuple(sorted(ranked[:limit]))
    return kept, max(0, len(ranked) - len(kept))


def contributor_edges(
    activations: Mapping[str, Activation], nodes: Collection[str]
) -> list[tuple[str, str]]:
    """The causal skeleton: one link per `Activation.contributors` entry.

    This is what "the web spread here" actually means, and it is about one
    edge per node - which is why it is the default. The induced subgraph
    shows every link that EXISTS between the drawn atoms, a different and far
    denser question.
    """
    drawn = set(nodes)
    found: set[tuple[str, str]] = set()
    for node, activation in activations.items():
        if node not in drawn:
            continue
        for contributor in activation.contributors:
            if contributor in drawn and contributor != node:
                found.add((contributor, node))
    return sorted(found)


def induced_edges(
    graph: Graph, nodes: Collection[str], *, limit: int
) -> tuple[list[tuple[str, str, float]], int]:
    """Strongest `limit` edges of the induced subgraph, plus the dropped count.

    A weight of exactly `0.0` is a dedup-suppressed edge, not a link: it is
    excluded here and drawn separately, dashed.
    """
    drawn = set(nodes)
    collected: dict[tuple[str, str], float] = {}
    for node in drawn:
        for neighbor, weight in graph.neighbors(node).items():
            if neighbor not in drawn or neighbor == node or weight <= 0.0:
                continue
            key = (node, neighbor) if node < neighbor else (neighbor, node)
            collected[key] = max(collected.get(key, 0.0), float(weight))
    ranked = sorted(collected.items(), key=lambda item: (-item[1], item[0]))
    kept = [(source, target, weight) for (source, target), weight in ranked[:limit]]
    return kept, max(0, len(ranked) - len(kept))


def _snippet(text: str, chars: int) -> str:
    """One-line preview of a passage, ellipsised at `chars`."""
    flat = " ".join(text.split())
    return flat if len(flat) <= chars else flat[: max(0, chars - 1)].rstrip() + "…"


def _vote_of(node: str, votes: Mapping[str, int], graph: Graph) -> int:
    """Corpus support of the idea this atom carries; votes key per SOURCE.

    The design counts votes per document, never per chunk, so the lookup
    falls back to the node's `source_id` before defaulting to the implicit 1.
    """
    direct = votes.get(node)
    if direct is not None:
        return direct
    data = graph.node(node)
    if data is not None:
        return votes.get(data.source_id, 1)
    return 1


def _node_kind(
    node: str, seeds: Collection[str], bridges: Collection[str], ghost: bool
) -> str:
    if ghost:
        return "suppressed"
    if node in bridges:
        return "bridge"
    if node in seeds:
        return "seed"
    return "activated"


def build_scene(
    *,
    activations: Mapping[str, Activation],
    seeds: Mapping[str, float],
    votes: Mapping[str, int],
    suppressed: Mapping[str, str],
    graph: Graph,
    layer_index: EdgeLayerIndex,
    weights: LayerWeights,
    view: ViewConfig | None = None,
    layout: LayoutConfig | None = None,
    salt: str = "",
    texts: Mapping[str, str] | None = None,
    bridges: Collection[str] = (),
    disputed: Collection[str] = (),
    snippet_chars: int = 160,
) -> GraphScene:
    """Assemble the drawable scene: nodes, layer-coloured edges, dashed cuts.

    The primitives arrive unpacked rather than as a `RetrievalResult` so the
    coloured path (whose ledgers are per colour) feeds the same builder.

    Two elements are required by the inspector's scope and therefore never
    trimmed: the dashed dedup links, and the ghost nodes they point at. A
    suppressed atom never activates, so without the ghost the dashed line
    would end in empty space.
    """
    cfg = view if view is not None else ViewConfig()
    energies = {node: act.energy for node, act in activations.items()}
    drawn, dropped_nodes = select_subgraph(activations, energies, limit=cfg.max_nodes)

    cuts = {
        duplicate: survivor
        for duplicate, survivor in suppressed.items()
        if survivor in drawn
    }
    ghosts = tuple(sorted(set(cuts) - set(drawn)))
    placed_ids = (*drawn, *ghosts)

    if cfg.edge_mode == "contributors":
        plain = [(u, v, 0.0) for u, v in contributor_edges(activations, drawn)]
        dropped_edges = 0
    else:
        plain, dropped_edges = induced_edges(graph, drawn, limit=cfg.max_edges)
    # The layout reads TOPOLOGY, not edge magnitude: every drawn link pulls
    # with the same unit strength. Merged weights span orders of magnitude
    # (entity edges are 1/df sums), and letting them drive the springs would
    # collapse the strong end of the graph into a dot.
    layout_edges: list[tuple[str, str, float]] = [
        (source, target, 1.0) for source, target, _ in plain
    ]
    # A cut pulls its duplicate NEAR the survivor but deliberately weaker than
    # a real link: at full strength the ghost lands on top of the survivor and
    # the dashed line - a required element of this view - becomes invisible.
    layout_edges += [
        (duplicate, survivor, 0.35) for duplicate, survivor in cuts.items()
    ]
    positions = spring_layout(placed_ids, layout_edges, layout, salt=salt)

    ranked = sorted(drawn, key=lambda node: (-energies.get(node, 0.0), node))
    labelled = set(ranked[: cfg.label_top_n])
    disputed_set = set(disputed)
    bridge_set = set(bridges)

    nodes: list[SceneNode] = []
    for node in placed_ids:
        x, y = positions.get(node, (0.5, 0.5))
        activation = activations.get(node)
        data = graph.node(node)
        text = (texts or {}).get(node, "")
        ghost = node in ghosts
        kind = _node_kind(node, seeds, bridge_set, ghost)
        vote = _vote_of(node, votes, graph)
        energy = 0.0 if activation is None else activation.energy
        hop = -1 if activation is None else activation.hop
        title = _snippet(text, snippet_chars) if text else node
        detail = f"{node} | energy {energy:.3f} | hop {hop} | votes {vote}"
        if ghost:
            detail = (
                f"{node} | suppressed duplicate of {cuts[node]} | votes -> survivor"
            )
        nodes.append(
            SceneNode(
                id=node,
                x=x,
                y=y,
                energy=energy,
                hop=hop,
                votes=vote,
                kind=kind,
                node_layer="chunk" if data is None else data.layer,
                source_id=node if data is None else data.source_id,
                polarity=1 if data is None else data.polarity,
                disputed=node in disputed_set,
                label=node if node in labelled else "",
                tooltip=f"{detail}\n{title}",
            )
        )

    edges: list[SceneEdge] = []
    for source, target, weight in plain:
        merged = float(graph.neighbors(source).get(target, weight))
        primary, held = layer_index.layer_of(source, target, weights)
        edges.append(
            _scene_edge(
                source,
                target,
                positions,
                weight=merged,
                layer=primary or "semantic",
                layers=held,
                kind="active",
                tooltip=_edge_tooltip(source, target, merged, held),
            )
        )
    for duplicate, survivor in sorted(cuts.items()):
        primary, held = layer_index.layer_of(duplicate, survivor, weights)
        edges.append(
            _scene_edge(
                duplicate,
                survivor,
                positions,
                weight=0.0,
                layer=primary or "semantic",
                layers=held,
                kind="suppressed",
                tooltip=(
                    f"{duplicate} duplicates {survivor}: edge cut, "
                    "its share redistributed and the survivor voted"
                ),
            )
        )

    caption = (
        f"drawing {len(drawn)} of {len(drawn) + dropped_nodes} activated atoms"
        f" and {len(plain)} of {len(plain) + dropped_edges} edges"
        f" ({cfg.edge_mode}); {len(cuts)} dedup cut(s) always shown"
    )
    legend = {**LAYER_COLORS}
    return GraphScene(
        nodes=tuple(nodes),
        edges=tuple(edges),
        legend=legend,
        dropped_nodes=dropped_nodes,
        dropped_edges=dropped_edges,
        caption=caption,
    )


def _edge_tooltip(
    source: str, target: str, weight: float, layers: tuple[str, ...]
) -> str:
    held = ", ".join(layers) if layers else "no enabled layer"
    return f"{source} — {target} | merged weight {weight:.4f} | {held}"


def _scene_edge(
    source: str,
    target: str,
    positions: Mapping[str, tuple[float, float]],
    *,
    weight: float,
    layer: str,
    layers: tuple[str, ...],
    kind: str,
    tooltip: str,
) -> SceneEdge:
    x1, y1 = positions.get(source, (0.5, 0.5))
    x2, y2 = positions.get(target, (0.5, 0.5))
    return SceneEdge(
        source=source,
        target=target,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        weight=weight,
        layer=layer,
        layers=layers,
        kind=kind,
        tooltip=tooltip,
    )


@dataclass(frozen=True)
class ComparisonRow:
    """One line of either column of the side-by-side panel."""

    rank: int
    node_id: str
    title: str
    snippet: str
    score: float
    hop: int | None
    votes: int
    in_other: bool
    badges: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    """The activated web against plain `top-k`, cut at the same k.

    This is the most important thing the inspector shows: the benchmark
    number is hard to interpret without seeing, on one question, which
    passages the web found that the baseline never returned.

    `contact_tau` travels in this view model rather than being read straight
    off the result in the page, so the design's "the computed duplicate cut
    must be visible" rule is a data contract a test can pin.
    """

    web: tuple[ComparisonRow, ...]
    baseline: tuple[ComparisonRow, ...]
    only_in_web: tuple[str, ...]
    only_in_baseline: tuple[str, ...]
    overlap: tuple[str, ...]
    contact_tau: float | None
    dedup_enabled: bool


def build_comparison(
    *,
    web_ranked: Sequence[tuple[str, float]],
    baseline_ids: Sequence[str],
    k: int,
    activations: Mapping[str, Activation] | None = None,
    votes: Mapping[str, int] | None = None,
    graph: Graph | None = None,
    seeds: Collection[str] = (),
    bridges: Collection[str] = (),
    disputed: Collection[str] = (),
    contact_tau: float | None = None,
    dedup_enabled: bool = False,
    titles: Mapping[str, str] | None = None,
    texts: Mapping[str, str] | None = None,
    baseline_scores: Mapping[str, float] | None = None,
    snippet_chars: int = 220,
) -> Comparison:
    """The side-by-side view model: activated web vs plain top-k at one k.

    Both columns are cut at the SAME k - the harness's context-budget parity
    rule, applied to the picture. Titles and texts are optional: an index
    whose dataset could not be resolved still renders, with ids standing in,
    because a developer tool must open what it is given.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    web_ids = [node for node, _ in web_ranked[:k]]
    base_ids = list(baseline_ids[:k])
    web_set, base_set = set(web_ids), set(base_ids)

    def row(
        rank: int, node: str, score: float, other: Collection[str], *, web_side: bool
    ) -> ComparisonRow:
        activation = (activations or {}).get(node)
        badges: list[str] = []
        if node in seeds:
            badges.append("seed")
        if node in bridges:
            badges.append("bridge")
        if node in disputed:
            badges.append("disputed")
        vote = _vote_of(node, votes or {}, graph) if graph is not None else 1
        if vote > 1:
            badges.append(f"voted x{vote}")
        text = (texts or {}).get(node, "")
        return ComparisonRow(
            rank=rank,
            node_id=node,
            title=(titles or {}).get(node, node),
            snippet=_snippet(text, snippet_chars) if text else "",
            score=score,
            hop=activation.hop if (web_side and activation is not None) else None,
            votes=vote,
            in_other=node in other,
            badges=tuple(badges),
        )

    web_rows = tuple(
        row(rank, node, score, base_set, web_side=True)
        for rank, (node, score) in enumerate(web_ranked[:k], start=1)
    )
    base_rows = tuple(
        row(rank, node, (baseline_scores or {}).get(node, 0.0), web_set, web_side=False)
        for rank, node in enumerate(base_ids, start=1)
    )
    return Comparison(
        web=web_rows,
        baseline=base_rows,
        only_in_web=tuple(node for node in web_ids if node not in base_set),
        only_in_baseline=tuple(node for node in base_ids if node not in web_set),
        overlap=tuple(node for node in web_ids if node in base_set),
        contact_tau=contact_tau,
        dedup_enabled=dedup_enabled,
    )


@dataclass(frozen=True)
class VectorMatrix:
    """Node embeddings as one L2-normalised matrix, addressable by id.

    `VectorStore` keeps its vectors private (FAISS owns them), so the
    inspector reads `vectors.npz` itself - the same artifact the index stage
    wrote, which keeps one source of truth.
    """

    ids: tuple[str, ...]
    position: Mapping[str, int]
    matrix: NDArray[np.float32]


def vector_matrix(
    ids: Sequence[str], vectors: NDArray[np.float32] | Sequence[Sequence[float]]
) -> VectorMatrix:
    """Wrap ids and rows, re-normalising defensively.

    The embedder already normalises, but the store is metric-agnostic, so a
    hand-built or migrated artifact could arrive unnormalised - and a cosine
    that is quietly not a cosine would corrupt every duplicate decision.
    """
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError(
            f"vectors have shape {matrix.shape}; expected ({len(ids)}, dimension)"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return VectorMatrix(
        ids=tuple(ids),
        position={node_id: index for index, node_id in enumerate(ids)},
        matrix=matrix,
    )


def make_similarity(vectors: VectorMatrix) -> SimilarityFn:
    """Node-pair cosine over the stored embeddings - what turns dedup ON.

    `retrieve()` disables redundancy suppression unless it gets BOTH an
    enabled `DedupConfig` and one of these; passing only the config leaves the
    mechanism silently off and `contact_tau` at `None`. The page says so out
    loud rather than letting the operator read an empty metric as "no
    duplicates found".

    An unknown id scores `0.0` instead of raising: an inspector must not die
    because an artifact and a graph disagree about one node.
    """

    def similarity(node: str, others: Sequence[str]) -> Sequence[float]:
        row = vectors.position.get(node)
        if row is None or not others:
            return [0.0] * len(others)
        columns = [vectors.position.get(other, -1) for other in others]
        known = np.array([index for index in columns if index >= 0], dtype=np.int64)
        scores = np.zeros(len(others), dtype=np.float64)
        if known.size:
            values = vectors.matrix[known] @ vectors.matrix[row]
            slot = 0
            for position, index in enumerate(columns):
                if index >= 0:
                    scores[position] = float(values[slot])
                    slot += 1
        return scores.tolist()

    return similarity


def _scene_records(scene: GraphScene) -> tuple[list[dict], list[dict], list[dict]]:
    """Scene -> three plain record lists: active edges, cut edges, nodes."""
    active = [
        {
            "x1": edge.x1,
            "y1": edge.y1,
            "x2": edge.x2,
            "y2": edge.y2,
            "layer": edge.layer,
            "weight": edge.weight,
            "tooltip": edge.tooltip,
        }
        for edge in scene.edges
        if edge.kind == "active"
    ]
    cut = [
        {
            "x1": edge.x1,
            "y1": edge.y1,
            "x2": edge.x2,
            "y2": edge.y2,
            "layer": edge.layer,
            "tooltip": edge.tooltip,
        }
        for edge in scene.edges
        if edge.kind == "suppressed"
    ]
    nodes = [
        {
            "x": node.x,
            "y": node.y,
            "id": node.id,
            "energy": node.energy,
            "hop": node.hop,
            "votes": node.votes,
            "kind": node.kind,
            "label": node.label,
            "tooltip": node.tooltip,
        }
        for node in scene.nodes
    ]
    return active, cut, nodes


def build_vega_spec(scene: GraphScene, *, height: int = 620) -> dict[str, object]:
    """Self-contained Vega-Lite v5 spec with the scene inlined - pure data.

    Two separate `rule` layers carry the two edge kinds rather than one
    `strokeDash` encoding channel: a dash scale wants an array-of-arrays
    range, which is version-sensitive, while two layers are unambiguous and
    a test can read them straight out of the dict.

    The colour scale's domain is the fixed layer order, so an edge keeps its
    colour across queries even when a layer happens to be absent from one.
    """
    active, cut, nodes = _scene_records(scene)
    domain = list(EDGE_LAYER_ORDER)
    colors = [LAYER_COLORS[layer] for layer in domain]
    kinds = list(NODE_KIND_ORDER)
    kind_colors = [NODE_KIND_COLORS[kind] for kind in kinds]
    axis = {"scale": {"domain": [0, 1]}, "axis": None, "type": "quantitative"}
    edge_encoding = {
        "x": {"field": "x1", **axis},
        "y": {"field": "y1", **axis},
        "x2": {"field": "x2"},
        "y2": {"field": "y2"},
        "color": {
            "field": "layer",
            "type": "nominal",
            "scale": {"domain": domain, "range": colors},
            "legend": {"title": "edge layer"},
        },
        "tooltip": {"field": "tooltip", "type": "nominal"},
    }
    # Independent colour scales give every layer its own legend, so the dashed
    # rule would print a second, identical "edge layer" key. It keeps the
    # colours and drops the duplicate legend; the caption explains the dashes.
    cut_encoding = {
        **edge_encoding,
        "color": {**edge_encoding["color"], "legend": None},
    }
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": height,
        "background": None,
        # Edges are coloured by layer and atoms by role; without an
        # independent colour scale Vega merges the two domains into one and
        # hands the atoms the edge palette, so a seed comes out wearing the
        # entity colour and the legend reads as one nonsensical list.
        "resolve": {"scale": {"color": "independent"}},
        "layer": [
            {
                "data": {"values": active},
                "mark": {"type": "rule", "opacity": 0.55},
                "encoding": edge_encoding,
            },
            {
                "data": {"values": cut},
                "mark": {
                    "type": "rule",
                    "strokeDash": [4, 3],
                    "opacity": 0.9,
                    "strokeWidth": 1.5,
                },
                "encoding": cut_encoding,
            },
            {
                "data": {"values": nodes},
                # The pan/zoom param lives on exactly ONE layer. Declared at
                # the top of a layered spec it is pushed into every layer, and
                # the two edge layers share the field name `x1`, so Vega ends
                # up asked to register the signal `zoom_x1` twice and the whole
                # chart fails to render ("Duplicate signal name").
                "params": [{"name": "zoom", "select": "interval", "bind": "scales"}],
                "mark": {"type": "circle", "stroke": "white", "strokeWidth": 0.5},
                "encoding": {
                    "x": {"field": "x", **axis},
                    "y": {"field": "y", **axis},
                    "size": {
                        "field": "energy",
                        "type": "quantitative",
                        "scale": {"range": [40, 600]},
                        "legend": None,
                    },
                    "color": {
                        "field": "kind",
                        "type": "nominal",
                        "scale": {"domain": kinds, "range": kind_colors},
                        "legend": {"title": "atom"},
                    },
                    "tooltip": {"field": "tooltip", "type": "nominal"},
                },
            },
            {
                "data": {"values": [node for node in nodes if node["label"]]},
                "mark": {"type": "text", "dy": -12, "fontSize": 10},
                "encoding": {
                    "x": {"field": "x", **axis},
                    "y": {"field": "y", **axis},
                    "text": {"field": "label", "type": "nominal"},
                },
            },
        ],
    }


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
    starting angle comes from the same `layout_seed` the spring layout uses,
    so this shares its determinism contract: same query, same picture.
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

    rng = np.random.default_rng(layout_seed(ids, salt, seed))
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
        offset = float(rng.random()) * 2.0 * np.pi
        for position, node_id in enumerate(members):
            angle = offset + 2.0 * np.pi * position / len(members)
            placed[node_id] = (
                0.5 + radius * float(np.cos(angle)),
                0.5 + radius * float(np.sin(angle)),
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

    Exposed so the browser can draw its guide circles from the SAME rule that
    placed the atoms. The canvas used to carry its own copy of the spacing
    formula; the moment this one changed, the guides described a layout that
    no longer existed and the rings sat where nothing was. One mechanism, one
    implementation - the same discipline the scene builder itself exists for.
    """
    present = sorted({int(hop) for hop in hops})
    levels = sorted(hop for hop in present if hop >= 0)
    order = {hop: index for index, hop in enumerate(levels)}
    return {hop: _ring_radius(hop, order, len(levels)) for hop in present}
