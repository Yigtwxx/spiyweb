"""Query traces: what an application's retrieval actually did (D38).

The browser face has two possible products and confusing them is the real
risk. One is **live inspection** - a server that owns an index and runs a NEW
query when you press a button. The other is a **trace viewer** - it shows the
calls that genuinely happened inside someone's application. The second is the
primary one: it needs no second copy of the graph and the vector store in
memory, it survives in production, and it is the only one that answers "why
did MY app retrieve that".

That decision is what shapes this module. A record has to stand on its own -
the activated subgraph's edges, the passages' text, the seeds, the votes, the
paths, the destroyed-energy events - because the thing reading it will not
have the index. A record that needed the index reloaded to be drawn would be
the live-inspection product wearing a trace's name.

Two consequences, both deliberate:

- **Memory by default, disk on request.** The store is a ring buffer of the
  last `capacity` calls. Passage text lands in the JSONL file, so writing one
  is an explicit `TraceConfig(directory=...)` choice, never a default. The
  one other request the store honours is a running `spiyweb` monitor: its
  fresh marker under `TraceConfig.attach_dir` makes the store append there
  too, into a folder that ignores itself from git (`attach_dir=None` is off).
- **No I/O and no heavy dependency to build a record.** Everything here is
  stdlib over `core` results, so `import spiyweb.trace` costs nothing and
  tracing never drags numpy into a caller's process that had avoided it.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spiyweb.config import TraceConfig
from spiyweb.core.propagate import Activation, PropagationResult
from spiyweb.ledger import build_ledger
from spiyweb.output import activation_paths, theme_clusters

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from spiyweb.config import PropagationConfig
    from spiyweb.core.colors import ColoredResult
    from spiyweb.core.graph import Graph
    from spiyweb.retrieve import ColoredRetrievalResult, RetrievalResult

__all__ = [
    "SCHEMA_VERSION",
    "TRACE_FILENAME",
    "WATCH_MARKER",
    "TraceCluster",
    "TraceEdge",
    "TraceEvent",
    "TraceLedger",
    "TraceNode",
    "TracePath",
    "TraceRecord",
    "TraceStore",
    "attached_trace_path",
    "build_trace",
    "ensure_private_dir",
    "load_traces",
]

SCHEMA_VERSION = 1
"""Bumped when a reader written against the old shape would misread the new
one. The viewer refuses a version it does not know rather than guessing."""

TRACE_FILENAME = "traces.jsonl"
"""One append-only file per trace directory; a line is one complete record."""

WATCH_MARKER = "watch"
"""The file a running `spiyweb` monitor keeps touching inside the attach
directory. Its mtime IS the heartbeat: fresh means somebody is watching."""

GITIGNORE_ALL = "*\n"
"""What the attach directory's own `.gitignore` says: passage text lands here,
and a folder that ignores itself cannot be committed by accident."""


def attached_trace_path(
    directory: str | Path, *, stale_s: float, now: float | None = None
) -> Path | None:
    """Where a listening monitor wants records appended, or `None`.

    One `stat`, no cache: a marker removed between two queries is seen by the
    second one, and a monitor that crashed stops being obeyed `stale_s`
    seconds later without anybody cleaning up.
    """
    if os.environ.get("SPIYWEB_NO_ATTACH"):
        # The test suite sets this: a developer's live monitor must not
        # collect every query the tests run in the same folder.
        return None
    marker = Path(directory) / WATCH_MARKER
    try:
        age = (time.time() if now is None else now) - marker.stat().st_mtime
    except OSError:
        return None
    if age > stale_s:
        return None
    return Path(directory) / TRACE_FILENAME


def ensure_private_dir(directory: Path) -> None:
    """Create `directory` with a `.gitignore` that excludes everything in it."""
    directory.mkdir(parents=True, exist_ok=True)
    ignore = directory / ".gitignore"
    if not ignore.exists():
        ignore.write_text(GITIGNORE_ALL, encoding="utf-8")


def _append_line(path: Path, record: TraceRecord) -> None:
    """One binary write in append mode - the closest thing to atomic that
    two application processes sharing a file get on every platform."""
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with path.open("ab") as handle:
        handle.write(line.encode("utf-8"))


@dataclass(frozen=True)
class TraceNode:
    """One atom as this run saw it - enough to draw it without the index.

    Attributes:
        id: Graph node id.
        source_id: Vote granularity - the document, never the chunk.
        layer: `"chunk"` or `"proposition"`.
        energy: Accumulated activation. `0.0` for an atom that only appears
            because duplicate suppression cut it.
        hop: Distance from the seed; `-1` for a suppressed atom that never
            activated.
        votes: Corpus support for the idea, merged across both suppression
            stages.
        text: What was indexed, possibly truncated or omitted by config.
        seed_similarity: Cosine at first contact, or `None` for an atom the
            energy reached by hopping.
        polarity: `+1` for an ordinary atom, `-1` for a negative-knowledge
            one - visible in the picture, so it has to survive the record.
        disputed: Survived a contradiction with energy left (D16).
        suppressed_by: The atom this one duplicated, or `""`. A non-empty
            value IS the vote mechanism firing, so it stays in the picture.
    """

    id: str
    source_id: str
    layer: str
    energy: float
    hop: int
    votes: int
    text: str = ""
    seed_similarity: float | None = None
    polarity: int = 1
    disputed: bool = False
    suppressed_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "layer": self.layer,
            "energy": self.energy,
            "hop": self.hop,
            "votes": self.votes,
            "text": self.text,
            "seed_similarity": self.seed_similarity,
            "polarity": self.polarity,
            "disputed": self.disputed,
            "suppressed_by": self.suppressed_by,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceNode:
        similarity = payload.get("seed_similarity")
        return cls(
            id=str(payload["id"]),
            source_id=str(payload["source_id"]),
            layer=str(payload["layer"]),
            energy=float(payload["energy"]),
            hop=int(payload["hop"]),
            votes=int(payload["votes"]),
            text=str(payload.get("text", "")),
            seed_similarity=None if similarity is None else float(similarity),
            polarity=int(payload.get("polarity", 1)),
            disputed=bool(payload.get("disputed", False)),
            suppressed_by=str(payload.get("suppressed_by", "")),
        )


@dataclass(frozen=True)
class TraceEdge:
    """One edge of the activated subgraph, with its merged weight.

    The merged adjacency deliberately forgets which layer a weight came from
    (`Graph.from_layers`), so neither does this - a trace never claims to
    know more about an edge than the graph it was read from.
    """

    source: str
    target: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "weight": self.weight}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceEdge:
        return cls(
            source=str(payload["source"]),
            target=str(payload["target"]),
            weight=float(payload["weight"]),
        )


@dataclass(frozen=True)
class TracePath:
    """How the energy reached one atom (D19) - explanation, not debug data."""

    node: str
    steps: tuple[str, ...]
    hop: int
    energy: float
    converging: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "steps": list(self.steps),
            "hop": self.hop,
            "energy": self.energy,
            "converging": self.converging,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TracePath:
        return cls(
            node=str(payload["node"]),
            steps=tuple(str(step) for step in payload["steps"]),
            hop=int(payload["hop"]),
            energy=float(payload["energy"]),
            converging=int(payload["converging"]),
        )


@dataclass(frozen=True)
class TraceCluster:
    """One theme cluster of the run (D20), with its share of the energy."""

    nodes: tuple[str, ...]
    energy: float
    energy_share: float
    top_node: str
    colors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "energy": self.energy,
            "energy_share": self.energy_share,
            "top_node": self.top_node,
            "colors": list(self.colors),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceCluster:
        return cls(
            nodes=tuple(str(node) for node in payload["nodes"]),
            energy=float(payload["energy"]),
            energy_share=float(payload["energy_share"]),
            top_node=str(payload["top_node"]),
            colors=tuple(str(color) for color in payload.get("colors", ())),
        )


@dataclass(frozen=True)
class TraceEvent:
    """One thing that happened to the energy ledger, kept flat on purpose.

    `kind` is `"conflict"`, `"negative_seed"`, `"polarity"` or `"suppressed"`.
    The first three DESTROY energy and `amount` says how much; the last
    REDISTRIBUTES it and carries `amount` `0.0` with the survivor in `other`.
    Keeping the four in one ordered list is what lets a reader replay the
    ledger without knowing which mechanisms were even enabled.
    """

    kind: str
    node: str
    other: str = ""
    amount: float = 0.0
    hop: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "node": self.node,
            "other": self.other,
            "amount": self.amount,
            "hop": self.hop,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceEvent:
        return cls(
            kind=str(payload["kind"]),
            node=str(payload["node"]),
            other=str(payload.get("other", "")),
            amount=float(payload.get("amount", 0.0)),
            hop=int(payload.get("hop", -1)),
        )


@dataclass(frozen=True)
class TraceLedger:
    """Where this run's injected energy went, and how well it added up.

    A flattened `spiyweb.ledger.Ledger`. It is computed here, at record time,
    rather than by whoever reads the file: the reconstruction needs the graph
    and the propagation config, and a reader that had those would be holding
    the index - the thing a trace exists to avoid.

    `balanced` false is a FINDING, not a rounding artefact, and `notes` says
    what the reconstruction knows about why.
    """

    injected: float
    held: float
    dissipated: float
    destroyed_conflict: float
    destroyed_negative_seed: float
    destroyed_polarity: float
    residual: float
    mismatch: float
    tolerance: float
    balanced: bool
    exact: bool
    dedup_cuts: int
    notes: tuple[str, ...] = ()

    @property
    def destroyed(self) -> float:
        """Total energy destroyed - conflicts, negative seeds and polarity."""
        return (
            self.destroyed_conflict
            + self.destroyed_negative_seed
            + self.destroyed_polarity
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "injected": self.injected,
            "held": self.held,
            "dissipated": self.dissipated,
            "destroyed_conflict": self.destroyed_conflict,
            "destroyed_negative_seed": self.destroyed_negative_seed,
            "destroyed_polarity": self.destroyed_polarity,
            "residual": self.residual,
            "mismatch": self.mismatch,
            "tolerance": self.tolerance,
            "balanced": self.balanced,
            "exact": self.exact,
            "dedup_cuts": self.dedup_cuts,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceLedger:
        return cls(
            injected=float(payload["injected"]),
            held=float(payload["held"]),
            dissipated=float(payload["dissipated"]),
            destroyed_conflict=float(payload["destroyed_conflict"]),
            destroyed_negative_seed=float(payload["destroyed_negative_seed"]),
            destroyed_polarity=float(payload["destroyed_polarity"]),
            residual=float(payload["residual"]),
            mismatch=float(payload["mismatch"]),
            tolerance=float(payload["tolerance"]),
            balanced=bool(payload["balanced"]),
            exact=bool(payload["exact"]),
            dedup_cuts=int(payload["dedup_cuts"]),
            notes=tuple(str(note) for note in payload.get("notes", ())),
        )


@dataclass(frozen=True)
class TraceRecord:
    """One recorded retrieval, complete enough to be drawn on its own."""

    trace_id: str
    sequence: int
    recorded_at: str
    kind: str
    query: str
    nodes: tuple[TraceNode, ...]
    edges: tuple[TraceEdge, ...]
    paths: tuple[TracePath, ...]
    clusters: tuple[TraceCluster, ...]
    events: tuple[TraceEvent, ...]
    stop_reason: str
    hops_used: int
    injected_energy: float
    threshold: float
    total_energy: float
    node_count: int
    dedup_mode: str
    dedup_thresholds: tuple[float, ...] = ()
    bridges: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    parts: Mapping[str, str] = field(default_factory=dict)
    index: str = ""
    profile: str = ""
    elapsed_ms: float = 0.0
    settings: Mapping[str, Any] = field(default_factory=dict)
    ledger: TraceLedger | None = None
    """The energy ledger, or `None` when the propagation config was not
    available to reconstruct it."""
    edges_truncated: bool = False
    schema_version: int = SCHEMA_VERSION

    def node(self, node_id: str) -> TraceNode | None:
        """The recorded atom with this id, or `None`."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def ranked(self) -> list[tuple[str, float]]:
        """Activated atoms by energy, strongest first - the run's ranking."""
        return sorted(
            ((node.id, node.energy) for node in self.nodes if node.energy > 0.0),
            key=lambda item: (-item[1], item[0]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "sequence": self.sequence,
            "recorded_at": self.recorded_at,
            "kind": self.kind,
            "query": self.query,
            "parts": dict(self.parts),
            "index": self.index,
            "profile": self.profile,
            "elapsed_ms": self.elapsed_ms,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "paths": [path.to_dict() for path in self.paths],
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "events": [event.to_dict() for event in self.events],
            "bridges": {node: list(colors) for node, colors in self.bridges.items()},
            "stop_reason": self.stop_reason,
            "hops_used": self.hops_used,
            "injected_energy": self.injected_energy,
            "threshold": self.threshold,
            "total_energy": self.total_energy,
            "node_count": self.node_count,
            "dedup_mode": self.dedup_mode,
            "dedup_thresholds": list(self.dedup_thresholds),
            "settings": dict(self.settings),
            "ledger": None if self.ledger is None else self.ledger.to_dict(),
            "edges_truncated": self.edges_truncated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceRecord:
        """Rebuild a record from its JSON form, refusing a shape it cannot read."""
        version = int(payload.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"trace schema version {version} cannot be read by this build "
                f"(it understands version {SCHEMA_VERSION}); read it with the "
                "version of spiyweb that wrote it"
            )
        return cls(
            trace_id=str(payload["trace_id"]),
            sequence=int(payload["sequence"]),
            recorded_at=str(payload["recorded_at"]),
            kind=str(payload["kind"]),
            query=str(payload["query"]),
            nodes=tuple(TraceNode.from_dict(node) for node in payload["nodes"]),
            edges=tuple(TraceEdge.from_dict(edge) for edge in payload["edges"]),
            paths=tuple(TracePath.from_dict(path) for path in payload["paths"]),
            clusters=tuple(
                TraceCluster.from_dict(cluster) for cluster in payload["clusters"]
            ),
            events=tuple(TraceEvent.from_dict(event) for event in payload["events"]),
            stop_reason=str(payload["stop_reason"]),
            hops_used=int(payload["hops_used"]),
            injected_energy=float(payload["injected_energy"]),
            threshold=float(payload["threshold"]),
            total_energy=float(payload["total_energy"]),
            node_count=int(payload["node_count"]),
            dedup_mode=str(payload["dedup_mode"]),
            dedup_thresholds=tuple(
                float(tau) for tau in payload.get("dedup_thresholds", ())
            ),
            bridges={
                str(node): tuple(str(color) for color in colors)
                for node, colors in payload.get("bridges", {}).items()
            },
            parts={
                str(label): str(text)
                for label, text in payload.get("parts", {}).items()
            },
            index=str(payload.get("index", "")),
            profile=str(payload.get("profile", "")),
            elapsed_ms=float(payload.get("elapsed_ms", 0.0)),
            settings=dict(payload.get("settings", {})),
            ledger=(
                None
                if payload.get("ledger") is None
                else TraceLedger.from_dict(payload["ledger"])
            ),
            edges_truncated=bool(payload.get("edges_truncated", False)),
            schema_version=version,
        )


class TraceStore:
    """The last `capacity` records, in memory, optionally mirrored to JSONL.

    A ring buffer and not a list: an application that runs for a week must
    not accumulate a week of passage text in RAM. The JSONL file, when a
    directory is configured, is append-only and keeps everything - the
    forgetting is the buffer's policy, not the file's.
    """

    def __init__(self, config: TraceConfig | None = None) -> None:
        self._config = config if config is not None else TraceConfig()
        self._records: deque[TraceRecord] = deque(maxlen=self._config.capacity)
        self._sequence = 0
        self._path: Path | None = None
        if self._config.directory is not None:
            directory = Path(self._config.directory)
            directory.mkdir(parents=True, exist_ok=True)
            self._path = directory / TRACE_FILENAME

    @property
    def config(self) -> TraceConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        """Whether anything is recorded at all - the §6 ablation switch."""
        return self._config.enabled

    @property
    def path(self) -> Path | None:
        """The JSONL file, or `None` while the disk is untouched."""
        return self._path

    def next_sequence(self) -> int:
        """The number the next recorded call will carry, without consuming it."""
        return self._sequence

    def append(self, record: TraceRecord) -> TraceRecord:
        """Keep `record`; write it too when a directory was configured, or
        when a `spiyweb` monitor is listening in `attach_dir` right now."""
        self._records.append(record)
        self._sequence = record.sequence + 1
        if self._path is not None:
            _append_line(self._path, record)
        elif self._config.attach_dir is not None:
            target = attached_trace_path(
                self._config.attach_dir, stale_s=self._config.attach_stale_s
            )
            if target is not None:
                try:
                    ensure_private_dir(target.parent)
                    _append_line(target, record)
                except OSError:
                    pass  # a monitor's convenience must never reach the app
        return record

    def records(self) -> tuple[TraceRecord, ...]:
        """Everything still held, oldest first."""
        return tuple(self._records)

    def latest(self) -> TraceRecord | None:
        return self._records[-1] if self._records else None

    def get(self, trace_id: str) -> TraceRecord | None:
        for record in reversed(self._records):
            if record.trace_id == trace_id:
                return record
        return None

    def clear(self) -> None:
        """Drop the held records. The JSONL file, if any, is left alone."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterable[TraceRecord]:
        return iter(tuple(self._records))


def load_traces(path: Path | str) -> tuple[TraceRecord, ...]:
    """Read a JSONL trace file back, oldest first.

    This is the whole reader half of D38: a viewer opens the file, gets these
    records, and draws them. No index, no vector store, no model.

    The file is usually being appended to WHILE this reads it - that is the
    normal case, not the edge case, because the application that wrote it is
    still answering questions. So a truncated LAST line is tolerated: it is a
    record mid-write, and it will be complete a moment later.

    A broken line anywhere else is not tolerated. That is data loss, and a
    reader that skipped it would show a viewer with a hole in it and no way
    to know. The error names the line.
    """
    target = Path(path)
    if target.is_dir():
        target = target / TRACE_FILENAME
    lines = [
        (number, line.strip())
        for number, line in enumerate(
            target.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    records: list[TraceRecord] = []
    for position, (number, line) in enumerate(lines):
        try:
            records.append(TraceRecord.from_dict(json.loads(line)))
        except (ValueError, KeyError, TypeError) as broken:
            if position == len(lines) - 1:
                # The writer is mid-append. Stop here rather than raise.
                break
            raise ValueError(
                f"{target}: line {number} is not a readable trace record "
                f"({broken}); the file is damaged, not merely being written to"
            ) from broken
    return tuple(records)


def build_trace(
    result: RetrievalResult | ColoredRetrievalResult,
    graph: Graph,
    *,
    query: str,
    sequence: int,
    config: TraceConfig | None = None,
    texts: Mapping[str, str] | None = None,
    parts: Mapping[str, str] | None = None,
    index: str = "",
    profile: str = "",
    elapsed_ms: float = 0.0,
    settings: Mapping[str, Any] | None = None,
    propagation_config: PropagationConfig | None = None,
) -> TraceRecord:
    """Turn one finished retrieval into a self-contained record.

    Pure: it reads a result and a graph and returns data. A coloured result
    is merged the same way `ColoredResult.ranked()` merges it - energies sum,
    the hop is the earliest colour's - so one reader draws both kinds.
    """
    cfg = config if config is not None else TraceConfig()
    colored = getattr(result, "colored", None)
    propagation = (
        _merge_colors(colored) if colored is not None else result.propagation  # type: ignore[union-attr]
    )
    seeds = _seed_similarities(result)
    suppressed = _suppressions(result, propagation)
    votes = result.votes()
    disputed = result.disputed

    node_ids = sorted(set(propagation.activations) | set(suppressed))
    nodes = tuple(
        _trace_node(
            node_id,
            graph=graph,
            activation=propagation.activations.get(node_id),
            votes=votes,
            texts=texts or {},
            seeds=seeds,
            disputed=node_id in disputed,
            suppressed_by=suppressed.get(node_id, ""),
            cfg=cfg,
        )
        for node_id in node_ids
    )
    edges, truncated = _subgraph_edges(graph, node_ids, cfg)
    total_energy = sum(
        activation.energy for activation in propagation.activations.values()
    )
    return TraceRecord(
        trace_id=uuid.uuid4().hex,
        sequence=sequence,
        recorded_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
        kind="colored" if colored is not None else "plain",
        query=query,
        nodes=nodes,
        edges=edges,
        paths=tuple(
            TracePath(
                node=path.node,
                steps=path.steps,
                hop=path.hop,
                energy=float(path.energy),
                converging=path.converging,
            )
            for path in activation_paths(propagation)
        ),
        clusters=tuple(
            TraceCluster(
                nodes=cluster.nodes,
                energy=float(cluster.energy),
                energy_share=float(cluster.energy_share),
                top_node=cluster.top_node,
                colors=cluster.colors,
            )
            for cluster in theme_clusters(
                propagation, graph, colors_of=_colors_of(colored)
            )
        ),
        events=_events(propagation, suppressed),
        stop_reason=propagation.stop_reason,
        hops_used=propagation.hops_used,
        injected_energy=float(propagation.injected_energy),
        threshold=float(propagation.threshold),
        total_energy=float(total_energy),
        node_count=len(propagation.activations),
        dedup_mode=result.dedup_mode,
        dedup_thresholds=tuple(float(tau) for tau in propagation.dedup_thresholds),
        bridges=(
            {node: tuple(colors) for node, colors in colored.bridges.items()}
            if colored is not None
            else {}
        ),
        parts=dict(parts or {}),
        index=index,
        profile=profile,
        elapsed_ms=elapsed_ms,
        settings=dict(settings or {}),
        ledger=_trace_ledger(colored, propagation, graph, propagation_config),
        edges_truncated=truncated,
    )


# --- internals -------------------------------------------------------------


def _trace_ledger(
    colored: ColoredResult | None,
    propagation: PropagationResult,
    graph: Graph,
    config: PropagationConfig | None,
) -> TraceLedger | None:
    """Reconstruct the energy ledger, per colour and then summed.

    A coloured run is several independent propagations, so its ledger is the
    sum of theirs - reconstructing one book from the MERGED view would replay
    a distribution that never happened, and the merged energies would not add
    up. `balanced` therefore means every colour balanced, not that the totals
    happened to cancel.
    """
    if config is None:
        return None
    books = (
        [
            build_ledger(result, graph, config)
            for _, result in sorted(colored.per_color.items())
        ]
        if colored is not None
        else [build_ledger(propagation, graph, config)]
    )
    notes: list[str] = []
    for book in books:
        for note in book.notes:
            if note not in notes:
                notes.append(note)
    return TraceLedger(
        injected=sum(book.injected for book in books),
        held=sum(book.held for book in books),
        dissipated=sum(book.dissipated for book in books),
        destroyed_conflict=sum(book.destroyed.conflict for book in books),
        destroyed_negative_seed=sum(book.destroyed.negative_seed for book in books),
        destroyed_polarity=sum(book.destroyed.polarity for book in books),
        residual=sum(book.residual for book in books),
        mismatch=sum(book.mismatch for book in books),
        tolerance=max(book.tolerance for book in books),
        balanced=all(book.balanced for book in books),
        exact=all(book.exact for book in books),
        dedup_cuts=sum(book.dedup_cuts for book in books),
        notes=tuple(notes),
    )


def _trace_node(
    node_id: str,
    *,
    graph: Graph,
    activation: Activation | None,
    votes: Mapping[str, int],
    texts: Mapping[str, str],
    seeds: Mapping[str, float],
    disputed: bool,
    suppressed_by: str,
    cfg: TraceConfig,
) -> TraceNode:
    data = graph.node(node_id)
    source_id = data.source_id if data is not None else node_id
    text = ""
    if cfg.include_texts:
        text = texts.get(node_id, "")
        if cfg.text_chars:
            text = text[: cfg.text_chars]
    return TraceNode(
        id=node_id,
        source_id=source_id,
        layer=data.layer if data is not None else "chunk",
        energy=float(activation.energy) if activation is not None else 0.0,
        hop=activation.hop if activation is not None else -1,
        # Votes are keyed by the SOURCE wherever the graph knows one (D7):
        # repetition inside a single document is not corroboration.
        votes=votes.get(source_id, votes.get(node_id, 1)),
        text=text,
        seed_similarity=seeds.get(node_id),
        polarity=data.polarity if data is not None else 1,
        disputed=disputed,
        suppressed_by=suppressed_by,
    )


def _subgraph_edges(
    graph: Graph, node_ids: Sequence[str], cfg: TraceConfig
) -> tuple[tuple[TraceEdge, ...], bool]:
    """Positive edges between recorded atoms, strongest first.

    Undirected: the adjacency carries both directions, and drawing the same
    connection twice would double every line in the picture.
    """
    if not cfg.include_edges:
        return (), False
    members = set(node_ids)
    seen: set[tuple[str, str]] = set()
    edges: list[TraceEdge] = []
    for source in node_ids:
        for target, weight in graph.neighbors(source).items():
            if weight <= 0.0 or target == source or target not in members:
                continue
            pair = (source, target) if source < target else (target, source)
            if pair in seen:
                continue
            seen.add(pair)
            edges.append(
                TraceEdge(source=pair[0], target=pair[1], weight=float(weight))
            )
    edges.sort(key=lambda edge: (-edge.weight, edge.source, edge.target))
    if cfg.max_edges and len(edges) > cfg.max_edges:
        return tuple(edges[: cfg.max_edges]), True
    return tuple(edges), False


def _seed_similarities(
    result: RetrievalResult | ColoredRetrievalResult,
) -> dict[str, float]:
    """First-contact cosines, merged across colours (strongest wins)."""
    per_color = getattr(result, "seeds_by_color", None)
    if per_color is None:
        return {node: float(value) for node, value in result.seeds.items()}  # type: ignore[union-attr]
    merged: dict[str, float] = {}
    for seeds in per_color.values():
        for node, value in seeds.items():
            merged[node] = max(merged.get(node, 0.0), float(value))
    return merged


def _suppressions(
    result: RetrievalResult | ColoredRetrievalResult,
    propagation: PropagationResult,
) -> dict[str, str]:
    """Duplicate -> survivor, across both suppression stages and all colours."""
    cuts: dict[str, str] = {}
    contact = result.contact_suppressed
    for key, value in contact.items():
        if isinstance(value, str):
            cuts[key] = value
        else:  # coloured runs nest one ledger per colour
            cuts.update(value)
    cuts.update(propagation.suppressed)
    return cuts


def _events(
    propagation: PropagationResult, suppressed: Mapping[str, str]
) -> tuple[TraceEvent, ...]:
    events: list[TraceEvent] = []
    for record in propagation.conflicts:
        events.append(
            TraceEvent(
                kind="conflict",
                node=record.node_a,
                other=record.node_b,
                amount=float(record.absorbed_each),
                hop=record.hop,
            )
        )
        events.append(
            TraceEvent(
                kind="conflict",
                node=record.node_b,
                other=record.node_a,
                amount=float(record.absorbed_each),
                hop=record.hop,
            )
        )
    events.extend(
        TraceEvent(
            kind="negative_seed",
            node=absorption.node,
            amount=float(absorption.absorbed),
            hop=absorption.hop,
        )
        for absorption in propagation.absorptions
    )
    events.extend(
        TraceEvent(
            kind="polarity",
            node=dispute.node,
            amount=float(dispute.absorbed),
            hop=dispute.hop,
        )
        for dispute in propagation.disputes
    )
    events.extend(
        TraceEvent(kind="suppressed", node=duplicate, other=survivor)
        for duplicate, survivor in sorted(suppressed.items())
    )
    return tuple(events)


def _colors_of(colored: ColoredResult | None) -> dict[str, tuple[str, ...]] | None:
    if colored is None:
        return None
    reached: dict[str, set[str]] = {}
    for color, result in colored.per_color.items():
        for node in result.activations:
            reached.setdefault(node, set()).add(color)
    return {node: tuple(sorted(colors)) for node, colors in reached.items()}


def _merge_colors(colored: ColoredResult) -> PropagationResult:
    """One propagation view of a coloured run, merged as `ranked()` merges it.

    Energies sum (the additive-accumulation rule one level up), the hop is
    the earliest colour's arrival, and the contributors are the union - so
    the paths and the clusters of a coloured record are computed by exactly
    the same functions that serve a plain one, and a viewer needs one reader
    instead of two.
    """
    activations: dict[str, Activation] = {}
    for _, result in sorted(colored.per_color.items()):
        for node, activation in result.activations.items():
            existing = activations.get(node)
            if existing is None:
                activations[node] = activation
                continue
            merged_contributors = tuple(
                dict.fromkeys(existing.contributors + activation.contributors)
            )
            activations[node] = Activation(
                energy=existing.energy + activation.energy,
                hop=min(existing.hop, activation.hop),
                contributors=merged_contributors,
            )
    results = sorted(
        colored.per_color.items(), key=lambda item: (-item[1].hops_used, item[0])
    )
    deepest = results[0][1]
    votes: dict[str, int] = {}
    for _, result in sorted(colored.per_color.items()):
        for key, count in result.votes.items():
            votes[key] = votes.get(key, 1) + (count - 1)
    suppressed: dict[str, str] = {}
    taus: list[float] = []
    for _, result in sorted(colored.per_color.items()):
        suppressed.update(result.suppressed)
        taus.extend(result.dedup_thresholds)
    return PropagationResult(
        activations=activations,
        injected_energy=sum(
            result.injected_energy for result in colored.per_color.values()
        ),
        threshold=max(result.threshold for result in colored.per_color.values()),
        hops_used=deepest.hops_used,
        stop_reason=deepest.stop_reason,
        votes=votes,
        suppressed=suppressed,
        dedup_thresholds=tuple(taus),
        conflicts=tuple(
            record
            for _, result in sorted(colored.per_color.items())
            for record in result.conflicts
        ),
        absorptions=tuple(
            record
            for _, result in sorted(colored.per_color.items())
            for record in result.absorptions
        ),
        disputes=tuple(
            record
            for _, result in sorted(colored.per_color.items())
            for record in result.disputes
        ),
    )
