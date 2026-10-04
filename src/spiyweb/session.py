"""`SpiywebIndex` - an opened index you can ask questions, with text back.

`retrieve()` takes a query EMBEDDING, a seed source, a graph and up to ten
keyword arguments, and returns node ids with energies. That is the right
shape for the library's core - it makes the mechanism testable and lets any
vector store satisfy `SeedSource` structurally - and it is the wrong shape
for someone who has a directory of documents and a question. Standing one up
took about a hundred lines, and `result.ranked()` handed back
`[("d00042:0", 5.6), ...]` with the passage text living somewhere else
entirely.

This module is that hundred lines, written once. It loads the graph, the
vector store, the texts and the similarity backend together, keeps them
together, and hands back passages that carry their own text.

Two things it deliberately does NOT do:

- **No `k`.** The web stops when its energy falls below the threshold; that
  self-termination is the project's whole argument against `top-k`, and a
  `k=` parameter here would quietly reintroduce the thing being argued
  against. Slice `answer.passages` if you want fewer.
- **No policy on confidence.** `Answer.confidence` reports total energy, node
  count and hop depth (D17). What counts as "I don't know" stays the
  caller's decision, and `answer.refusal()` builds the LLM-free explanation
  when the caller decides it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spiyweb.config import (
    ColoredRetrievalConfig,
    ConflictConfig,
    DedupConfig,
    PolarityConfig,
    RetrievalConfig,
    TraceConfig,
)
from spiyweb.core.conflict import conflict_adjacency
from spiyweb.indexing import (
    IndexLayout,
    load_graph,
    load_nli_edges,
    load_texts,
    read_manifest,
)
from spiyweb.output import (
    activation_paths,
    build_refusal_report,
    gap_warnings,
    theme_clusters,
)
from spiyweb.profiles import DEFAULT_PROFILE, PROFILES, Profile
from spiyweb.retrieve import retrieve as _retrieve
from spiyweb.retrieve import retrieve_colored as _retrieve_colored
from spiyweb.trace import TraceStore, build_trace

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from spiyweb.config import LayerWeights, OutputConfig, PropagationConfig
    from spiyweb.core.colors import ColoredResult
    from spiyweb.core.conflict import ConflictRecord
    from spiyweb.core.dedup import SimilarityFn
    from spiyweb.core.graph import Graph
    from spiyweb.core.polarity import DisputeRecord
    from spiyweb.embedding import Embedder
    from spiyweb.indexing import IndexManifest
    from spiyweb.output import ActivationPath, GapWarning, RefusalReport, ThemeCluster
    from spiyweb.retrieve import (
        ColoredRetrievalResult,
        Confidence,
        RetrievalResult,
    )
    from spiyweb.store import VectorStore
    from spiyweb.trace import TraceRecord

__all__ = ["Answer", "ColoredAnswer", "Passage", "SpiywebIndex", "open_index"]


@dataclass(frozen=True)
class Passage:
    """One activated node, with the text it was indexed as.

    Attributes:
        node_id: Graph id (`{source_id}:{position}` for a chunk).
        source_id: Vote granularity - the document, not the chunk.
        layer: `"chunk"` or `"proposition"`.
        text: What the embedder and the extractor saw. Empty for an index
            built before the `texts.json` artifact existed.
        energy: Accumulated activation; the ranking key.
        votes: Corpus support for this idea, merged across both suppression
            stages. `1` means "said once", never "unsupported".
        hop: Distance from the seed. `0` is a first-contact atom.
    """

    node_id: str
    source_id: str
    layer: str
    text: str
    energy: float
    votes: int
    hop: int


@dataclass(frozen=True)
class Answer:
    """A query's activated web, with text attached and the raw result kept.

    The raw `RetrievalResult` is a field rather than a hidden detail: this
    class is a convenience over the library, never a wall in front of it.
    """

    query: str
    passages: tuple[Passage, ...]
    result: RetrievalResult
    graph: Graph = field(repr=False)
    trace: TraceRecord | None = field(default=None, repr=False)
    """This call's trace record, or `None` while tracing is off (D38)."""
    profile: str = ""
    """The profile this call ran - `DEFAULT_PROFILE` when the caller named
    none and supplied no config, `""` when a raw config ran as given."""
    config: RetrievalConfig | None = field(default=None, repr=False)
    """The resolved configuration the web actually ran with."""

    @property
    def confidence(self) -> Confidence:
        """Total energy, node count and hop depth. Policy is the caller's."""
        return self.result.confidence

    @property
    def conflicts(self) -> tuple[ConflictRecord, ...]:
        """Contradiction records; both sides survive, neither is picked."""
        return self.result.conflicts

    @property
    def disputed(self) -> frozenset[str]:
        return self.result.disputed

    @property
    def disputes(self) -> tuple[DisputeRecord, ...]:
        """Negative-polarity absorptions - "the corpus disputes this"."""
        return self.result.disputes

    @property
    def dedup_mode(self) -> str:
        """Which duplicate rules actually ran on this query."""
        return self.result.dedup_mode

    def votes(self) -> dict[str, int]:
        return self.result.votes()

    def paths(self) -> tuple[ActivationPath, ...]:
        """How the energy reached each node - explanation, not debug data."""
        return activation_paths(self.result.propagation)

    def clusters(self) -> tuple[ThemeCluster, ...]:
        """Separate themes in what this query lit up."""
        return theme_clusters(self.result.propagation, self.graph)

    def gaps(self, config: OutputConfig | None = None) -> tuple[GapWarning, ...]:
        """Dense clusters with no bridge between them - a corpus gap (D18)."""
        return gap_warnings(self.clusters(), config)

    def refusal(self, config: OutputConfig | None = None) -> RefusalReport:
        """The LLM-free "why is this weak" report (D35), built on demand."""
        return build_refusal_report(self.result.propagation, self.graph, config=config)


@dataclass(frozen=True)
class ColoredAnswer:
    """The coloured-path twin of `Answer`; bridges are the point."""

    query: str
    passages: tuple[Passage, ...]
    result: ColoredRetrievalResult
    graph: Graph = field(repr=False)
    trace: TraceRecord | None = field(default=None, repr=False)
    """This call's trace record, or `None` while tracing is off (D38)."""

    @property
    def colored(self) -> ColoredResult:
        return self.result.colored

    @property
    def bridges(self) -> Mapping[str, tuple[str, ...]]:
        """Nodes two or more colours reached - where a multi-hop answer lives."""
        return self.result.bridges

    @property
    def dedup_mode(self) -> str:
        return self.result.dedup_mode

    def votes(self) -> dict[str, int]:
        return self.result.votes()


class SpiywebIndex:
    """An index directory, opened once and queried many times.

    Loading is the expensive part - a graph merge, a FAISS rebuild and a text
    map - so it happens once here instead of per query, and anything that
    serves an index should hold one of these rather than a second copy of
    everything in memory.
    """

    def __init__(
        self,
        layout: IndexLayout,
        *,
        graph: Graph,
        store: VectorStore,
        texts: Mapping[str, str],
        similarity: SimilarityFn,
        manifest: IndexManifest,
        negative: Mapping[str, Mapping[str, float]] | None = None,
        embedder: Embedder | None = None,
        config: RetrievalConfig | None = None,
        dedup: DedupConfig | None = None,
        conflict: ConflictConfig | None = None,
        polarity: PolarityConfig | None = None,
        trace: TraceConfig | None = None,
    ) -> None:
        self._layout = layout
        self._graph = graph
        self._store = store
        self._texts = dict(texts)
        self._similarity = similarity
        self._manifest = manifest
        self._negative = negative
        self._embedder = embedder
        # `None` is kept, not replaced: it is how `_resolve` knows the caller
        # expressed no preference and the default profile may apply.
        self._config: RetrievalConfig | None = config
        self._dedup = dedup if dedup is not None else DedupConfig()
        self._conflict = conflict if conflict is not None else ConflictConfig()
        self._polarity = polarity if polarity is not None else PolarityConfig()
        self._traces = TraceStore(trace)
        self._checked_model = False
        # Votes are per DOCUMENT, never per chunk (D7): repetition inside
        # one source is not corroboration. `retrieve` derives the
        # suppression key from the graph on its own, but the vote LEDGER
        # keys on this map, so an index that does not pass it reports
        # chunk-level votes and quietly answers a different question.
        self._source_of = {
            node_id: data.source_id for node_id, data in graph.node_data.items()
        }

    @classmethod
    def open(
        cls,
        path: IndexLayout | Path | str,
        *,
        weights: LayerWeights | None = None,
        embedder: Embedder | None = None,
        config: RetrievalConfig | None = None,
        dedup: DedupConfig | None = None,
        conflict: ConflictConfig | None = None,
        polarity: PolarityConfig | None = None,
        trace: TraceConfig | None = None,
    ) -> SpiywebIndex:
        """Load every artifact of an index directory into one object.

        Duplicate suppression is ON by default here, and that is the whole
        reason this class exists rather than a documentation page: the
        mechanism needs a `DedupConfig` AND a similarity backend, callers
        forget the second one, and this class wires both. Contradiction edges
        join automatically when the index has them - building
        `edges_nli.json` was already the opt-in.

        The embedder is constructed lazily on the first query, so opening an
        index costs no torch import.
        """
        from spiyweb.indexing import load_similarity, load_store

        layout = IndexLayout.at(path)
        negative_edges = load_nli_edges(layout)
        return cls(
            layout,
            graph=load_graph(layout, weights),
            store=load_store(layout),
            texts=load_texts(layout),
            similarity=load_similarity(layout),
            manifest=read_manifest(layout),
            negative=conflict_adjacency(negative_edges) if negative_edges else None,
            embedder=embedder,
            config=config,
            dedup=dedup,
            conflict=conflict,
            polarity=polarity,
            trace=trace,
        )

    @property
    def layout(self) -> IndexLayout:
        return self._layout

    @property
    def manifest(self) -> IndexManifest:
        return self._manifest

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def can_query(self) -> bool:
        """Whether a TEXT query could actually be embedded right now.

        Holding an index is not the same as being able to ask it something:
        `retrieve()` needs an embedder, and on an install without the
        `embed` extra there is none. A caller that offers a search box has
        to know the difference - the first browser face offered one on an
        install without sentence-transformers and returned an opaque 500.

        Cheap on purpose: it asks whether the module can be FOUND, and never
        loads the model. An injected embedder answers yes without any import
        at all.
        """
        if self._embedder is not None:
            return True
        from importlib.util import find_spec

        try:
            return find_spec("sentence_transformers") is not None
        except (ImportError, ValueError):  # pragma: no cover - broken meta path
            return False

    @property
    def traces(self) -> TraceStore:
        """The last calls this index answered, newest last (D38).

        In memory and on by default; `TraceConfig(directory=...)` mirrors
        them to JSONL and `TraceConfig(enabled=False)` records nothing.
        """
        return self._traces

    @property
    def store(self) -> VectorStore:
        return self._store

    def text_of(self, node_id: str) -> str:
        """The indexed text of one node; empty when the index carries none."""
        return self._texts.get(node_id, "")

    def retrieve(
        self,
        query: str,
        *,
        profile: str | None = None,
        config: RetrievalConfig | None = None,
        exclude: Sequence[str] | None = None,
        residue: Mapping[str, float] | None = None,
    ) -> Answer:
        """Inject `query` into the web and return what lit up, with text.

        `profile` names one of `precise` / `explore` / `compare`; it overlays
        exactly damping, threshold and seed width onto the base config and
        leaves everything else alone. Left as `None` with no config supplied
        here or at `open()`, the library picks `DEFAULT_PROFILE` - the bare
        `RetrievalConfig()` cannot spread past five seeds, and a silent top-k
        is the one thing this class must never hand back. A config you DID
        supply is never overlaid; `retrieve()` warns if it cannot spread.
        The answer reports which profile ran. `exclude` carries "without X" phrases -
        each becomes an energy-ABSORBING negative seed rather than a filter,
        because a filter cannot remove a passage that arrived only by being
        X's neighbour.
        """
        cfg, ran = self._resolve(config, profile)
        embedder = self._require_embedder()
        vector = embedder.embed_queries([query])[0]
        negatives = embedder.embed_queries(list(exclude)) if exclude else None
        started = time.perf_counter()
        result = _retrieve(
            vector,
            self._store,
            self._graph,
            cfg,
            similarity=self._similarity,
            dedup=self._dedup,
            source_of=self._source_of,
            negative=self._negative,
            conflict=self._conflict if self._negative else None,
            negative_queries=negatives,
            residue=residue,
            polarity=self._polarity,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return Answer(
            query=query,
            passages=self._passages(result.ranked(), result.votes(), result),
            result=result,
            graph=self._graph,
            trace=self._record(
                result,
                query=query,
                profile=ran,
                elapsed_ms=elapsed_ms,
                settings=_settings(cfg),
                propagation=cfg.propagation,
            ),
            profile=ran,
            config=cfg,
        )

    def retrieve_colored(
        self,
        parts: Mapping[str, str],
        *,
        profile: str | None = None,
        config: ColoredRetrievalConfig | None = None,
    ) -> ColoredAnswer:
        """One coloured seed set per query part; a node two colours reach is a bridge.

        The caller decomposes - never an LLM inside this class. `parts` maps a
        colour label to that part's text, and insertion order matters: the
        FIRST part is the primary one, and only its failure to touch the index
        is an error.

        No default profile here, unlike `retrieve()`: `ColoredRetrievalConfig()`
        is the measured operating point and clears the spread bar on every
        colour, so there is no trap to step around, and overlaying `explore`
        would move a measured width (2 per colour) for nothing.
        """
        base = config if config is not None else ColoredRetrievalConfig()
        if profile is not None:
            base = self._profile(profile).as_colored(base)
        embedder = self._require_embedder()
        labels = list(parts)
        vectors = embedder.embed_queries([parts[label] for label in labels])
        started = time.perf_counter()
        result = _retrieve_colored(
            dict(zip(labels, vectors, strict=True)),
            self._store,
            self._graph,
            base,
            similarity=self._similarity,
            dedup=self._dedup,
            source_of=self._source_of,
            negative=self._negative,
            conflict=self._conflict if self._negative else None,
            polarity=self._polarity,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        query = " | ".join(parts.values())
        return ColoredAnswer(
            query=query,
            passages=self._passages(result.ranked(), result.votes(), None),
            result=result,
            graph=self._graph,
            trace=self._record(
                result,
                query=query,
                profile=profile,
                elapsed_ms=elapsed_ms,
                settings=_settings(base),
                propagation=base.propagation,
                parts=dict(parts),
            ),
        )

    def close(self) -> None:
        """Drop the loaded artifacts. The object is not usable afterwards."""
        self._texts = {}
        self._negative = None

    def __enter__(self) -> SpiywebIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- internals ---------------------------------------------------------

    def _record(
        self,
        result: RetrievalResult | ColoredRetrievalResult,
        *,
        query: str,
        profile: str | None,
        elapsed_ms: float,
        settings: Mapping[str, object],
        propagation: PropagationConfig,
        parts: Mapping[str, str] | None = None,
    ) -> TraceRecord | None:
        """Keep what this call did, unless tracing was switched off (D38)."""
        if not self._traces.enabled:
            return None
        record = build_trace(
            result,
            self._graph,
            query=query,
            sequence=self._traces.next_sequence(),
            config=self._traces.config,
            texts=self._texts,
            parts=parts,
            index=str(self._layout.root),
            profile=profile or "",
            elapsed_ms=elapsed_ms,
            settings=settings,
            propagation_config=propagation,
        )
        return self._traces.append(record)

    def _resolve(
        self, config: RetrievalConfig | None, profile: str | None
    ) -> tuple[RetrievalConfig, str]:
        """The config this call runs with, and the name of the profile in it.

        The default profile applies only when the caller expressed no
        preference at all - no profile here, no config here, none at
        `open()`. A config the caller built is theirs as built.
        """
        base = config if config is not None else self._config
        if profile is None and base is None:
            profile = DEFAULT_PROFILE
        base = base if base is not None else RetrievalConfig()
        if profile:
            return self._profile(profile).as_retrieval(base), profile
        return base, ""

    @staticmethod
    def _profile(name: str | None) -> Profile:
        if name not in PROFILES:
            raise ValueError(
                f"unknown profile {name!r}; pick one of {sorted(PROFILES)}"
            )
        return PROFILES[name]

    def _require_embedder(self) -> Embedder:
        if self._embedder is None:
            from spiyweb.embedding import SentenceTransformerEmbedder

            self._embedder = SentenceTransformerEmbedder()
        if not self._checked_model:
            self._check_model(self._embedder)
            self._checked_model = True
        return self._embedder

    def _check_model(self, embedder: Embedder) -> None:
        """Refuse a query embedded by a model the index was not built with.

        Both names have to be known for this to fire. An index built before
        the store recorded one - the four sealed Phase 1 directories - simply
        cannot be checked, and a check that cannot be made is not a failure.
        """
        stored = self._store.model_name
        asked = getattr(embedder, "model_name", None)
        if stored and asked and stored != asked:
            raise ValueError(
                f"this index was built with embedding model {stored!r} but the "
                f"query embedder is {asked!r}; two models can share a dimension "
                "and cosine across two spaces returns confident nonsense. "
                "Rebuild the index or pass the matching embedder."
            )

    def _passages(
        self,
        ranked: list[tuple[str, float]],
        votes: Mapping[str, int],
        result: RetrievalResult | None,
    ) -> tuple[Passage, ...]:
        hops = (
            {node: act.hop for node, act in result.propagation.activations.items()}
            if result is not None
            else {}
        )
        passages: list[Passage] = []
        for node_id, energy in ranked:
            data = self._graph.node(node_id)
            source_id = data.source_id if data is not None else node_id
            passages.append(
                Passage(
                    node_id=node_id,
                    source_id=source_id,
                    layer=data.layer if data is not None else "chunk",
                    text=self._texts.get(node_id, ""),
                    energy=energy,
                    # Votes are keyed by the vote key, which is the SOURCE
                    # when the graph knows one - per document, never per
                    # chunk, because repetition inside one document is not
                    # corroboration.
                    votes=votes.get(source_id, votes.get(node_id, 1)),
                    hop=hops.get(node_id, 0),
                )
            )
        return tuple(passages)


def _settings(
    config: RetrievalConfig | ColoredRetrievalConfig,
) -> dict[str, object]:
    """The knobs this call ran with, flattened for the record.

    A trace whose settings are missing cannot be compared with another one,
    and comparing two runs is most of what a trace reader is for.
    """
    propagation = config.propagation
    return {
        "seed_width": config.seed_width,
        "contact_overfetch": getattr(config, "contact_overfetch", 1),
        "seed_energy": propagation.seed_energy,
        "damping": propagation.damping,
        "threshold_ratio": propagation.threshold_ratio,
        "max_hop": propagation.max_hop,
        "max_nodes": propagation.max_nodes,
        "split_alpha": propagation.split_alpha,
        "mass_enabled": propagation.mass.enabled,
    }


def open_index(
    path: IndexLayout | Path | str,
    *,
    weights: LayerWeights | None = None,
    embedder: Embedder | None = None,
    config: RetrievalConfig | None = None,
    dedup: DedupConfig | None = None,
    conflict: ConflictConfig | None = None,
    polarity: PolarityConfig | None = None,
    trace: TraceConfig | None = None,
) -> SpiywebIndex:
    """Open an index directory. Shorthand for `SpiywebIndex.open`."""
    return SpiywebIndex.open(
        path,
        weights=weights,
        embedder=embedder,
        config=config,
        dedup=dedup,
        conflict=conflict,
        polarity=polarity,
        trace=trace,
    )
