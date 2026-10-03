"""Index-time facade: everything needed to BUILD a web, in one namespace.

`spiyweb` itself is the QUERY-time contract - `retrieve`, `propagate` and the
configs they read. Building the graph is a different job with a different
dependency profile, so it gets its own front door instead of doubling the
top-level `__all__` and blurring the layering the package is built on.

`build_index` takes documents and a directory. It knows nothing about any
benchmark: the MuSiQue/HotpotQA/2Wiki adapter lives in `spiyweb.evaluation`,
which now supplies `DocumentInput`s and a composed-text map to this module
rather than owning the pipeline. That split is the whole point - a corpus is
a corpus, and the sealed measurement runs go through exactly the same code a
user's own corpus does.

Every stage writes one file and skips itself when that file exists (the
resume mechanism; `force=True` rebuilds). The graph is deliberately NOT an
artifact: raw per-layer edges are, and `load_graph` re-merges them through
`Graph.from_layers` on every load - the same "one source of truth, rebuild
the derived thing" philosophy as the vector store, and it makes
`LayerWeights` ablations free (re-merge, never re-index).

The zero-dependency rule survives intact, and not by accident. Every name
re-exported below is pure Python at import time: `embedding` imports torch
inside `detect_device`, `entities` imports spaCy inside
`load_spacy_pipeline`, and `edges/` states the rule in its own docstring. The
numpy/faiss users here - `build_index`, `load_store`, `load_similarity` and
the two names in `_LAZY` - reach for `spiyweb.store` inside the call, so
`import spiyweb.indexing` still works with nothing installed and only DOING
index work asks for `spiyweb[index]`.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from spiyweb.config import (
    EdgeLayer,
    EntityEdgeConfig,
    EntityExtractionConfig,
    LLMConfig,
    NLICandidateConfig,
    NLIEdgeConfig,
    PropositionConfig,
    SemanticEdgeConfig,
    StructuralEdgeConfig,
)
from spiyweb.core.conflict import NegativeEdge
from spiyweb.core.graph import Graph, Node
from spiyweb.edges import (
    ChunkRef,
    build_derivation_edges,
    build_entity_edges,
    build_nli_edges,
    build_semantic_edges,
    build_structural_edges,
    shared_subject_pairs,
)
from spiyweb.embedding import (
    Embedder,
    EncoderLike,
    SentenceTransformerEmbedder,
    detect_device,
    resolve_device,
)
from spiyweb.entities import EntityPipeline, extract_entities, load_spacy_pipeline
from spiyweb.llm import LLMClient, LLMError, NativeOllamaClient, OpenAICompatClient
from spiyweb.nodes import (
    Chunk,
    DocumentInput,
    Proposition,
    TextUnit,
    chunk_documents,
    extract_propositions,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from spiyweb.config import CorpusLintConfig, LayerWeights
    from spiyweb.core.dedup import SimilarityFn
    from spiyweb.edges import NLIModel
    from spiyweb.lint import LintReport
    from spiyweb.store import VectorStore, build_semantic_edges_fast

SCHEMA_VERSION = 1
"""Artifact-schema version written into `meta.json`.

Bumped only when a reader has to behave differently, never for an added
field: every loader here treats an absent key as its documented default, and
the four sealed Phase 1 indexes carry no version at all - they read as 0.
"""

_EDGE_LAYER_FILES: tuple[EdgeLayer, ...] = (
    "semantic",
    "entity",
    "structural",
    "derivation",
)

_LAZY: dict[str, str] = {
    "VectorStore": "spiyweb.store",
    "build_semantic_edges_fast": "spiyweb.store",
}

__all__ = [
    "SCHEMA_VERSION",
    "Chunk",
    "ChunkRef",
    "DocumentInput",
    "Embedder",
    "EncoderLike",
    "EntityPipeline",
    "IndexLayout",
    "IndexManifest",
    "LLMClient",
    "LLMError",
    "NativeOllamaClient",
    "OpenAICompatClient",
    # `Proposition` and `extract_propositions` are imported above and
    # used here, but they are NOT re-declared: they already belong to the
    # query-time contract in `spiyweb.__all__`. One name, one home - it is
    # what makes the two surfaces testably disjoint.
    "SentenceTransformerEmbedder",
    "TextUnit",
    "VectorStore",
    "build_entity_edges",
    "build_index",
    "build_semantic_edges",
    "build_semantic_edges_fast",
    "build_structural_edges",
    "chunk_documents",
    "detect_device",
    "extract_entities",
    "lint_index",
    "load_entities",
    "load_graph",
    "load_nli_edges",
    "load_propositions",
    "load_similarity",
    "load_spacy_pipeline",
    "load_store",
    "load_texts",
    "read_manifest",
    "resolve_device",
    "shared_subject_pairs",
]


@dataclass(frozen=True)
class IndexLayout:
    """Path schema of one index directory - artifact names live here only.

    Corpus-agnostic on purpose. `spiyweb.evaluation.index.IndexPaths` extends
    it with the benchmark's own files (the dataset downloads, `results.json`,
    `per_query.jsonl`), so the harness keeps its names without pushing them
    into the library.
    """

    root: Path

    @classmethod
    def at(cls, target: IndexLayout | Path | str) -> IndexLayout:
        """Coerce a layout, a path or a string into a layout.

        A subclass passes through intact, which is what lets the harness hand
        its own `IndexPaths` to every loader here without conversion.
        """
        if isinstance(target, IndexLayout):
            return target
        return cls(root=Path(target))

    @property
    def vectors_npz(self) -> Path:
        return self.root / "vectors.npz"

    @property
    def entities_json(self) -> Path:
        return self.root / "entities.json"

    @property
    def nodes_json(self) -> Path:
        return self.root / "nodes.json"

    @property
    def texts_json(self) -> Path:
        """Chunk id -> the text that was actually indexed.

        Added 2026-08-25. Without it a result is a list of node ids and the
        caller has to hold the corpus themselves, which is exactly what kept
        `retrieve()` from being usable outside this repository. Absent in the
        four sealed indexes; `load_texts` reads that as "no texts".
        """
        return self.root / "texts.json"

    @property
    def propositions_json(self) -> Path:
        return self.root / "propositions.json"

    def edges_json(self, layer: str) -> Path:
        return self.root / f"edges_{layer}.json"

    @property
    def nli_json(self) -> Path:
        # NOT an EdgeLayer artifact: negative edges never enter the graph
        # merge; `load_nli_edges` hands them to the conflict mechanism.
        return self.root / "edges_nli.json"

    @property
    def meta_json(self) -> Path:
        return self.root / "meta.json"

    @property
    def llm_cache_jsonl(self) -> Path:
        return self.root / "llm_cache.jsonl"

    def llm_cache_for(self, model: str | None) -> Path:
        """The LLM cache file for `model` - one file per model.

        The cache key is the prompt alone, by design, so two models sharing a
        file would silently answer for each other. The default model keeps the
        historical `llm_cache.jsonl`; any other gets its own tagged file.
        """
        if model is None or model == LLMConfig().model:
            return self.llm_cache_jsonl
        tag = re.sub(r"[^A-Za-z0-9._-]", "-", model)
        return self.root / f"llm_cache_{tag}.jsonl"


@dataclass(frozen=True)
class IndexManifest:
    """What an index directory holds, as one value instead of a file read.

    Returned by `build_index` and reconstructed by `read_manifest`. It is a
    VIEW over `meta.json`, never a second artifact: one receipt, one place it
    can go stale.
    """

    schema_version: int
    root: Path
    chunks: int
    propositions: int
    nodes: int
    dimension: int
    embedding_model: str | None
    edges: Mapping[str, int]
    nli_edges: int | None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_vectors(layout: IndexLayout) -> tuple[list[str], list[list[float]]]:
    """Ids and vectors straight from the store artifact (single source)."""
    import numpy as np

    with np.load(layout.vectors_npz) as payload:
        ids = [str(chunk_id) for chunk_id in payload["ids"]]
        vectors = payload["vectors"].tolist()
    return ids, vectors


def build_index(
    documents: Sequence[DocumentInput],
    out_dir: IndexLayout | Path | str,
    *,
    embedder: Embedder,
    entity_pipeline: EntityPipeline,
    texts: Mapping[str, str] | None = None,
    llm: LLMClient | None = None,
    embedding_model: str | None = None,
    extraction_config: EntityExtractionConfig | None = None,
    semantic_config: SemanticEdgeConfig | None = None,
    structural_config: StructuralEdgeConfig | None = None,
    entity_config: EntityEdgeConfig | None = None,
    llm_model: str | None = None,
    entity_llm: bool = True,
    propositions: bool = False,
    proposition_config: PropositionConfig | None = None,
    nli_model: NLIModel | None = None,
    nli_model_name: str | None = None,
    nli_config: NLIEdgeConfig | None = None,
    nli_candidates: NLICandidateConfig | None = None,
    extra_meta: Mapping[str, object] | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> IndexManifest:
    """Build (or resume) every index artifact for `documents` under `out_dir`.

    `texts` overrides what each chunk is embedded and entity-extracted AS,
    keyed by chunk id (`{source_id}:{position}`). The default is the chunk's
    own text. The benchmark adapter passes `title + newline + text`, because
    MuSiQue titles are entity-dense and both consumers must see the same
    string; a caller whose corpus has no titles simply omits it. Keys that
    name no chunk are rejected rather than ignored - a mis-keyed map would
    otherwise silently embed the wrong strings.

    `llm_model`, `nli_model_name` and `embedding_model` are receipt data only
    (the model names behind the injected objects, recorded in `meta.json`);
    the protocols deliberately hide them. `entity_llm=False` keeps the entity
    fallback off even when `llm` is supplied - the two LLM consumers (entity
    fallback, proposition extraction) are independently switchable ablations.

    `propositions=True` adds the second node layer (D10): one LLM call per
    chunk extracts atomic propositions, which then join every downstream
    stage - embedding, entity extraction, and the `derivation` edge layer.
    Opt-in because the cost is an open question (#2); the call count is
    logged BEFORE the first call, as ever.

    `nli_model` (opt-in, D26) adds the index-time contradiction stage: the
    corpus's high-cosine pairs - propositions when the index has them, chunks
    otherwise - are scored by the model and the survivors land in
    `edges_nli.json` as negative edges for the conflict mechanism, loaded
    back by `load_nli_edges`. Candidates must also NAME THE SAME SUBJECT
    (`NLICandidateConfig.require_shared_subject`); both counts are logged
    BEFORE the model runs.

    `extra_meta` is merged into the receipt, for a caller that records
    something this module has no business knowing - the harness writes its
    sampled question count that way.
    """
    from spiyweb.store import VectorStore, build_semantic_edges_fast

    layout = IndexLayout.at(out_dir)
    extraction_cfg = (
        extraction_config if extraction_config is not None else EntityExtractionConfig()
    )

    chunks = chunk_documents(list(documents))
    ids = [chunk.node.id for chunk in chunks]
    known = set(ids)
    supplied = dict(texts) if texts is not None else {}
    unknown = sorted(key for key in supplied if key not in known)
    if unknown:
        raise ValueError(
            f"`texts` names {len(unknown)} chunk id(s) this corpus does not "
            f"contain, first {unknown[:3]}; keys are `{{source_id}}:{{position}}`"
        )
    chunk_texts = {
        chunk.node.id: supplied.get(chunk.node.id, chunk.text) for chunk in chunks
    }

    extracted: list[Proposition] = []
    if propositions:
        if force or not layout.propositions_json.exists():
            # The LLM is required to EXTRACT, not to reopen. Demanding one
            # before checking the artifact forced every later stage - the NLI
            # pass, an ablation re-merge - to stand up an Ollama client for a
            # file that was already on disk.
            if llm is None:
                raise ValueError("proposition extraction requires an LLM client")
            # The cost is visible up front: exactly one call per passage.
            log(f"extracting propositions: {len(ids)} passages, one LLM call each ...")
            fresh = extract_propositions(
                chunks, llm, proposition_config, texts=chunk_texts
            )
            _write_json(
                layout.propositions_json,
                [
                    {
                        "id": p.node.id,
                        "chunk_id": p.chunk_id,
                        "source_id": p.node.source_id,
                        "length": p.node.length,
                        "timestamp": p.node.timestamp,
                        "polarity": p.node.polarity,
                        "text": p.text,
                    }
                    for p in fresh
                ],
            )
        else:
            log("propositions exist, skipping the extraction stage")
        extracted = load_propositions(layout)
        log(f"proposition layer: {len(extracted)} nodes")

    all_ids = ids + [p.node.id for p in extracted]
    texts_all = {**chunk_texts, **{p.node.id: p.text for p in extracted}}
    all_nodes = [chunk.node for chunk in chunks] + [p.node for p in extracted]

    if force or not layout.nodes_json.exists():
        _write_json(
            layout.nodes_json,
            [
                {
                    "id": node.id,
                    "layer": node.layer,
                    "source_id": node.source_id,
                    "length": node.length,
                    "timestamp": node.timestamp,
                    "cluster_id": node.cluster_id,
                    "polarity": node.polarity,
                }
                for node in all_nodes
            ],
        )

    if force or not layout.texts_json.exists():
        # Chunks only: a proposition's text is already in its own artifact,
        # and one node's text living in two files is a drift waiting to
        # happen. `load_texts` reads both and merges.
        _write_json(layout.texts_json, chunk_texts)

    dimension: int | None = None
    if force or not layout.vectors_npz.exists():
        log(f"embedding {len(all_ids)} passages ...")
        vectors = embedder.embed_passages([texts_all[node_id] for node_id in all_ids])
        dimension = len(vectors[0])
        store = VectorStore(dimension)
        store.add(all_ids, vectors)
        layout.root.mkdir(parents=True, exist_ok=True)
        store.save(layout.vectors_npz, model_name=embedding_model)
    else:
        log("vectors exist, skipping the embed stage")

    if force or not layout.entities_json.exists():
        log(f"extracting entities from {len(all_ids)} passages (spaCy bulk) ...")
        entities = extract_entities(texts_all, entity_pipeline, extraction_cfg)
        pending = {
            chunk_id: texts_all[chunk_id]
            for chunk_id, found in entities.items()
            if len(found) < extraction_cfg.min_entities
        }
        # The count is logged BEFORE any LLM call: the cost must be visible
        # up front, never discovered from a stalled progress bar.
        fallback = llm is not None and entity_llm
        log(
            f"{len(pending)} of {len(all_ids)} passages fall below "
            f"min_entities={extraction_cfg.min_entities}"
            + (" and go to the LLM" if fallback else "; LLM fallback is OFF")
        )
        if fallback and pending:
            refreshed = extract_entities(
                pending, entity_pipeline, extraction_cfg, llm=llm
            )
            entities.update(refreshed)
        _write_json(layout.entities_json, entities)
    else:
        log("entities exist, skipping the extraction stage")

    edge_builders: dict[str, Callable[[], list[tuple[str, str, float]]]] = {
        "semantic": lambda: build_semantic_edges_fast(
            *_load_vectors(layout), config=semantic_config
        ),
        "entity": lambda: build_entity_edges(
            load_entities(layout), config=entity_config
        ),
        "structural": lambda: build_structural_edges(
            [chunk.ref for chunk in chunks], config=structural_config
        ),
        # Written even when empty, so `load_graph` never guesses whether an
        # index was built with the proposition layer.
        "derivation": lambda: build_derivation_edges(extracted),
    }
    edge_counts: dict[str, int] = {}
    for layer, builder in edge_builders.items():
        target = layout.edges_json(layer)
        if not force and target.exists():
            log(f"{layer} edges exist, skipping")
            existing = _read_json(target)
            edge_counts[layer] = len(existing) if isinstance(existing, list) else 0
            continue
        edges = builder()
        # Per-layer counts make an entity-clique blow-up visible immediately;
        # the fix is config (max_df_ratio), never code.
        log(f"{layer} layer: {len(edges)} edges")
        _write_json(target, [[u, v, w] for u, v, w in edges])
        edge_counts[layer] = len(edges)

    nli_edge_count: int | None = None
    if nli_model is not None:
        if not force and layout.nli_json.exists():
            log("nli edges exist, skipping the contradiction stage")
            nli_edge_count = len(load_nli_edges(layout))
        else:
            candidate_cfg = (
                nli_candidates if nli_candidates is not None else NLICandidateConfig()
            )
            # Contradiction is sharp on propositions and blurry on chunks
            # (D26); pair within whichever layer this index actually has.
            subset = [p.node.id for p in extracted] if extracted else list(ids)
            positions = {node_id: index for index, node_id in enumerate(all_ids)}
            vector_ids, vectors_all = _load_vectors(layout)
            if vector_ids != all_ids:
                raise ValueError(
                    "vectors.npz id order does not match the node registry; "
                    "rebuild the embed stage before the NLI stage"
                )
            pairs = build_semantic_edges_fast(
                subset,
                [vectors_all[positions[node_id]] for node_id in subset],
                config=SemanticEdgeConfig(
                    k=candidate_cfg.top_k,
                    min_similarity=candidate_cfg.min_similarity,
                ),
            )
            pairs.sort(key=lambda edge: edge[2], reverse=True)
            candidates = [(u, v) for u, v, _ in pairs]
            if candidate_cfg.require_shared_subject:
                # Cosine says "these two texts look alike", which on an
                # encyclopaedic corpus is mostly "same kind of thing" - two
                # radio stations, two villages. NLI then reads them as claims
                # about ONE subject and reports a contradiction that lives in
                # its own assumption (#10, measured 2026-08-16). The subject
                # test runs BEFORE the cap so the cap spends its budget on
                # pairs that can actually contradict.
                before = len(candidates)
                candidates = shared_subject_pairs(
                    candidates,
                    texts_all,
                    load_entities(layout),
                    candidate_cfg.subject_prefix_chars,
                    candidate_cfg.max_subject_df_ratio,
                )
                log(
                    f"nli candidates: {before - len(candidates)} of {before} pairs "
                    "cut for naming no shared subject"
                )
            dropped = max(0, len(candidates) - candidate_cfg.max_pairs)
            if dropped:
                log(f"nli candidates capped: {dropped} lowest-similarity pairs cut")
            candidates = candidates[: candidate_cfg.max_pairs]
            layer_name = "proposition" if extracted else "chunk"
            # The cost is visible up front: two directed scores per pair.
            log(
                f"nli stage: scoring {len(candidates)} {layer_name} pairs "
                "(2 directed scores each) ..."
            )
            negative = build_nli_edges(candidates, texts_all, nli_model, nli_config)
            log(f"nli stage: {len(negative)} contradiction edges")
            _write_json(
                layout.nli_json,
                [[edge.source, edge.target, edge.strength] for edge in negative],
            )
            nli_edge_count = len(negative)

    if dimension is None:
        dimension = _stored_dimension(layout)

    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "corpus_chunks": len(ids),
        "propositions": len(extracted) if propositions else None,
        "proposition_config": (
            asdict(
                proposition_config
                if proposition_config is not None
                else PropositionConfig()
            )
            if propositions
            else None
        ),
        "nodes": len(all_ids),
        "dimension": dimension,
        "embedding_model": embedding_model,
        "edges": dict(edge_counts),
        "entity_llm": llm is not None and entity_llm,
        "llm_model": llm_model if llm is not None else None,
        "nli_model": nli_model_name if nli_model is not None else None,
        "nli_edges": nli_edge_count,
        "nli_config": (
            asdict(nli_config if nli_config is not None else NLIEdgeConfig())
            if nli_model is not None
            else None
        ),
        "nli_candidates": (
            asdict(
                nli_candidates if nli_candidates is not None else NLICandidateConfig()
            )
            if nli_model is not None
            else None
        ),
        "llm_cache": layout.llm_cache_jsonl.name,
        "extraction_config": {
            **asdict(extraction_cfg),
            "labels": sorted(extraction_cfg.labels),
        },
        "semantic_config": asdict(
            semantic_config if semantic_config is not None else SemanticEdgeConfig()
        ),
        "structural_config": asdict(
            structural_config
            if structural_config is not None
            else StructuralEdgeConfig()
        ),
        "entity_config": asdict(
            entity_config if entity_config is not None else EntityEdgeConfig()
        ),
    }
    if extra_meta:
        receipt.update(extra_meta)
    _write_json(layout.meta_json, receipt)

    return IndexManifest(
        schema_version=SCHEMA_VERSION,
        root=layout.root,
        chunks=len(ids),
        propositions=len(extracted),
        nodes=len(all_ids),
        dimension=dimension,
        embedding_model=embedding_model,
        edges=dict(edge_counts),
        nli_edges=nli_edge_count,
    )


def _stored_dimension(layout: IndexLayout) -> int:
    import numpy as np

    with np.load(layout.vectors_npz) as payload:
        return int(payload["vectors"].shape[1])


def read_manifest(target: IndexLayout | Path | str) -> IndexManifest:
    """Reconstruct the manifest from `meta.json`.

    Every field falls back to what a pre-versioned index implies, so the four
    sealed Phase 1 directories read without special-casing: no
    `schema_version` means 0, no `embedding_model` means unknown, and the
    edge counts come from the artifacts themselves.
    """
    layout = IndexLayout.at(target)
    meta = _read_json(layout.meta_json)
    if not isinstance(meta, dict):
        raise ValueError(f"{str(layout.meta_json)!r} is not an index receipt")
    edges = meta.get("edges")
    if not isinstance(edges, dict):
        edges = {}
        for layer in _EDGE_LAYER_FILES:
            path = layout.edges_json(layer)
            payload = _read_json(path) if path.exists() else []
            edges[layer] = len(payload) if isinstance(payload, list) else 0
    chunks = int(meta.get("corpus_chunks", 0))
    propositions = meta.get("propositions") or 0
    dimension = meta.get("dimension")
    return IndexManifest(
        schema_version=int(meta.get("schema_version", 0)),
        root=layout.root,
        chunks=chunks,
        propositions=int(propositions),
        nodes=int(meta.get("nodes", chunks + int(propositions))),
        dimension=int(dimension)
        if dimension is not None
        else _stored_dimension(layout),
        embedding_model=meta.get("embedding_model"),
        edges=edges,
        nli_edges=meta.get("nli_edges"),
    )


def load_propositions(target: IndexLayout | Path | str) -> list[Proposition]:
    """Rebuild `Proposition`s from the artifact (single source of truth)."""
    layout = IndexLayout.at(target)
    if not layout.propositions_json.exists():
        return []
    payload = _read_json(layout.propositions_json)
    if not isinstance(payload, list):
        raise ValueError(f"{str(layout.propositions_json)!r} is not a proposition list")
    propositions: list[Proposition] = []
    for record in payload:
        timestamp = record.get("timestamp")
        propositions.append(
            Proposition(
                node=Node(
                    id=str(record["id"]),
                    layer="proposition",
                    source_id=str(record["source_id"]),
                    length=int(record["length"]),
                    timestamp=float(timestamp) if timestamp is not None else None,
                    # Absent in pre-#11 artifacts; positive is Node's default.
                    polarity=int(record.get("polarity", 1)),  # type: ignore[arg-type]
                ),
                chunk_id=str(record["chunk_id"]),
                text=str(record["text"]),
            )
        )
    return propositions


def load_entities(target: IndexLayout | Path | str) -> dict[str, list[str]]:
    layout = IndexLayout.at(target)
    payload = _read_json(layout.entities_json)
    if not isinstance(payload, dict):
        raise ValueError(f"{str(layout.entities_json)!r} is not an entities mapping")
    return {str(chunk_id): list(found) for chunk_id, found in payload.items()}


def load_texts(target: IndexLayout | Path | str) -> dict[str, str]:
    """Node id -> the text that was indexed, chunks and propositions alike.

    An index built before `texts.json` existed - the four sealed Phase 1
    directories - returns whatever its proposition artifact carries, and
    nothing for its chunks. That is reported as an empty string per node
    rather than an error: a caller that only wants the ranking should not be
    stopped by a missing convenience artifact.
    """
    layout = IndexLayout.at(target)
    texts: dict[str, str] = {}
    if layout.texts_json.exists():
        payload = _read_json(layout.texts_json)
        if not isinstance(payload, dict):
            raise ValueError(f"{str(layout.texts_json)!r} is not a text mapping")
        texts.update({str(node_id): str(text) for node_id, text in payload.items()})
    for proposition in load_propositions(layout):
        texts[proposition.node.id] = proposition.text
    return texts


def load_nli_edges(target: IndexLayout | Path | str) -> list[NegativeEdge]:
    """Negative (contradiction) edges of an index, empty when never built.

    An index without the NLI stage simply has no `edges_nli.json` - like the
    derivation layer, an absent optional artifact means "none", not an error.
    Feed the result through `conflict_adjacency` into `retrieve(negative=)`.
    """
    layout = IndexLayout.at(target)
    if not layout.nli_json.exists():
        return []
    payload = _read_json(layout.nli_json)
    if not isinstance(payload, list):
        raise ValueError(f"{str(layout.nli_json)!r} is not an edge list")
    return [
        NegativeEdge(source=str(u), target=str(v), strength=float(s))
        for u, v, s in payload
    ]


def load_store(target: IndexLayout | Path | str) -> VectorStore:
    """The query-time seed source, rebuilt from the vectors artifact."""
    from spiyweb.store import VectorStore as _VectorStore

    return _VectorStore.load(IndexLayout.at(target).vectors_npz)


def load_similarity(target: IndexLayout | Path | str) -> SimilarityFn:
    """Node-pair cosine over the stored vectors - the OTHER half of dedup.

    `retrieve()` keeps redundancy suppression off unless it receives BOTH an
    enabled `DedupConfig` and one of these. That is a deliberate contract (a
    caller with no embeddings cannot honestly dedup), and it is also how the
    harness ran the whole 2026-08-14/16 campaign with the mechanism silently
    disabled: it passed neither. `SpiywebIndex` wires both halves for you.

    An unknown id scores `0.0` instead of raising, matching the inspector's
    reading of the same artifacts: a graph and a vector file that disagree
    about one node must not end a run.
    """
    import numpy as np

    layout = IndexLayout.at(target)
    ids, rows = _load_vectors(layout)
    matrix = np.asarray(rows, dtype=np.float32)
    position = {node_id: index for index, node_id in enumerate(ids)}

    def similarity(node: str, others: Sequence[str]) -> Sequence[float]:
        row = position.get(node)
        if row is None or not others:
            return [0.0] * len(others)
        columns = [position.get(other, -1) for other in others]
        known = np.array([index for index in columns if index >= 0], dtype=np.int64)
        scores = np.zeros(len(others), dtype=np.float64)
        if known.size:
            values = matrix[known] @ matrix[row]
            slot = 0
            for offset, index in enumerate(columns):
                if index >= 0:
                    scores[offset] = float(values[slot])
                    slot += 1
        return scores.tolist()

    return similarity


def load_graph(
    target: IndexLayout | Path | str, weights: LayerWeights | None = None
) -> Graph:
    """Re-merge the raw edge layers into a graph under `weights`.

    Merging at load time (instead of persisting a merged graph) is what makes
    a `LayerWeights` ablation a re-merge instead of a re-index.
    """
    layout = IndexLayout.at(target)
    layers: dict[EdgeLayer, list[tuple[str, str, float]]] = {}
    for layer in _EDGE_LAYER_FILES:
        path = layout.edges_json(layer)
        if layer == "derivation" and not path.exists():
            # Indexes built before the proposition layer carry no derivation
            # file; an absent optional layer is empty, not an error.
            layers[layer] = []
            continue
        payload = _read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"{str(path)!r} is not an edge list")
        layers[layer] = [(str(u), str(v), float(w)) for u, v, w in payload]

    nodes_payload = _read_json(layout.nodes_json)
    if not isinstance(nodes_payload, list):
        raise ValueError(f"{str(layout.nodes_json)!r} is not a node list")
    nodes = []
    for record in nodes_payload:
        timestamp = record.get("timestamp")
        cluster_id = record.get("cluster_id")
        nodes.append(
            Node(
                id=str(record["id"]),
                layer=record["layer"],
                source_id=str(record["source_id"]),
                length=int(record["length"]),
                # Absent in pre-2026-08-14 artifacts; defaults match Node's.
                timestamp=float(timestamp) if timestamp is not None else None,
                cluster_id=str(cluster_id) if cluster_id is not None else None,
                polarity=int(record.get("polarity", 1)),  # type: ignore[arg-type]
            )
        )

    return Graph.from_layers(layers, weights=weights, nodes=nodes)


def lint_index(
    target: IndexLayout | Path | str,
    *,
    weights: LayerWeights | None = None,
    config: CorpusLintConfig | None = None,
) -> LintReport:
    """Read an index's artifacts and inspect the shape of its corpus (D37).

    The thin I/O half of `spiyweb.lint`, which is pure by design. Two things
    are read that `load_graph` deliberately throws away:

    - the RAW semantic layer, because the merged adjacency sums layers and a
      summed weight is not a cosine - "are these two passages near-identical"
      cannot be asked of it;
    - the NLI edges, which never enter the merged graph at all (contradiction
      is negative charge, not a negative weight).

    An index built without either simply yields no findings of that kind.

    The per-layer edge COUNTS are read too, and they are the reason this
    function is worth having rather than inlining. The merged adjacency
    cannot tell an empty layer from an absent one, so a layer configured at
    a real weight that carries nothing is invisible after the merge - which
    is exactly how the structural layer stayed empty across four sealed
    measurement runs without anyone noticing.
    """
    from spiyweb.lint import lint_corpus

    layout = IndexLayout.at(target)
    semantic_path = layout.edges_json("semantic")
    semantic: list[tuple[str, str, float]] = []
    if semantic_path.exists():
        payload = _read_json(semantic_path)
        if not isinstance(payload, list):
            raise ValueError(f"{str(semantic_path)!r} is not an edge list")
        semantic = [(str(u), str(v), float(w)) for u, v, w in payload]
    counts: dict[str, int] = {}
    for layer in _EDGE_LAYER_FILES:
        path = layout.edges_json(layer)
        if not path.exists():
            continue
        payload = _read_json(path)
        counts[layer] = len(payload) if isinstance(payload, list) else 0
    return lint_corpus(
        load_graph(layout, weights),
        semantic_edges=semantic,
        negative_edges=load_nli_edges(layout),
        layer_edges=counts,
        weights=weights,
        config=config,
    )


def __getattr__(name: str) -> object:
    """PEP 562: the two FAISS-bound names, resolved on first touch.

    `spiyweb.store` raises the install hint itself, so this neither wraps nor
    re-words it - the caller gets `pip install spiyweb[store]` verbatim.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    """List the declared surface, the two lazy names included.

    Without this they are missing from `dir()` until something touches them -
    they are not module attributes before that. The override exists for those
    two and nothing else, and the cost is that module internals stop showing
    up here.
    """
    return sorted(__all__)
