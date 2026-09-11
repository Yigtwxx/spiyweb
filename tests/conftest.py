"""Shared fixtures. The tiny corpus lives here so nothing imports a test file.

`test_trace.py` and `test_viewer.py` both need a real index - built by the
real `build_index`, with real entity edges and a real FAISS store - over a
corpus small enough to reason about by hand. Building it twice would double
the suite's slowest fixture, and importing one test module from another makes
collection order load-bearing, which is a bug waiting for a rainy day.

The corpus is four passages in two documents, and the numbers in it are
chosen rather than sampled: the passage directions are orthogonal enough that
a seed's cosine neighbourhood never crosses the document boundary on its own,
so `beta:0` can only be reached through the rare entity the two documents
share. That is the multi-hop shape the whole project is about, in miniature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spiyweb import open_index
from spiyweb.config import (
    EntityEdgeConfig,
    EntityExtractionConfig,
    SemanticEdgeConfig,
    TraceConfig,
)
from spiyweb.indexing import DocumentInput, TextUnit, build_index

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spiyweb.session import SpiywebIndex
    from spiyweb.trace import TraceRecord

DOCUMENTS = (
    DocumentInput(
        source_id="alpha",
        units=(
            TextUnit(text="The tower was raised on the shore."),
            TextUnit(text="Wardenclyffe drew power from the ground."),
        ),
    ),
    DocumentInput(
        source_id="beta",
        units=(
            TextUnit(text="What happened at Wardenclyffe afterwards."),
            TextUnit(text="An unrelated closing paragraph."),
        ),
    ),
)

PASSAGES = {
    "The tower was raised on the shore.": [1.0, 0.0, 0.0],
    "Wardenclyffe drew power from the ground.": [0.8, 0.6, 0.0],
    "What happened at Wardenclyffe afterwards.": [0.0, 1.0, 0.0],
    "An unrelated closing paragraph.": [0.0, 0.0, 1.0],
}

DIRECTIONS = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.8, 0.0])
"""Three query directions, picked by a stable property of the text, so a
fifty-question loop needs no fifty-entry lookup table."""


class FakeEmbedder:
    """Deterministic vectors: the tests are about the web, not the model."""

    model_name = "fake-e5"

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [PASSAGES[text] for text in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [DIRECTIONS[len(text) % len(DIRECTIONS)] for text in texts]


class KeywordPipeline:
    """A spaCy stand-in that finds exactly the two entities that matter."""

    def pipe(self, texts: list[str]) -> list[object]:
        class Span:
            def __init__(self, text: str) -> None:
                self.text = text
                self.label_ = "ORG"

        class Doc:
            def __init__(self, ents: tuple[object, ...]) -> None:
                self.ents = ents

        docs: list[object] = []
        for text in texts:
            found: list[object] = []
            if "Wardenclyffe" in text:
                found.append(Span("Wardenclyffe"))
            if "tower" in text.lower():
                found.append(Span("tower"))
            docs.append(Doc(tuple(found)))
        return docs


@pytest.fixture(scope="session")
def tiny_index_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real index over the tiny corpus, built once for the whole session."""
    root = tmp_path_factory.mktemp("tiny-index")
    build_index(
        DOCUMENTS,
        root,
        embedder=FakeEmbedder(),
        entity_pipeline=KeywordPipeline(),
        embedding_model="fake-e5",
        extraction_config=EntityExtractionConfig(min_entities=0),
        semantic_config=SemanticEdgeConfig(k=2, min_similarity=0.0),
        entity_config=EntityEdgeConfig(max_df_ratio=1.0),
        log=lambda message: None,
    )
    return root


@pytest.fixture
def open_tiny(
    tiny_index_root: Path,
) -> Callable[..., SpiywebIndex]:
    """Open the tiny index with the fake embedder already wired in."""

    def _open(trace: TraceConfig | None = None, **options: object) -> SpiywebIndex:
        # Never attach to a developer's live monitor: a `spiyweb` running in
        # the repo root would otherwise collect every test query.
        if trace is None:
            trace = TraceConfig(attach_dir=None)
        return open_index(
            tiny_index_root, embedder=FakeEmbedder(), trace=trace, **options
        )

    return _open


def canonical_record(**overrides: object) -> TraceRecord:
    """The §2.6 trace as a hand-built record: A, C at hop 0; B, D at hop 1
    with D converging; F at hop 2; A_dup suppressed by A. No index needed."""
    from spiyweb.trace import TraceEvent, TraceNode, TracePath, TraceRecord

    def node(id_: str, hop: int, energy: float, **extra: object) -> TraceNode:
        return TraceNode(
            id=id_,
            source_id=id_.lower(),
            layer="chunk",
            energy=energy,
            hop=hop,
            votes=int(extra.pop("votes", 1)),
            **extra,  # type: ignore[arg-type]
        )

    fields: dict[str, object] = dict(
        trace_id="canonical",
        sequence=0,
        recorded_at="2026-09-11T00:00:00.000+00:00",
        kind="plain",
        query="which tower did tesla build",
        nodes=(
            node("A", 0, 5.625, votes=2, seed_similarity=0.9),
            node("C", 0, 4.375, seed_similarity=0.7),
            node("D", 1, 2.875),
            node("B", 1, 2.25),
            node("F", 2, 1.725),
            node("A_dup", -1, 0.0, suppressed_by="A"),
        ),
        edges=(),
        paths=(
            TracePath(node="D", steps=("A", "D"), hop=1, energy=2.875, converging=1),
            TracePath(
                node="F", steps=("A", "D", "F"), hop=2, energy=1.725, converging=0
            ),
        ),
        clusters=(),
        events=(TraceEvent(kind="suppressed", node="A_dup", other="A"),),
        stop_reason="threshold",
        hops_used=2,
        injected_energy=10.0,
        threshold=1.5,
        total_energy=16.85,
        node_count=5,
        dedup_mode="adaptive",
        profile="explore",
        elapsed_ms=3.0,
    )
    fields.update(overrides)
    return TraceRecord(**fields)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _never_attach_to_a_live_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test, always: whatever `.spiyweb/watch` sits in the cwd, records
    stay out of it. The attach tests opt back in by clearing the variable."""
    monkeypatch.setenv("SPIYWEB_NO_ATTACH", "1")
