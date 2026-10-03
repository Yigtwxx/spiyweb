"""Incremental indexing: `sync_index` must equal a fresh `build_index`.

The guarantee under test is the simple one the docstring promises: after a
sync, every artifact on disk is the one `build_index` writes for the same
corpus from scratch - while only new or changed chunks were embedded,
entity-extracted and proposition-extracted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spiyweb.indexing import (
    DocumentInput,
    TextUnit,
    build_index,
    sync_index,
)


class CountingEmbedder:
    """Deterministic 3-d vectors from the text; records what it embedded."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [
            [1.0 + len(text) % 5, float(sum(map(ord, text)) % 11), 1.0]
            for text in texts
        ]


class CapitalisedWords:
    """spaCy stand-in: every capitalised word is an ORG entity."""

    def pipe(self, texts: list[str]) -> list[object]:
        class Span:
            def __init__(self, text: str) -> None:
                self.text = text
                self.label_ = "ORG"

        class Doc:
            def __init__(self, text: str) -> None:
                self.ents = tuple(
                    Span(word.strip(".,"))
                    for word in text.split()
                    if word[:1].isupper()
                )

        return [Doc(text) for text in texts]


class LinesLLM:
    """Proposition extractor stand-in: one proposition per sentence."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        passage = prompt.rsplit("Passage:", 1)[-1]
        return "\n".join(s.strip() for s in passage.split(".") if len(s.strip()) > 12)


def doc(source: str, *texts: str) -> DocumentInput:
    return DocumentInput(source_id=source, units=tuple(TextUnit(text=t) for t in texts))


BEFORE = [
    doc("d0", "Tesla built Wardenclyffe on Long Island.", "Morgan funded Tesla."),
    doc("d1", "Wardenclyffe drew power from the ground."),
    doc("d2", "Edison opposed Tesla in the current war."),
]
AFTER = [
    doc("d0", "Tesla built Wardenclyffe on Long Island.", "Morgan funded Tesla."),
    doc("d1", "Wardenclyffe was demolished by Morgan in 1917."),  # changed
    doc("d3", "Westinghouse licensed Tesla patents."),  # added; d2 removed
]


def _artifacts(root: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for path in sorted(root.glob("*.json")):
        out[path.name] = json.loads(path.read_text(encoding="utf-8"))
    with np.load(root / "vectors.npz") as payload:
        out["vector_ids"] = [str(i) for i in payload["ids"]]
        out["vectors"] = np.round(payload["vectors"], 6).tolist()
    return out


@pytest.mark.parametrize("propositions", [False, True])
def test_a_synced_index_equals_a_fresh_build(
    tmp_path: Path, propositions: bool
) -> None:
    llm = LinesLLM() if propositions else None
    fresh, synced = tmp_path / "fresh", tmp_path / "synced"
    options = {
        "entity_pipeline": CapitalisedWords(),
        "llm": llm,
        "embedding_model": "fake",
        "propositions": propositions,
        "log": lambda message: None,
    }
    build_index(AFTER, fresh, embedder=CountingEmbedder(), **options)  # type: ignore[arg-type]
    build_index(BEFORE, synced, embedder=CountingEmbedder(), **options)  # type: ignore[arg-type]
    report = sync_index(AFTER, synced, embedder=CountingEmbedder(), **options)  # type: ignore[arg-type]

    assert _artifacts(synced) == _artifacts(fresh), (
        "a sync must leave exactly the artifacts a fresh build writes"
    )
    assert (report.reused, report.added, report.changed, report.removed) == (2, 1, 1, 1)


def test_only_new_and_changed_chunks_are_embedded(tmp_path: Path) -> None:
    common = {"entity_pipeline": CapitalisedWords(), "log": lambda message: None}
    build_index(BEFORE, tmp_path, embedder=CountingEmbedder(), **common)  # type: ignore[arg-type]
    embedder = CountingEmbedder()
    sync_index(AFTER, tmp_path, embedder=embedder, **common)  # type: ignore[arg-type]

    assert sorted(embedder.embedded) == sorted(
        [
            "Wardenclyffe was demolished by Morgan in 1917.",
            "Westinghouse licensed Tesla patents.",
        ]
    )


def test_only_fresh_chunks_go_back_to_the_proposition_llm(tmp_path: Path) -> None:
    common = {
        "entity_pipeline": CapitalisedWords(),
        "propositions": True,
        "log": lambda message: None,
    }
    build_index(BEFORE, tmp_path, embedder=CountingEmbedder(), llm=LinesLLM(), **common)  # type: ignore[arg-type]
    llm = LinesLLM()
    sync_index(AFTER, tmp_path, embedder=CountingEmbedder(), llm=llm, **common)  # type: ignore[arg-type]

    assert llm.calls == 2, "one call for the changed chunk, one for the added one"


def test_an_unchanged_corpus_embeds_nothing(tmp_path: Path) -> None:
    common = {"entity_pipeline": CapitalisedWords(), "log": lambda message: None}
    build_index(BEFORE, tmp_path, embedder=CountingEmbedder(), **common)  # type: ignore[arg-type]
    embedder = CountingEmbedder()
    report = sync_index(BEFORE, tmp_path, embedder=embedder, **common)  # type: ignore[arg-type]

    assert embedder.embedded == []
    assert (report.added, report.changed, report.removed) == (0, 0, 0)


def test_an_empty_directory_is_simply_built(tmp_path: Path) -> None:
    report = sync_index(
        BEFORE,
        tmp_path,
        embedder=CountingEmbedder(),
        entity_pipeline=CapitalisedWords(),
        log=lambda message: None,
    )

    assert report.added == 4 and report.reused == 0
    assert (tmp_path / "meta.json").exists()


def test_another_embedding_model_is_refused_not_mixed(tmp_path: Path) -> None:
    common = {"entity_pipeline": CapitalisedWords(), "log": lambda message: None}
    build_index(
        BEFORE, tmp_path, embedder=CountingEmbedder(), embedding_model="a", **common
    )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="another"):
        sync_index(
            AFTER, tmp_path, embedder=CountingEmbedder(), embedding_model="b", **common
        )  # type: ignore[arg-type]
