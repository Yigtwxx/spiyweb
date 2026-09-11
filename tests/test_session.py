"""Bring your own corpus: build an index, open it, ask a question, get text.

The claim under test is the one Phase 2.3 exists for. Before it, `retrieve()`
needed a hundred lines of setup, `build_index` demanded a `MusiqueDataset`,
and the answer came back as node ids with the passage text living somewhere
else. Everything below runs on four ordinary documents and never mentions a
benchmark.

The dedup assertion is the load-bearing one. Duplicate suppression needs a
`DedupConfig` AND a similarity backend; callers supply the first and forget
the second, which is exactly how this project's own measurement campaign ran
with the mechanism silently off. `SpiywebIndex` wires both, and
`dedup_mode == "full"` is the receipt that says so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spiyweb import open_index
from spiyweb.config import EntityEdgeConfig, EntityExtractionConfig, SemanticEdgeConfig
from spiyweb.indexing import DocumentInput, TextUnit, build_index, read_manifest

if TYPE_CHECKING:
    from pathlib import Path

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

# Passage directions are orthogonal enough that the seed's cosine
# neighbourhood never crosses the document boundary on its own - beta:0 can
# only be reached through the shared rare entity.
PASSAGES = {
    "The tower was raised on the shore.": [1.0, 0.0, 0.0],
    "Wardenclyffe drew power from the ground.": [0.8, 0.6, 0.0],
    "What happened at Wardenclyffe afterwards.": [0.0, 1.0, 0.0],
    "An unrelated closing paragraph.": [0.0, 0.0, 1.0],
}
QUERIES = {
    "who raised the tower": [1.0, 0.0, 0.0],
    "what happened afterwards": [0.0, 1.0, 0.0],
}


class FakeEmbedder:
    model_name = "fake-e5"

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [PASSAGES[text] for text in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [QUERIES[text] for text in texts]


class OtherEmbedder(FakeEmbedder):
    model_name = "some-other-model"


class KeywordPipeline:
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


def _build(root: Path) -> None:
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


def test_a_corpus_needs_no_benchmark_type_to_be_indexed(tmp_path: Path) -> None:
    """`build_index` takes documents and a directory. That is the whole API."""
    manifest = build_index(
        DOCUMENTS,
        tmp_path,
        embedder=FakeEmbedder(),
        entity_pipeline=KeywordPipeline(),
        embedding_model="fake-e5",
        extraction_config=EntityExtractionConfig(min_entities=0),
        semantic_config=SemanticEdgeConfig(k=2, min_similarity=0.0),
        entity_config=EntityEdgeConfig(max_df_ratio=1.0),
        log=lambda message: None,
    )
    assert manifest.chunks == 4
    assert manifest.propositions == 0
    assert manifest.dimension == 3
    assert manifest.embedding_model == "fake-e5"
    assert manifest.edges["entity"] == 1, "the cross-document hop must exist"
    assert manifest.edges["structural"] == 2, (
        "two-unit documents give the structural layer something to do - the "
        "benchmark corpora never did"
    )
    assert read_manifest(tmp_path) == manifest


def test_the_answer_carries_the_passage_text(tmp_path: Path) -> None:
    """The gap that made results unusable outside this repository."""
    _build(tmp_path)
    with open_index(tmp_path, embedder=FakeEmbedder()) as index:
        answer = index.retrieve("who raised the tower")

    assert answer.passages, "the query touches the index; something must light up"
    first = answer.passages[0]
    assert first.node_id == "alpha:0"
    assert first.text == "The tower was raised on the shore."
    assert first.source_id == "alpha"
    assert first.layer == "chunk"
    assert first.hop == 0, "a first-contact atom sits at hop 0"
    assert all(passage.text for passage in answer.passages), (
        "every activated node must resolve to the text it was indexed as"
    )


def test_the_entity_hop_still_carries_the_answer_across_documents(
    tmp_path: Path,
) -> None:
    """The scenario the project exists for, through the public API this time."""
    _build(tmp_path)
    with open_index(tmp_path, embedder=FakeEmbedder()) as index:
        answer = index.retrieve("who raised the tower")

    reached = {passage.node_id for passage in answer.passages}
    assert "beta:0" in reached, (
        "beta:0 is orthogonal to the query and reachable only through the "
        "rare entity it shares with the seed's neighbour"
    )
    assert answer.confidence.hop_depth >= 2


def test_opening_an_index_wires_both_halves_of_dedup(tmp_path: Path) -> None:
    """The trap, closed by construction rather than by documentation."""
    _build(tmp_path)
    with open_index(tmp_path, embedder=FakeEmbedder()) as index:
        answer = index.retrieve("who raised the tower")

    assert answer.dedup_mode == "full", (
        "a caller who opens an index should never have to remember to build "
        "a similarity backend as well"
    )
    # alpha:0 and alpha:1 are two passages of ONE document, so the source
    # rule keeps one seed slot for the document and votes the other.
    assert answer.votes()["alpha"] == 2
    assert answer.result.contact_suppressed == {"alpha:1": "alpha:0"}
    assert answer.passages[0].votes == 2


def test_a_query_embedded_by_another_model_is_refused(tmp_path: Path) -> None:
    """Dimensions can match across unrelated models; cosine cannot tell."""
    _build(tmp_path)
    with open_index(tmp_path, embedder=OtherEmbedder()) as index:
        with pytest.raises(ValueError, match="confident nonsense"):
            index.retrieve("who raised the tower")


def test_the_honesty_outputs_are_reachable_from_the_answer(tmp_path: Path) -> None:
    """Paths, clusters, gaps and the refusal report - D17/D18/D35, on demand."""
    _build(tmp_path)
    with open_index(tmp_path, embedder=FakeEmbedder()) as index:
        answer = index.retrieve("who raised the tower")

    assert answer.paths(), "every activated node has a path back to a seed"
    assert answer.clusters(), "the activated subgraph has at least one theme"
    assert isinstance(answer.gaps(), tuple)
    report = answer.refusal()
    assert report.text, "the refusal report is template-built, never empty"


def test_two_colours_meeting_on_one_node_make_a_bridge(tmp_path: Path) -> None:
    _build(tmp_path)
    with open_index(tmp_path, embedder=FakeEmbedder()) as index:
        answer = index.retrieve_colored(
            {"c0": "who raised the tower", "c1": "what happened afterwards"}
        )

    assert answer.passages
    assert answer.dedup_mode == "full"
    assert set(answer.colored.per_color) == {"c0", "c1"}


def test_an_unknown_profile_names_the_ones_that_exist(tmp_path: Path) -> None:
    _build(tmp_path)
    with open_index(tmp_path, embedder=FakeEmbedder()) as index:
        with pytest.raises(ValueError, match="compare"):
            index.retrieve("who raised the tower", profile="fast")


def test_a_profile_overlays_only_its_three_knobs(tmp_path: Path) -> None:
    """`explore` widens the ball; the rest of the config survives untouched."""
    _build(tmp_path)
    with open_index(tmp_path, embedder=FakeEmbedder()) as index:
        wide = index.retrieve("who raised the tower", profile="explore")
        narrow = index.retrieve("who raised the tower", profile="precise")

    assert wide.confidence.total_energy >= narrow.confidence.total_energy, (
        "a higher damping forwards more energy onward, so more of it survives"
    )


def test_no_preference_means_the_default_profile(tmp_path: Path) -> None:
    """The 0.1.2 trap, closed in the library and not just the terminal.

    `RetrievalConfig()` cannot spread past five seeds, so a caller who names
    no profile and builds no config must not silently get top-k. The answer
    and the trace both say which profile ran.
    """
    from spiyweb.profiles import DEFAULT_PROFILE, PROFILES

    _build(tmp_path)
    with open_index(tmp_path, embedder=FakeEmbedder()) as index:
        answer = index.retrieve("who raised the tower")

    assert answer.profile == DEFAULT_PROFILE
    assert answer.trace is not None and answer.trace.profile == DEFAULT_PROFILE
    assert answer.config is not None
    expected = PROFILES[DEFAULT_PROFILE]
    assert answer.config.propagation.damping == expected.damping
    assert answer.config.propagation.threshold_ratio == expected.threshold_ratio
    assert answer.config.seed_width == expected.seed_width


def test_a_config_given_at_the_call_is_never_overlaid(tmp_path: Path) -> None:
    from spiyweb.config import RetrievalConfig

    _build(tmp_path)
    mine = RetrievalConfig(seed_width=3)
    with open_index(tmp_path, embedder=FakeEmbedder()) as index:
        answer = index.retrieve("who raised the tower", config=mine)

    assert answer.profile == ""
    assert answer.config == mine, "a config the caller built is theirs as built"
    assert answer.trace is not None and answer.trace.profile == ""


def test_a_config_given_at_open_counts_as_a_preference(tmp_path: Path) -> None:
    from spiyweb.config import RetrievalConfig

    _build(tmp_path)
    mine = RetrievalConfig(seed_width=3)
    with open_index(tmp_path, embedder=FakeEmbedder(), config=mine) as index:
        answer = index.retrieve("who raised the tower")

    assert answer.profile == ""
    assert answer.config == mine


def test_a_named_profile_wins_over_any_config(tmp_path: Path) -> None:
    from spiyweb.config import RetrievalConfig
    from spiyweb.profiles import PRECISE

    _build(tmp_path)
    mine = RetrievalConfig(seed_width=3, contact_overfetch=7)
    with open_index(tmp_path, embedder=FakeEmbedder(), config=mine) as index:
        answer = index.retrieve("who raised the tower", profile="precise")

    assert answer.profile == "precise"
    assert answer.config is not None
    assert answer.config.propagation.damping == PRECISE.damping
    assert answer.config.contact_overfetch == 7, "only the three knobs move"
