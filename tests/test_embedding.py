"""Embedding wrapper: device order, e5 prefixes, normalisation contract."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest

from spiyweb import EmbeddingConfig
from spiyweb.embedding import (
    SentenceTransformerEmbedder,
    detect_device,
    resolve_device,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeEncoder:
    """Captures every encode call; returns one fixed row per sentence."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Sequence[Sequence[float]]:
        self.calls.append(
            {
                "sentences": list(sentences),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return [[1.0, 0.0] for _ in sentences]


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_resolve_device_order_is_cuda_then_mps_then_cpu(
    cuda: bool, mps: bool, expected: str
) -> None:
    assert resolve_device(cuda_available=cuda, mps_available=mps) == expected


def test_embed_queries_prepends_the_query_prefix() -> None:
    encoder = FakeEncoder()
    embedder = SentenceTransformerEmbedder(model=encoder)
    embedder.embed_queries(["what is X"])
    assert encoder.calls[0]["sentences"] == ["query: what is X"], (
        "e5 silently degrades without the role prefix; the API bakes it in"
    )


def test_embed_passages_prepends_the_passage_prefix() -> None:
    encoder = FakeEncoder()
    embedder = SentenceTransformerEmbedder(model=encoder)
    embedder.embed_passages(["some corpus text"])
    assert encoder.calls[0]["sentences"] == ["passage: some corpus text"]


def test_prefixes_follow_the_config_for_other_model_families() -> None:
    encoder = FakeEncoder()
    config = EmbeddingConfig(
        model="some/other-model",
        query_prefix="Instruct: find\nQuery: ",
        passage_prefix="",
    )
    embedder = SentenceTransformerEmbedder(config, model=encoder)
    embedder.embed_queries(["what is X"])
    embedder.embed_passages(["some corpus text"])
    assert encoder.calls[0]["sentences"] == ["Instruct: find\nQuery: what is X"]
    assert encoder.calls[1]["sentences"] == ["some corpus text"], (
        "an empty passage prefix must send the text untouched"
    )


def test_encode_always_normalises_and_forwards_batch_size() -> None:
    encoder = FakeEncoder()
    config = EmbeddingConfig(batch_size=7)
    embedder = SentenceTransformerEmbedder(config, model=encoder)
    embedder.embed_passages(["a", "b"])
    call = encoder.calls[0]
    assert call["normalize_embeddings"] is True, (
        "L2 normalisation is what makes the store's inner product a cosine"
    )
    assert call["batch_size"] == 7


def test_rows_come_back_as_plain_float_lists() -> None:
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())
    rows = embedder.embed_queries(["a"])
    assert rows == [[pytest.approx(1.0), pytest.approx(0.0)]]
    assert isinstance(rows[0][0], float)


def test_config_rejects_empty_model_and_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="model"):
        EmbeddingConfig(model="")
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingConfig(batch_size=0)


@pytest.mark.skipif(
    find_spec("torch") is not None, reason="torch installed; hint unreachable"
)
def test_detect_device_without_torch_names_the_extra() -> None:
    with pytest.raises(ImportError, match=r"spiyweb\[embed\]"):
        detect_device()


@pytest.mark.skipif(
    find_spec("sentence_transformers") is not None,
    reason="sentence-transformers installed; hint unreachable",
)
def test_real_model_load_without_the_extra_names_it() -> None:
    with pytest.raises(ImportError, match=r"spiyweb\[embed\]"):
        SentenceTransformerEmbedder()


def test_model_name_stays_the_bare_name_under_the_e5_defaults() -> None:
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())
    assert embedder.model_name == EmbeddingConfig().model, (
        "every existing index recorded the bare name; the defaults must keep it"
    )


def test_model_name_carries_non_default_prefixes() -> None:
    plain = SentenceTransformerEmbedder(
        EmbeddingConfig(model="m", query_prefix="", passage_prefix=""),
        model=FakeEncoder(),
    )
    e5_style = SentenceTransformerEmbedder(
        EmbeddingConfig(model="m"), model=FakeEncoder()
    )
    assert plain.model_name != e5_style.model_name, (
        "one model under two prompt formats is two vector spaces - the "
        "recorded identity must tell them apart"
    )
    assert plain.model_name.startswith("m ")
