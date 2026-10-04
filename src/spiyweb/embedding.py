"""Embedding wrapper: e5 prefixes baked in, device order CUDA -> MPS -> CPU.

The e5 model family requires a role prefix on every input - `"query: "` for
questions, `"passage: "` for corpus text - and silently degrades without it.
The prefix contract is therefore baked into the API: there is no un-prefixed
encode method, so callers cannot forget it. The prefixes themselves come from
`EmbeddingConfig` (e5's by default), because other model families use other
ones.

Vectors are always L2-normalised, which is what makes the store's inner
product equal cosine similarity - that contract lives here, not in the store.

This module imports cleanly with nothing installed: the pure `resolve_device`
and the Protocols need no torch. Only constructing a real
`SentenceTransformerEmbedder` without an injected model touches the optional
`embed` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from spiyweb.config import EmbeddingConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

Device = Literal["cuda", "mps", "cpu"]


def resolve_device(*, cuda_available: bool, mps_available: bool) -> Device:
    """Pure device policy: CUDA -> MPS -> CPU, in that fixed order."""
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def detect_device() -> Device:
    """Probe torch for the available backends and apply `resolve_device`."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for device detection; "
            "install it with `pip install spiyweb[embed]`"
        ) from error
    return resolve_device(
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
    )


class EncoderLike(Protocol):
    """The single sentence-transformers method the wrapper depends on."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Sequence[Sequence[float]]: ...


class Embedder(Protocol):
    """Role-aware embedding interface the index pipeline depends on."""

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


def _require_sentence_transformers() -> type:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for embedding; "
            "install it with `pip install spiyweb[embed]`"
        ) from error
    return SentenceTransformer


class SentenceTransformerEmbedder:
    """e5-style embedder over any `EncoderLike` (a real model or a test fake).

    Without an injected model the sentence-transformers dependency is
    checked at construction but the weights load on the first embed call, on
    the device resolved by `detect_device` unless the config names one
    explicitly. Construction must not take the GPU: `build_index` runs its
    LLM stages (proposition extraction) before the embed stage, and on an
    8 GB card the idle weights beside the LLM pushed VRAM to ~94%.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        model: EncoderLike | None = None,
    ) -> None:
        self._config = config if config is not None else EmbeddingConfig()
        self._model: EncoderLike | None = model
        if model is None:
            # Fail before hours of LLM calls, not after them.
            _require_sentence_transformers()

    @property
    def model_name(self) -> str:
        """Which model produced these vectors - receipt data, not behaviour.

        `build_index` records it in the store so a query embedded by a
        different model can be refused instead of silently answered: two
        unrelated models can share a dimension, and cosine across two spaces
        returns confident nonsense. The role prefixes are part of the
        identity for the same reason - one model under two prompt formats
        is two vector spaces - so a non-default pair is appended to the
        name. The e5 defaults are not, which keeps every existing index's
        recorded name valid.
        """
        defaults = EmbeddingConfig()
        query, passage = self._config.query_prefix, self._config.passage_prefix
        if (query, passage) == (defaults.query_prefix, defaults.passage_prefix):
            return self._config.model
        return f"{self._config.model} [query={query!r} passage={passage!r}]"

    def _load_model(self) -> EncoderLike:
        SentenceTransformer = _require_sentence_transformers()
        device = self._config.device
        if device is None:
            device = detect_device()
        return SentenceTransformer(self._config.model, device=device)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed question-side texts with the configured query prefix."""
        prefix = self._config.query_prefix
        return self._encode([prefix + text for text in texts])

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus-side texts with the configured passage prefix."""
        prefix = self._config.passage_prefix
        return self._encode([prefix + text for text in texts])

    def _encode(self, prefixed: list[str]) -> list[list[float]]:
        if self._model is None:
            self._model = self._load_model()
        rows = self._model.encode(
            prefixed,
            batch_size=self._config.batch_size,
            normalize_embeddings=True,
        )
        return [[float(value) for value in row] for row in rows]
