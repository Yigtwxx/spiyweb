"""Thermal conversation memory (D22/D32) - the session wrapper.

A follow-up question lives in the same region of the box as the question
before it, so `ThermalSession` keeps a fraction of each turn's activated
energy warm and injects it as residue under the next turn's seeds. The
session is the STATEFUL shell around the stateless `retrieve()` - core
knows nothing about turns, it only accepts a `residue` energy map.

Reset is hybrid (D32): call `reset()` when the caller knows the topic
changed; optionally enable `ThermalConfig.auto_reset` and the session
compares the new query's index contacts against the warm node set (owner's
2026-08-14 signal choice: contact overlap - embedding-free and
language-independent) and resets itself when they no longer overlap.

Scope note: the wrapper covers the plain `retrieve()` path; the coloured
path needs its own residue-vs-colour design and is deliberately deferred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spiyweb.config import RetrievalConfig, ThermalConfig
from spiyweb.profiles import DEFAULT_PROFILE, PROFILES
from spiyweb.retrieve import RetrievalResult, SeedSource, retrieve

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from spiyweb.config import (
        ConflictConfig,
        DedupConfig,
        NegativeSeedConfig,
        PolarityConfig,
    )
    from spiyweb.core.dedup import SimilarityFn
    from spiyweb.core.graph import Graph
    from spiyweb.core.propagate import PropagationResult


class ThermalSession:
    """Multi-turn retrieval over one index and graph, with warm ground.

    Each turn injects `residue_ratio` of the previous turn's accumulated
    energies alongside the new seeds; the relative stop threshold scales
    with the larger injected total inside the core (D5/D27). A disabled
    config makes every call exactly a stateless `retrieve()`.
    """

    def __init__(
        self,
        index: SeedSource,
        graph: Graph,
        config: RetrievalConfig | None = None,
        *,
        profile: str | None = None,
        thermal: ThermalConfig | None = None,
    ) -> None:
        """`profile` overlays its three knobs onto `config`, exactly as
        `SpiywebIndex.retrieve()` does - and with the same default: neither
        given means `DEFAULT_PROFILE`, because the bare `RetrievalConfig()`
        cannot spread past five seeds. A config you did give runs as given.
        """
        self._index = index
        self._graph = graph
        if profile is None and config is None:
            profile = DEFAULT_PROFILE
        if profile is not None and profile not in PROFILES:
            raise ValueError(
                f"unknown profile {profile!r}; pick one of {sorted(PROFILES)}"
            )
        base = config if config is not None else RetrievalConfig()
        self._config = PROFILES[profile].as_retrieval(base) if profile else base
        self._profile = profile or ""
        self._thermal = thermal if thermal is not None else ThermalConfig()
        self._previous: PropagationResult | None = None

    @property
    def profile(self) -> str:
        """The profile in the running config; `""` when a raw config runs."""
        return self._profile

    @property
    def config(self) -> RetrievalConfig:
        """The configuration every turn of this session runs with."""
        return self._config

    @property
    def warm(self) -> bool:
        """Whether a previous turn's residue is currently held."""
        return self._previous is not None

    def reset(self) -> None:
        """Drop the warm ground - the caller-controlled half of D32."""
        self._previous = None

    def residue(self) -> dict[str, float]:
        """The energy map the next turn would inject; empty when cold."""
        if self._previous is None or not self._thermal.enabled:
            return {}
        ratio = self._thermal.residue_ratio
        return {
            node: activation.energy * ratio
            for node, activation in self._previous.activations.items()
            if activation.energy > 0.0
        }

    def retrieve(
        self,
        query_embedding: Sequence[float],
        *,
        similarity: SimilarityFn | None = None,
        dedup: DedupConfig | None = None,
        source_of: Mapping[str, str] | None = None,
        negative: Mapping[str, Mapping[str, float]] | None = None,
        conflict: ConflictConfig | None = None,
        negative_queries: Sequence[Sequence[float]] | None = None,
        negative_seed: NegativeSeedConfig | None = None,
        polarity: PolarityConfig | None = None,
    ) -> RetrievalResult:
        """One conversation turn: auto-reset check, residue, then retrieve."""
        if self._thermal.enabled and self._thermal.auto_reset and self.warm:
            if self._topic_changed(query_embedding):
                self.reset()
        warmth = self.residue()
        result = retrieve(
            query_embedding,
            self._index,
            self._graph,
            self._config,
            similarity=similarity,
            dedup=dedup,
            source_of=source_of,
            negative=negative,
            conflict=conflict,
            negative_queries=negative_queries,
            negative_seed=negative_seed,
            residue=warmth or None,
            polarity=polarity,
        )
        if self._thermal.enabled:
            self._previous = result.propagation
        return result

    def _topic_changed(self, query_embedding: Sequence[float]) -> bool:
        """Contact overlap against the warm set (the auto-reset signal).

        The new query's top contacts are looked up once, independently of the
        retrieval's own (dedup-aware) contact search: the signal wants the
        raw neighbourhood, not the deduplicated seed slots.
        """
        assert self._previous is not None
        contacts = self._index.search(query_embedding, self._config.seed_width)
        if not contacts:
            return True
        warm_nodes = self._previous.activations.keys()
        overlap = sum(1 for node, _ in contacts if node in warm_nodes)
        return overlap / len(contacts) < self._thermal.min_overlap
