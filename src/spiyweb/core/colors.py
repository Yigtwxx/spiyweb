"""Coloured multi-seed activation and bridge detection (D12).

A decomposed query does not inject one ball of energy - each part becomes a
differently coloured seed set, every colour spreads as its own web, and a node
reached by two or more colours is a bridge: the place where a multi-hop answer
lives. The colours never mix during propagation; they only meet in the result.

The injected total is conserved: `seed_energy` is split EQUALLY among the
colours (there is no evidence for unequal priors between query parts, so there
is deliberately no weighting knob - query profiles may add one later). Each
colour's relative threshold scales with its own share, exactly as thermal
residue and profiles scale the plain run's threshold.

Like the rest of `core/`, this module knows nothing about queries or
embeddings: the caller decomposes the query and supplies per-colour contact
points; colours are opaque labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from spiyweb.config import (
    ConflictConfig,
    DedupConfig,
    NegativeSeedConfig,
    PolarityConfig,
    PropagationConfig,
)
from spiyweb.core.dedup import merge_support, votes_of
from spiyweb.core.propagate import PropagationResult, propagate

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spiyweb.core.dedup import SimilarityFn
    from spiyweb.core.graph import Graph


@dataclass(frozen=True)
class ColoredResult:
    """Everything a coloured run produced: per-colour webs and their meetings.

    Attributes:
        per_color: Each colour's full propagation outcome, unmerged, so every
            plain-run diagnostic (stop reason, hop depth, contributors) stays
            available per query part.
        bridges: Nodes reached by two or more colours, mapped to the sorted
            tuple of colours that reached them. Single-colour nodes are not
            bridges and do not appear.
    """

    per_color: Mapping[str, PropagationResult]
    bridges: Mapping[str, tuple[str, ...]]

    def energy_of(self, node: str) -> float:
        """Energy of `node` summed across colours (`0.0` if never reached)."""
        return sum(result.energy_of(node) for result in self.per_color.values())

    def ranked(self) -> list[tuple[str, float]]:
        """All activated nodes by summed energy, strongest first.

        Summing across colours is the additive-accumulation rule applied one
        level up: a bridge fed by two colours rises for the same reason a node
        fed by two paths does. Ties break on node id, as in the plain run.
        """
        combined: dict[str, float] = {}
        for result in self.per_color.values():
            for node, activation in result.activations.items():
                combined[node] = combined.get(node, 0.0) + activation.energy
        return sorted(combined.items(), key=lambda item: (-item[1], item[0]))

    def supporters(self) -> dict[str, frozenset[str]]:
        """Each idea's supporting sources, unioned across colours."""
        return merge_support(*(r.supporters for r in self.per_color.values()))

    def votes(self) -> dict[str, int]:
        """Corpus support per idea, merged across colours.

        The merge unions each colour's supporting sources rather than adding
        counts: one source found by two colours is still one source, and two
        different sources found one per colour are two. Ideas nobody
        supported are not listed.
        """
        return votes_of(self.supporters())


def propagate_colored(
    graph: Graph,
    colored_seeds: Mapping[str, Mapping[str, float]],
    config: PropagationConfig | None = None,
    *,
    similarity: SimilarityFn | None = None,
    dedup: DedupConfig | None = None,
    source_of: Mapping[str, str] | None = None,
    negative: Mapping[str, Mapping[str, float]] | None = None,
    conflict: ConflictConfig | None = None,
    absorb: Mapping[str, float] | None = None,
    negative_seed: NegativeSeedConfig | None = None,
    polarity: PolarityConfig | None = None,
) -> ColoredResult:
    """Spread one web per colour and report where the colours meet.

    Args:
        graph: The graph every colour spreads over.
        colored_seeds: Colour label -> that part's contact points (node ->
            contact strength). Labels are opaque; iteration is sorted for
            cross-platform determinism.
        config: Propagation settings for the whole query; each colour runs
            under an equal share of `config.seed_energy`.
        similarity: Batch node-to-nodes similarity for dynamic dedup; `None`
            disables it. Each colour's web runs its own suppression - two
            colours may legitimately disagree on what is redundant.
        dedup: Dedup settings, consulted only when `similarity` is given.
        source_of: Node id -> document/source id for vote granularity, exactly
            as in `propagate`.
        negative: Pre-marked contradiction adjacency, exactly as in
            `propagate`. Each colour's web neutralises independently - the
            colours never mix during propagation, conflicts included.
        conflict: Neutralisation settings, consulted only when `negative` is
            given.
        absorb: The negative seeds' absorbing field, exactly as in
            `propagate`; one field, applied to every colour's run.
        negative_seed: Absorption settings, consulted only when `absorb` is
            given.
        polarity: Negative-knowledge settings (D34), exactly as in
            `propagate`; the corpus's negative atoms absorb every colour's
            energy alike - the "no" does not depend on which query part
            arrived.

    Returns:
        Per-colour results, bridge nodes, and the combined ranking.

    Raises:
        ValueError: If no colour is given, or any colour has no seeds (a
            decomposed part that touches nothing must fail loudly, not vanish
            from the ranking in silence).
    """
    cfg = config if config is not None else PropagationConfig()
    if not colored_seeds:
        raise ValueError("at least one colour is required")
    for color, seeds in colored_seeds.items():
        if not seeds:
            raise ValueError(f"colour {color!r} has no seed contacts")

    share = replace(cfg, seed_energy=cfg.seed_energy / len(colored_seeds))
    per_color = {
        color: propagate(
            graph,
            colored_seeds[color],
            share,
            similarity=similarity,
            dedup=dedup,
            source_of=source_of,
            negative=negative,
            conflict=conflict,
            absorb=absorb,
            negative_seed=negative_seed,
            polarity=polarity,
        )
        for color in sorted(colored_seeds)
    }

    reached_by: dict[str, list[str]] = {}
    for color, result in per_color.items():
        for node in result.activations:
            reached_by.setdefault(node, []).append(color)
    bridges = {
        node: tuple(sorted(colors))
        for node, colors in reached_by.items()
        if len(colors) >= 2
    }
    return ColoredResult(per_color=per_color, bridges=bridges)
