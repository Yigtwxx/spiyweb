"""Entity edges: shared entity / concept links - the main hop fuel.

Cosine neighbours are paraphrases; entity edges are what let the web hop to a
*different* document that talks about the same thing. The builder consumes a
prepared ``chunk id -> entity strings`` mapping (extraction lives in
`spiyweb.entities`, outside this package) and treats entity strings as opaque
keys - normalisation is the extractor's job, a pure function holds no
linguistic opinion.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from spiyweb.config import EntityEdgeConfig

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


_MIN_DF_CEILING = 2.0
"""Lowest document-frequency ceiling the guard may impose.

An entity in fewer than two chunks pairs with nothing, so a ceiling below 2
drops every entity that could have made an edge. Not a config field: it is
not a tuning knob but the point below which the layer stops existing.
"""


def build_entity_edges(
    entities: Mapping[str, Collection[str]],
    config: EntityEdgeConfig | None = None,
) -> list[tuple[str, str, float]]:
    """Emit raw entity edges: per shared entity a pair gains ``1 / df(entity)``.

    ``df(entity)`` is the number of chunks in `entities` mentioning it, so a
    rare shared entity binds strongly (df=2 contributes 0.5 per side of the
    pair) while a ubiquitous one barely counts. Contributions from several
    shared entities sum - converging evidence, same stance as the additive
    layer merge. A pair sharing nothing is omitted entirely - `0.0` is never
    emitted, that value belongs to the dedup-suppressed-edge contract.

    Entities mentioned by more than ``max(2, max_df_ratio * n_chunks)`` chunks
    are dropped before pairing (strict `>`, so a ratio of `1.0` truly disables
    the guard): 1/df bounds their weight but not the near-clique of edges they
    would emit. Duplicate mentions of one entity within one chunk count once.

    The floor of 2 is not a rounding nicety, it is what keeps the layer from
    being structurally impossible on a small corpus. An entity has to appear
    in at least TWO chunks to produce an edge at all, so a threshold below 2
    means the layer cannot emit a single one - and at the measured default
    ratio of 0.02 that is every corpus under 100 chunks. Found on 2026-08-26
    by running the real pipeline over three documents: the one entity that
    bridged them (`morgan`, df=2) was dropped against a threshold of 0.16,
    and the entity layer - the main hop fuel of CLAUDE.md section 2.2 - came
    out empty without saying so.

    The floor cannot move a sealed number: the smallest measured index holds
    3336 chunks, where `0.02 * n` is 66.7 and the floor never binds. It binds
    only where the alternative was an empty layer.

    Only chunks that share an entity are ever paired - the inverted index
    keeps the cost at O(mentions + sum(df^2)), never O(n^2) over the corpus.
    Iteration order is fixed (sorted entities, sorted chunk ids), so the
    accumulated float sums are bit-identical across platforms. Output is
    canonically ordered (`u < v` within each tuple) and sorted.
    """
    cfg = config if config is not None else EntityEdgeConfig()

    mentions: dict[str, set[str]] = {}
    for chunk_id, chunk_entities in entities.items():
        if not chunk_id:
            raise ValueError("chunk id must not be empty")
        for entity in chunk_entities:
            if not entity:
                raise ValueError(
                    f"chunk {chunk_id!r} carries an empty entity string; "
                    "extraction must never emit one"
                )
            mentions.setdefault(entity, set()).add(chunk_id)

    max_df = max(_MIN_DF_CEILING, cfg.max_df_ratio * len(entities))
    if cfg.max_df_cap is not None:
        max_df = min(max_df, float(cfg.max_df_cap))
    weights: dict[tuple[str, str], float] = {}
    for entity in sorted(mentions):
        chunk_ids = mentions[entity]
        df = len(chunk_ids)
        if df < 2 or df > max_df:
            continue
        contribution = 1.0 / df
        for u, v in combinations(sorted(chunk_ids), 2):
            weights[(u, v)] = weights.get((u, v), 0.0) + contribution

    return [(u, v, weight) for (u, v), weight in sorted(weights.items())]
