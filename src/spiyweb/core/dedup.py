"""Dynamic redundancy suppression: near-duplicate -> edge zeroed, idea voted.

Duplicate detection is dynamic and query-scoped: it runs among the currently
active nodes, at the moment a candidate neighbour is about to receive energy.
The threshold is not a constant but is computed per hop from the active set's
own similarity distribution - what counts as "the same thing said again"
depends on how tight the activated region already is.

Like the rest of `core/`, this module computes nothing itself: the caller
supplies a similarity function (in practice cosine over the stored embeddings)
and the module only decides. No vector store, no I/O, no model.
"""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Protocol

from spiyweb.config import DedupConfig

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence


class SimilarityFn(Protocol):
    """Batch similarity of one node against many, supplied by the caller.

    Scores are expected to be cosine similarities in `[-1, 1]`. The batch
    shape exists so a numpy-backed caller can answer one call with one
    matrix-vector product instead of len(others) scalar lookups.
    """

    def __call__(self, node: str, others: Sequence[str]) -> Sequence[float]: ...


def adaptive_threshold(
    active: Sequence[str],
    similarity: SimilarityFn,
    config: DedupConfig,
) -> float:
    """Duplicate cut for this hop, from the active set's similarity spread.

    `tau = max(floor, mean + sigma * std)` over all pairwise similarities of
    `active`. With fewer than `min_pairs` observed pairs the distribution is
    noise and `floor` is returned unchanged. The returned value is recorded in
    the propagation result - the design requires the computed cut to be
    visible, never a hidden internal.
    """
    values: list[float] = []
    for position, node in enumerate(active[:-1]):
        values.extend(float(s) for s in similarity(node, active[position + 1 :]))
    if len(values) < config.min_pairs:
        return config.floor
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return max(config.floor, mean + config.sigma * sqrt(variance))


def find_survivor(
    candidate: str,
    active: Sequence[str],
    similarity: SimilarityFn,
    threshold: float,
) -> str | None:
    """The active node `candidate` duplicates, or `None` if it duplicates none.

    Among active nodes at or above `threshold`, the most similar one wins the
    vote; ties break on node id so the outcome is stable across platforms.
    """
    if not active:
        return None
    scores = similarity(candidate, active)
    best: tuple[float, str] | None = None
    for node, score in zip(active, scores, strict=True):
        value = float(score)
        if value < threshold:
            continue
        if best is None or value > best[0] or (value == best[0] and node < best[1]):
            best = (value, node)
    return best[1] if best is not None else None


def record_support(
    supporters: dict[str, set[str]],
    candidate: str,
    survivor: str,
    source_of: Mapping[str, str] | None,
) -> None:
    """Note that `candidate`'s source supports the idea `survivor` carries.

    Votes count corpus support per SOURCE: an idea's vote is one plus the
    number of OTHER sources found restating it, however many copies each
    holds and at however many stages they were suppressed. A copy from the
    survivor's own source is suppressed like any other - edge zeroed, share
    redistributed - and supports nothing: a document restating itself is not
    a second document agreeing. Both cases are the norm on a two-layer index,
    where a passage's propositions restate it and each other. Without a
    source mapping every node stands for itself, so every copy is a source.
    """
    if source_of is None:
        key, supporter = survivor, candidate
    else:
        key = source_of.get(survivor, survivor)
        supporter = source_of.get(candidate, candidate)
    if supporter != key:
        supporters.setdefault(key, set()).add(supporter)


def merge_support(
    *ledgers: Mapping[str, Collection[str]],
) -> dict[str, frozenset[str]]:
    """Union supporter sets across stages or colours, key by key.

    The union, never a sum: one source found at contact selection and again
    in the web is still one source.
    """
    merged: dict[str, set[str]] = {}
    for ledger in ledgers:
        for key, sources in ledger.items():
            merged.setdefault(key, set()).update(sources)
    return {key: frozenset(sources) for key, sources in merged.items() if sources}


def votes_of(supporters: Mapping[str, Collection[str]]) -> dict[str, int]:
    """Votes from a support ledger: the idea itself plus each other source.

    Ideas nobody supported carry an implicit 1 and are not listed.
    """
    return {key: 1 + len(sources) for key, sources in supporters.items() if sources}
