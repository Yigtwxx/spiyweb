"""Seed injection glue: query embedding -> first contact -> propagation.

This is the end-to-end entry point the spec's §4.2 describes: cosine against
the index picks the first-contact atoms (the seed width), the seed energy is
split among them proportionally to similarity, and the web spreads from there.
The module imports only the stdlib, `core` and `config` - the vector store
satisfies `SeedSource` structurally, so `import spiyweb.retrieve` never needs
FAISS.

The §2.5 output contract is now covered two ways: fields on the result for
what the run itself produced (votes, conflicts, absorptions, confidence), and
ON-DEMAND functions in `spiyweb.output` for the derived honesty outputs
(activation paths, theme clusters, gap warnings, the D35 refusal report) -
derived views take the graph as input and are the caller's to invoke, which
is also how D17 keeps the "when is this weak?" policy out of the library.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from spiyweb.config import (
    ColoredRetrievalConfig,
    ConflictConfig,
    DedupConfig,
    NegativeSeedConfig,
    PolarityConfig,
    PropagationConfig,
    RetrievalConfig,
)
from spiyweb.core.colors import propagate_colored
from spiyweb.core.dedup import adaptive_threshold, find_survivor
from spiyweb.core.negative import negative_field
from spiyweb.core.propagate import propagate

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from spiyweb.core.colors import ColoredResult
    from spiyweb.core.conflict import ConflictRecord
    from spiyweb.core.dedup import SimilarityFn
    from spiyweb.core.graph import Graph
    from spiyweb.core.negative import AbsorptionRecord
    from spiyweb.core.polarity import DisputeRecord
    from spiyweb.core.propagate import PropagationResult

    SourceKeyFn = Callable[[str], str]
    """Node id -> the source it belongs to; built by `_source_key`."""


class SeedSource(Protocol):
    """Anything that can rank contact points for a query embedding.

    `VectorStore.search` satisfies this structurally; tests use a dict-backed
    fake. Scores are expected to be cosine similarities (the store guarantees
    this for L2-normalised inputs, which the embedder always produces).
    """

    def search(self, query: Sequence[float], k: int) -> list[tuple[str, float]]: ...


@dataclass(frozen=True)
class Confidence:
    """The confidence triple: signal for the caller, never a policy.

    The library reports how much of the web lit up; what counts as
    "I don't know" is the caller's decision (D17).

    Attributes:
        total_energy: Sum of accumulated energy over all activated nodes.
        node_count: Number of activated nodes.
        hop_depth: Deepest hop the activation reached.
    """

    total_energy: float
    node_count: int
    hop_depth: int


@dataclass(frozen=True)
class RetrievalResult:
    """What one `retrieve()` call returns - deliberately partial, see module
    docstring.

    Attributes:
        seeds: First-contact atoms and their cosine similarities - exactly the
            weights the proportional energy split used.
        propagation: The full propagation outcome (activations with hop and
            contributor data, stop reason, threshold). Activation paths are
            derivable by walking `contributors` backwards; a dedicated path
            object arrives with the output-contract work.
    """

    seeds: Mapping[str, float]
    propagation: PropagationResult
    contact_suppressed: Mapping[str, str] = field(default_factory=dict)
    """Contact-stage duplicate -> the selected contact it duplicated (the
    elastic refill's ledger; empty while dedup is off)."""
    contact_votes: Mapping[str, int] = field(default_factory=dict)
    """Votes earned at contact selection, same keying as propagation votes."""
    contact_tau: float | None = None
    """The adaptive cut used at contact selection - visible, as required."""
    dedup_mode: str = "off"
    """Which duplicate rules actually RAN: "off", "sources_only",
    "cosine_only" or "full". A `DedupConfig` alone does not switch the
    mechanism on - the cosine twin test also needs a `similarity` backend -
    and the 2026-08 campaign measured tour after tour believing otherwise.
    The ledger now states what ran instead of leaving it to be inferred.

    This names CAPABILITY, not effect: the source rule is a no-op on a
    single-layer index, so "sources_only" can still suppress nothing.
    What actually happened is in `contact_suppressed` and the
    propagation ledger."""

    def ranked(self) -> list[tuple[str, float]]:
        """Activated nodes with accumulated energy, strongest first."""
        return self.propagation.ranked()

    def votes(self) -> dict[str, int]:
        """Corpus support per idea, merged across BOTH suppression stages.

        Contact-stage and propagation-stage votes each carry an implicit
        baseline of 1, so the merge adds only the increments.
        """
        merged = dict(self.propagation.votes)
        for key, count in self.contact_votes.items():
            merged[key] = merged.get(key, 1) + (count - 1)
        return merged

    @property
    def confidence(self) -> Confidence:
        """Aggregate activation signal; the caller decides what it means."""
        activations = self.propagation.activations
        return Confidence(
            total_energy=sum(a.energy for a in activations.values()),
            node_count=len(activations),
            hop_depth=self.propagation.hops_used,
        )

    @property
    def conflicts(self) -> tuple[ConflictRecord, ...]:
        """Neutralisation events of the run - the destroyed-energy ledger."""
        return self.propagation.conflicts

    @property
    def disputed(self) -> frozenset[str]:
        """Nodes that survived a conflict with energy left (D16).

        Both sides of an unresolved contradiction enter the context flagged
        as disputed; a side neutralised to zero is gone from the ranking and
        carries nothing to flag.
        """
        return frozenset(
            node
            for record in self.propagation.conflicts
            for node in (record.node_a, record.node_b)
            if self.propagation.energy_of(node) > 0.0
        )

    @property
    def absorptions(self) -> tuple[AbsorptionRecord, ...]:
        """Negative-seed absorption events - the excluded region's ledger."""
        return self.propagation.absorptions

    @property
    def disputes(self) -> tuple[DisputeRecord, ...]:
        """Negative-polarity absorptions (D34) - the corpus-dispute ledger."""
        return self.propagation.disputes


def retrieve(
    query_embedding: Sequence[float],
    index: SeedSource,
    graph: Graph,
    config: RetrievalConfig | None = None,
    *,
    similarity: SimilarityFn | None = None,
    dedup: DedupConfig | None = None,
    source_of: Mapping[str, str] | None = None,
    negative: Mapping[str, Mapping[str, float]] | None = None,
    conflict: ConflictConfig | None = None,
    negative_queries: Sequence[Sequence[float]] | None = None,
    negative_seed: NegativeSeedConfig | None = None,
    residue: Mapping[str, float] | None = None,
    polarity: PolarityConfig | None = None,
) -> RetrievalResult:
    """Inject a query into the web and return the activated result.

    Contacts with similarity <= 0.0 are dropped before injection: a
    non-positive weight would poison the proportional seed split, and a
    non-positive cosine is no evidence of contact in the first place. If no
    contact survives, that is a hard error - retrieval over an index that
    cannot even touch the query must not silently return an empty web.

    `negative_queries` are the embeddings of the EXCLUDED phrases
    ("excluding X"): each finds its own contacts, their merged field spreads
    with the ordinary rules, and positive energy activating inside it is
    destroyed (see `NegativeSeedConfig`). An exclusion that touches nothing
    absorbs nothing - that is a legitimate outcome, not an error, unlike the
    positive query's zero-contact case above.

    Seed ids absent from the graph's adjacency are allowed: an isolated chunk
    legitimately receives and holds its share of the energy.

    This is the config-in primitive: `config=None` means the bare
    `RetrievalConfig()`, which carries the canonical trace and cannot spread
    past five seeds - it warns when that happens rather than picking a
    profile for you. The default profile lives one level up, in
    `SpiywebIndex.retrieve()` and `ThermalSession`.

    `residue` is the thermal conversation memory (D22/D32): leftover energy
    from the previous turn, injected on top of the seed split so a follow-up
    lands on warm ground - `ThermalSession` builds and manages it across
    turns. `polarity` enables the negative-knowledge atoms (D34); both pass
    straight through to `propagate`.
    """
    cfg = config if config is not None else RetrievalConfig()
    _check_dedup_contract(similarity, dedup)
    contacts = index.search(
        query_embedding,
        _contact_depth(cfg.seed_width, cfg.contact_overfetch, similarity, dedup),
    )
    seeds, contact_suppressed, contact_tau = _select_contacts(
        contacts, cfg.seed_width, similarity, dedup, _source_key(graph, source_of)
    )
    if not seeds:
        raise ValueError(
            "no seed contact with positive similarity - the query touches "
            "nothing in the index; check that query and passages were embedded "
            "with the same model and the store is not empty"
        )
    _check_propagation_can_spread(seeds, cfg.propagation)
    absorb, negative_cfg = _absorbing_field(
        index, graph, cfg.propagation, negative_queries, negative_seed
    )
    result = propagate(
        graph,
        seeds,
        cfg.propagation,
        similarity=similarity,
        dedup=dedup,
        source_of=source_of,
        negative=negative,
        conflict=conflict,
        absorb=absorb,
        negative_seed=negative_cfg,
        residue=residue,
        polarity=polarity,
    )
    return RetrievalResult(
        seeds=seeds,
        propagation=result,
        contact_suppressed=contact_suppressed,
        contact_votes=_contact_votes(contact_suppressed, source_of),
        contact_tau=contact_tau,
        dedup_mode=_dedup_mode(similarity, dedup),
    )


def _check_propagation_can_spread(
    seeds: Mapping[str, float], config: PropagationConfig
) -> None:
    """Warn when a second hop is arithmetically impossible for THIS query.

    The seed energy is split among the contacts in proportion to similarity,
    so the strongest seed holds `seed_energy * s_max / sum(s)` and forwards
    `damping` of it. That product is the most any hop-1 arrival can be, BEFORE
    it is divided among neighbours. Under the threshold, nothing can ever
    clear it and the web stops at first contact - `top-k` with extra steps,
    the one thing this library exists to not be.

    Computed from the seeds actually selected rather than from `seed_width`:
    a query that touched two passages splits the energy two ways whatever the
    configured width says, and warning on the configured number would cry
    wolf on every small index.

    A real trap, not a hypothetical. CLAUDE.md sets the threshold at 15% of
    the injected energy (§2.1, sized for the TWO-seed worked example) and the
    seed width at 5 (§4); together they cannot propagate, so the shipped
    defaults returned hop-0-only webs on a real index until this was measured
    on 2026-08-26. Every named profile clears the bar comfortably.

    A warning and not an error: first contact only is a legitimate thing to
    want, and refusing would break a caller who wants it. But it must not be
    SILENT, for the same reason the dedup trap now raises - a mechanism that
    is off while the caller believes it is on is how this project lost a
    measurement campaign.
    """
    total = sum(seeds.values())
    if total <= 0.0:
        return
    strongest = config.seed_energy * max(seeds.values()) / total
    ceiling = strongest * config.damping
    if ceiling >= config.threshold:
        return
    warnings.warn(
        f"these settings cannot spread past the seed: {len(seeds)} contact(s) "
        f"share {config.seed_energy} energy, so a hop-1 arrival is at most "
        f"{ceiling:.2f} against a threshold of {config.threshold:.2f}. The "
        "result will be first contact only - top-k with extra steps. Pass a "
        "profile (precise / explore / compare), or lower threshold_ratio.",
        UserWarning,
        stacklevel=3,
    )


def _dedup_mode(similarity: SimilarityFn | None, dedup: DedupConfig | None) -> str:
    """Name which duplicate rules the call can actually run.

    Two rules suppress a duplicate and they need different things: the cosine
    twin test needs a `similarity` backend, the SOURCE test needs none. A
    config alone therefore says what the caller WANTED, not what happened,
    and the gap between the two is exactly how the 2026-08 campaign measured
    with the mechanism off. The result reports the answer rather than leaving
    it to be inferred from an empty ledger.
    """
    if dedup is None or not dedup.enabled:
        return "off"
    by_cosine = similarity is not None
    by_source = dedup.distinct_sources
    if by_cosine and by_source:
        return "full"
    if by_cosine:
        return "cosine_only"
    if by_source:
        return "sources_only"
    # Unreachable through `retrieve`/`retrieve_colored`: this is exactly
    # the combination `_check_dedup_contract` refuses upstream. Left as a
    # total function rather than a raise - naming a mode is not the
    # place to enforce one.
    return "off"


def _check_dedup_contract(
    similarity: SimilarityFn | None, dedup: DedupConfig | None
) -> None:
    """Refuse - or flag - a dedup request the call cannot honour.

    With an enabled config and NEITHER rule live, `dedup=` is a no-op the
    caller believes in. That is a hard error: silently returning an
    unsuppressed web under a config that asks for suppression is the failure
    this project already paid for once.

    With the source rule alone the call is legitimate - it is a measured,
    embedding-free mechanism - but half the config is dead: no cosine cut at
    contact, no neighbour-level suppression inside propagation, no
    propagation votes, no adaptive threshold. That warns rather than raises,
    because ablating one rule is a supported thing to do.
    """
    if dedup is None or not dedup.enabled or similarity is not None:
        return
    if not dedup.distinct_sources:
        raise ValueError(
            "dedup is enabled but nothing can run it: the cosine twin test "
            "needs `similarity=`, and `distinct_sources=False` switched off "
            "the only rule that works without one. Pass a SimilarityFn, or "
            "pass `dedup=None` to retrieve without suppression."
        )
    warnings.warn(
        "dedup is enabled without `similarity=`: only the distinct-source "
        "seed rule can run. The cosine twin test, neighbour-level "
        "suppression and the adaptive threshold all stay off - the result "
        'reports `dedup_mode="sources_only"`. Pass a SimilarityFn to enable '
        "them.",
        UserWarning,
        stacklevel=3,
    )


def _contact_depth(
    width: int,
    overfetch: int,
    similarity: SimilarityFn | None,
    dedup: DedupConfig | None,
) -> int:
    """How deep to search: overfetched only when the refill can actually run.

    Two refill rules can trigger it - the cosine twin test needs a similarity
    backend, the source test does not - so the depth follows whichever is
    live. Overfetching with neither would just cost index reads.
    """
    if dedup is None or not dedup.enabled:
        return width
    if similarity is None and not dedup.distinct_sources:
        return width
    return width * overfetch


def _source_key(graph: Graph, source_of: Mapping[str, str] | None) -> SourceKeyFn:
    """Node -> the source it belongs to; caller's map wins, then the graph.

    A proposition inherits its parent passage's `source_id`, so reading the
    registry gives "same passage" without teaching retrieval the `{chunk}#p{n}`
    id convention. A node the graph does not know stands for itself.
    """

    def key(node: str) -> str:
        if source_of is not None and node in source_of:
            return source_of[node]
        data = graph.node(node)
        return data.source_id if data is not None else node

    return key


def _select_contacts(
    contacts: Sequence[tuple[str, float]],
    width: int,
    similarity: SimilarityFn | None,
    dedup: DedupConfig | None,
    source_key: SourceKeyFn | None = None,
) -> tuple[dict[str, float], dict[str, str], float | None]:
    """Fill the seed slots with the first DISTINCT positive contacts.

    The elastic refill (2026-08-14 A1 decision): a contact that duplicates an
    already selected one is skipped - it becomes a vote, and its slot goes to
    the next distinct contact, so a duplicated corpus cannot halve the number
    of ideas the web starts from. With dedup off (or with neither refill rule
    live) this reduces exactly to "top `width` positive contacts". The adaptive
    cut is computed over the positive candidate list and returned for the
    result ledger; a duplicate skipped after the slots are already full is not
    examined and not voted (the scan stops with the slots).

    "Distinct" is two tests, not one. The cosine test catches a copied
    passage. The SOURCE test (`DedupConfig.distinct_sources`) catches what
    cosine structurally cannot: two propositions of the same passage are
    different sentences, so they are never near-duplicates, yet a query part
    that seeds both explores one passage instead of two. Both tests end in the
    same place - skipped, voted, slot refilled - because they are the same
    rule about the same damage.
    """
    positive = [(node, score) for node, score in contacts if score > 0.0]
    if dedup is None or not dedup.enabled:
        return dict(positive[:width]), {}, None
    by_cosine = similarity is not None
    by_source = dedup.distinct_sources and source_key is not None
    if not by_cosine and not by_source:
        return dict(positive[:width]), {}, None

    tau: float | None = None
    if by_cosine:
        assert similarity is not None
        tau = adaptive_threshold([node for node, _ in positive], similarity, dedup)
    kept: dict[str, float] = {}
    suppressed: dict[str, str] = {}
    holder_of: dict[str, str] = {}  # source -> the seed already holding it
    for node, score in positive:
        if len(kept) == width:
            break
        survivor: str | None = None
        if by_cosine:
            assert similarity is not None and tau is not None
            survivor = find_survivor(node, list(kept), similarity, tau)
        if survivor is None and by_source:
            assert source_key is not None
            survivor = holder_of.get(source_key(node))
        if survivor is None:
            kept[node] = score
            if by_source:
                assert source_key is not None
                holder_of[source_key(node)] = node
        else:
            suppressed[node] = survivor
    return kept, suppressed, tau


def _contact_votes(
    suppressed: Mapping[str, str],
    source_of: Mapping[str, str] | None,
) -> dict[str, int]:
    """Vote increments earned at contact selection, keyed like core votes."""
    votes: dict[str, int] = {}
    for survivor in suppressed.values():
        key = source_of.get(survivor, survivor) if source_of else survivor
        votes[key] = votes.get(key, 1) + 1
    return votes


def _absorbing_field(
    index: SeedSource,
    graph: Graph,
    propagation: PropagationConfig,
    negative_queries: Sequence[Sequence[float]] | None,
    negative_seed: NegativeSeedConfig | None,
) -> tuple[dict[str, float] | None, NegativeSeedConfig | None]:
    """Contact and spread the negative queries; `(None, None)` when absent.

    Contacts merge with MAX per node across the negative queries - two
    exclusions touching the same atom are corroboration, not accumulation,
    the same rule `conflict_adjacency` applies to repeated negative edges.
    """
    if not negative_queries:
        return None, None
    cfg = negative_seed if negative_seed is not None else NegativeSeedConfig()
    merged: dict[str, float] = {}
    for embedding in negative_queries:
        for node, score in index.search(embedding, cfg.seed_width):
            if score > 0.0:
                merged[node] = max(merged.get(node, 0.0), score)
    if not merged:
        return None, cfg
    return negative_field(graph, merged, propagation, cfg.energy_ratio), cfg


@dataclass(frozen=True)
class ColoredRetrievalResult:
    """What one `retrieve_colored()` call returns - as partial as
    `RetrievalResult`, and for the same reason (see module docstring).

    Attributes:
        seeds_by_color: The contact atoms each colour actually injected -
            colours the index could not touch are absent (see
            `retrieve_colored` for the exact rule).
        colored: The per-colour propagation outcomes and the bridge nodes
            (nodes reached by two or more colours).
    """

    seeds_by_color: Mapping[str, Mapping[str, float]]
    colored: ColoredResult
    contact_suppressed: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    """Per colour: contact-stage duplicate -> the selected contact it
    duplicated (the elastic refill's ledger; empty while dedup is off)."""
    contact_votes: Mapping[str, int] = field(default_factory=dict)
    """Votes earned at contact selection across all colours."""
    contact_taus: Mapping[str, float] = field(default_factory=dict)
    """Per colour: the adaptive cut used at contact selection."""
    dedup_mode: str = "off"
    """Which duplicate rules actually RAN; see `RetrievalResult`."""

    def ranked(self) -> list[tuple[str, float]]:
        """Activated nodes by energy summed across colours, strongest first."""
        return self.colored.ranked()

    def votes(self) -> dict[str, int]:
        """Corpus support per idea, merged across BOTH suppression stages."""
        merged = dict(self.colored.votes())
        for key, count in self.contact_votes.items():
            merged[key] = merged.get(key, 1) + (count - 1)
        return merged

    @property
    def bridges(self) -> Mapping[str, tuple[str, ...]]:
        """Bridge nodes mapped to the colours that met there."""
        return self.colored.bridges

    @property
    def confidence(self) -> Confidence:
        """Aggregate activation signal across all colours; caller's policy."""
        results = self.colored.per_color.values()
        return Confidence(
            total_energy=sum(
                activation.energy
                for result in results
                for activation in result.activations.values()
            ),
            node_count=len({node for result in results for node in result.activations}),
            hop_depth=max(result.hops_used for result in results),
        )

    @property
    def conflicts(self) -> tuple[ConflictRecord, ...]:
        """Neutralisation events across all colours, in colour order."""
        return tuple(
            record
            for _, result in sorted(self.colored.per_color.items())
            for record in result.conflicts
        )

    @property
    def disputed(self) -> frozenset[str]:
        """Nodes that survived a conflict in ANY colour with energy left."""
        return frozenset(
            node
            for _, result in sorted(self.colored.per_color.items())
            for record in result.conflicts
            for node in (record.node_a, record.node_b)
            if result.energy_of(node) > 0.0
        )

    @property
    def absorptions(self) -> tuple[AbsorptionRecord, ...]:
        """Negative-seed absorption events across all colours, colour order."""
        return tuple(
            record
            for _, result in sorted(self.colored.per_color.items())
            for record in result.absorptions
        )

    @property
    def disputes(self) -> tuple[DisputeRecord, ...]:
        """Negative-polarity absorptions (D34) across colours, colour order."""
        return tuple(
            record
            for _, result in sorted(self.colored.per_color.items())
            for record in result.disputes
        )


def retrieve_colored(
    colored_queries: Mapping[str, Sequence[float]],
    index: SeedSource,
    graph: Graph,
    config: ColoredRetrievalConfig | None = None,
    *,
    similarity: SimilarityFn | None = None,
    dedup: DedupConfig | None = None,
    source_of: Mapping[str, str] | None = None,
    negative: Mapping[str, Mapping[str, float]] | None = None,
    conflict: ConflictConfig | None = None,
    negative_queries: Sequence[Sequence[float]] | None = None,
    negative_seed: NegativeSeedConfig | None = None,
    polarity: PolarityConfig | None = None,
) -> ColoredRetrievalResult:
    """Inject one coloured seed set per query part and return the merged web.

    `colored_queries` maps colour labels to query-part embeddings; insertion
    order matters, because the FIRST colour is the primary part of the query.
    Per colour, the top `seed_width` contacts are taken and non-positive
    similarities are dropped, as in `retrieve()`. A LATER colour with no
    positive contact is SKIPPED - a decomposed fragment the index cannot touch
    would only seed noise into a wrong region (the rule the 2026-08-14
    measurement campaign ran). The PRIMARY colour without a positive contact
    is the same hard error `retrieve()` raises: the question itself failing to
    touch the index must never degrade silently. (The campaign scripts carried
    a raw-contact fallback here instead; it never fired in 1000 questions, and
    it could only ever have crashed inside the core - all-non-positive seed
    weights cannot be split - so the library states the error properly.)
    """
    if not colored_queries:
        raise ValueError("at least one coloured query is required")
    cfg = config if config is not None else ColoredRetrievalConfig()
    _check_dedup_contract(similarity, dedup)

    depth = _contact_depth(cfg.seed_width, cfg.contact_overfetch, similarity, dedup)
    source_key = _source_key(graph, source_of)
    seeds_by_color: dict[str, dict[str, float]] = {}
    contact_suppressed: dict[str, dict[str, str]] = {}
    contact_taus: dict[str, float] = {}
    contact_votes: dict[str, int] = {}
    for position, (color, embedding) in enumerate(colored_queries.items()):
        contacts = index.search(embedding, depth)
        # Per COLOUR, deliberately: two colours touching one passage is a
        # legitimate overlap between sub-questions, and the measured damage
        # was inside a single colour's slots.
        seeds, suppressed_here, tau_here = _select_contacts(
            contacts, cfg.seed_width, similarity, dedup, source_key
        )
        if not seeds:
            if position == 0:
                raise ValueError(
                    "no seed contact with positive similarity for the primary "
                    "colour - the query touches nothing in the index; check "
                    "that query and passages were embedded with the same "
                    "model and the store is not empty"
                )
            continue
        seeds_by_color[color] = seeds
        if suppressed_here:
            contact_suppressed[color] = suppressed_here
            for key, count in _contact_votes(suppressed_here, source_of).items():
                contact_votes[key] = contact_votes.get(key, 1) + (count - 1)
        if tau_here is not None:
            contact_taus[color] = tau_here

    # Every colour injects the full seed energy and splits it among ITS own
    # contacts, so the check runs per colour - `seeds` here is whatever the
    # last iteration happened to leave behind, which is the wrong colour as
    # often as not and stale entirely when a colour was skipped.
    for color_seeds in seeds_by_color.values():
        _check_propagation_can_spread(color_seeds, cfg.propagation)

    # One field for the whole query: exclusion is a property of the question,
    # not of any one colour, and it scales on the UNDIVIDED energy budget.
    absorb, negative_cfg = _absorbing_field(
        index, graph, cfg.propagation, negative_queries, negative_seed
    )
    result = propagate_colored(
        graph,
        seeds_by_color,
        cfg.propagation,
        similarity=similarity,
        dedup=dedup,
        source_of=source_of,
        negative=negative,
        conflict=conflict,
        absorb=absorb,
        negative_seed=negative_cfg,
        polarity=polarity,
    )
    return ColoredRetrievalResult(
        seeds_by_color=seeds_by_color,
        colored=result,
        contact_suppressed=contact_suppressed,
        contact_votes=contact_votes,
        contact_taus=contact_taus,
        dedup_mode=_dedup_mode(similarity, dedup),
    )
