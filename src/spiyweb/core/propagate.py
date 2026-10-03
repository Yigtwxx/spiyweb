"""Spreading activation: multiplicative decay, proportional split, accumulation.

The query is not a filter over the graph, it is an energy seed injected into it.
Each activated node forwards `damping` of its energy, split among its still
inactive neighbours in proportion to edge weight, and whatever arrives at a node
from several paths is summed. Energy that falls below the relative threshold
dies there, which is what makes the web stop on its own instead of at `top-k`.

Formally this is a truncated Personalized PageRank; practically it is a handful
of dictionary passes over a sparse graph.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from spiyweb.config import (
    ConflictConfig,
    DedupConfig,
    NegativeSeedConfig,
    PolarityConfig,
    PropagationConfig,
)
from spiyweb.core.conflict import ConflictRecord, neutralize
from spiyweb.core.dedup import (
    SimilarityFn,
    adaptive_threshold,
    find_survivor,
    merge_support,
    record_support,
    votes_of,
)
from spiyweb.core.graph import Graph
from spiyweb.core.mass import node_masses
from spiyweb.core.negative import AbsorptionRecord
from spiyweb.core.polarity import DisputeRecord

StopReason = Literal["threshold", "max_hop", "max_nodes"]


@dataclass(frozen=True)
class Activation:
    """One node that the web reached, and how it got there."""

    energy: float
    hop: int
    contributors: tuple[str, ...]
    """Nodes that fed this one. More than one means converging evidence."""


@dataclass(frozen=True)
class PropagationResult:
    """Everything a single run produced, plus why it stopped.

    The last three fields exist only when the run carried a similarity
    function and an enabled `DedupConfig`; without dedup they stay empty.

    Attributes:
        votes: Corpus support per surviving idea, keyed by the vote key the
            caller chose (`source_of`, or the surviving node id itself). A
            value of `n` means the idea plus `n - 1` OTHER sources found
            restating it - distinct sources, not suppressed copies; ideas
            nobody supported carry an implicit 1 and are not listed.
        suppressed: Suppressed duplicate node -> the active node it
            duplicated. Suppressed nodes never activate and never receive
            energy; their edge shares were renormalised over the survivors.
        dedup_thresholds: The adaptive duplicate cut computed for each stage
            that checked duplicates - the seed injection first (when
            `include_seeds` is on and more than one seed arrived), then each
            hop that distributed energy. Recorded because the design requires
            the computed value to be visible, never a hidden internal.
    """

    activations: Mapping[str, Activation]
    injected_energy: float
    threshold: float
    hops_used: int
    stop_reason: StopReason
    votes: Mapping[str, int] = field(default_factory=dict)
    supporters: Mapping[str, frozenset[str]] = field(default_factory=dict)
    """Vote key -> the OTHER sources found restating that idea; `votes` is
    one plus the size of each set. Kept because merging two runs' votes
    (stages, colours) must union these, never add counts."""
    suppressed: Mapping[str, str] = field(default_factory=dict)
    dedup_thresholds: tuple[float, ...] = ()
    conflicts: tuple[ConflictRecord, ...] = ()
    """Neutralisation events, in firing order - the destroyed-energy ledger.
    Empty unless negative edges and an enabled `ConflictConfig` were given."""
    absorptions: tuple[AbsorptionRecord, ...] = ()
    """Negative-seed absorption events, in firing order - the second arm of
    the destroyed-energy ledger. Empty unless an absorbing field and an
    enabled `NegativeSeedConfig` were given."""
    disputes: tuple[DisputeRecord, ...] = ()
    """Negative-polarity absorptions (D34), in firing order - the third and
    last arm of the destroyed-energy ledger. Empty unless the graph carries
    `polarity == -1` nodes and an enabled `PolarityConfig` was given."""

    def ranked(self) -> list[tuple[str, float]]:
        """Activated nodes by accumulated energy, strongest first.

        Ties break on node id so the ordering is stable across platforms; a
        freshness tie-breaker replaces that once nodes carry timestamps.
        """
        return sorted(
            ((node, act.energy) for node, act in self.activations.items()),
            key=lambda item: (-item[1], item[0]),
        )

    def energy_of(self, node: str) -> float:
        """Accumulated energy of `node`, or `0.0` if the web never reached it."""
        activation = self.activations.get(node)
        return activation.energy if activation is not None else 0.0


def propagate(
    graph: Graph,
    seeds: Mapping[str, float],
    config: PropagationConfig | None = None,
    *,
    similarity: SimilarityFn | None = None,
    dedup: DedupConfig | None = None,
    source_of: Mapping[str, str] | None = None,
    negative: Mapping[str, Mapping[str, float]] | None = None,
    conflict: ConflictConfig | None = None,
    absorb: Mapping[str, float] | None = None,
    negative_seed: NegativeSeedConfig | None = None,
    residue: Mapping[str, float] | None = None,
    polarity: PolarityConfig | None = None,
) -> PropagationResult:
    """Inject the query into `graph` at `seeds` and let it spread until it dies.

    Args:
        graph: The graph to spread over.
        seeds: Contact points of the query, mapped to their contact strength
            (cosine similarity, in practice). The injected energy is split
            among them in proportion to that strength, exactly the way a node
            later splits energy among its neighbours. Callers pass similarities;
            this module never computes them.
        config: Propagation settings; defaults to `PropagationConfig()`.
        similarity: Batch node-to-nodes similarity for dynamic dedup, supplied
            by the caller (cosine over the stored embeddings, in practice).
            `None` disables dedup entirely.
        dedup: Dedup settings; only consulted when `similarity` is given.
            `None` or `enabled=False` disables the mechanism.
        source_of: Node id -> document/source id, for vote granularity. Votes
            count corpus support per SOURCE, never per chunk: one vote per
            other source restating an idea, none for the idea's own source.
            Without the mapping every node is its own source.
        negative: Symmetric node -> {opponent: strength} contradiction view
            (`conflict_adjacency` builds it from `NegativeEdge`s). Pre-marked
            at index time; this module never detects contradictions itself.
            `None` disables the mechanism entirely.
        conflict: Neutralisation settings; only consulted when `negative` is
            given. `None` or `enabled=False` disables the mechanism. When on,
            each hop fires every not-yet-fired negative pair whose endpoints
            are both active: both sides lose the same absorbed amount, and a
            frontier node damped below the threshold stops spreading.
        absorb: Node -> absorbing energy, the negative seeds' spread field
            (`core.negative.negative_field` builds it). A node activating
            inside the field loses `min(E, coefficient * field)` once, at
            activation; damped below the threshold it stops spreading - the
            paths into the excluded region die, not just the node.
        negative_seed: Absorption settings; only consulted when `absorb` is
            given. `None` or `enabled=False` disables the mechanism.
        residue: Thermal conversation memory (D22/D32) - node -> leftover
            energy from the previous turn, injected verbatim on top of the
            seed split so the follow-up lands on warm ground. The injected
            TOTAL becomes `seed_energy + sum(residue)` and the relative stop
            threshold scales with it (D5/D27) - that scaling is why the
            threshold is relative at all. Residue is raw energy, not a
            contact weight, so it never enters the proportional seed split;
            non-positive entries are ignored.
        polarity: Negative-knowledge settings (D34); only consulted when the
            graph's `node_data` carries `polarity == -1` atoms. When on, an
            atom the query's energy reaches destroys `coefficient` of its
            arriving energy, once, at activation, and the event lands in the
            `disputes` ledger. `None` or `enabled=False` disables the
            mechanism - negative atoms then behave like ordinary nodes.

    Returns:
        The activated set with accumulated energies, and the reason the web
        stopped growing.

    Raises:
        ValueError: If `seeds` is empty or its weights do not sum above zero.
    """
    config = config or PropagationConfig()
    warmth = {node: energy for node, energy in (residue or {}).items() if energy > 0.0}
    injected = config.seed_energy + sum(warmth.values())
    # Relative on purpose (D5/D27): thermal residue changes the injected
    # total, and the stop rule must scale with it, not with the seed alone.
    threshold = config.threshold_ratio * injected
    dedup_on = similarity is not None and dedup is not None and dedup.enabled
    conflict_on = negative is not None and conflict is not None and conflict.enabled
    absorb_on = (
        absorb is not None and negative_seed is not None and negative_seed.enabled
    )
    # Mass (D11): empty when disabled - every consumer treats a missing
    # entry as 1.0, so the massless path is bit-identical to before.
    masses = node_masses(graph, config.mass)
    negative_atoms: frozenset[str] = frozenset()
    if polarity is not None and polarity.enabled:
        negative_atoms = frozenset(
            node_id for node_id, node in graph.node_data.items() if node.polarity == -1
        )

    contributors: defaultdict[str, list[str]] = defaultdict(list)
    activations: dict[str, Activation] = {}
    suppressed: dict[str, str] = {}
    supporters: dict[str, set[str]] = {}
    taus: list[float] = []
    if dedup_on and dedup is not None and dedup.include_seeds and len(seeds) > 1:
        assert similarity is not None
        seeds, seed_tau = _dedup_seeds(
            seeds, similarity, dedup, source_of, suppressed, supporters
        )
        taus.append(seed_tau)
    frontier = _inject(seeds, config.seed_energy)
    for node, energy in warmth.items():
        # Warm ground (D22): residue is raw energy on top of the seed split;
        # a node that is both a contact and warm simply starts hotter.
        frontier[node] = frontier.get(node, 0.0) + energy
    conflicts: list[ConflictRecord] = []
    fired: set[tuple[str, str]] = set()
    absorptions: list[AbsorptionRecord] = []
    absorbed_nodes: set[str] = set()
    disputes: list[DisputeRecord] = []
    disputed_nodes: set[str] = set()
    stop_reason: StopReason = "threshold"
    hop = 0
    hops_used = 0

    while frontier:
        if hop > config.max_hop:
            stop_reason = "max_hop"
            break

        room = config.max_nodes - len(activations)
        if len(frontier) > room:
            # Overflow guard, not a ranking mechanism: keep the strongest and
            # say so through `stop_reason` rather than truncating in silence.
            frontier = dict(
                sorted(frontier.items(), key=lambda item: (-item[1], item[0]))[:room]
            )
            stop_reason = "max_nodes"

        for node, energy in frontier.items():
            activations[node] = Activation(
                energy=energy,
                hop=hop,
                contributors=tuple(contributors[node]),
            )
        hops_used = hop

        if negative_atoms:
            # Polarity fires before the exclusion field: the atom's "no" was
            # in the corpus before any query existed, so it is the most
            # permanent layer of environment.
            assert polarity is not None
            frontier = _apply_polarity(
                activations,
                frontier,
                negative_atoms,
                polarity,
                disputed_nodes,
                disputes,
                hop,
                threshold,
            )

        if absorb_on:
            # The field fires before conflicts: it is environment, laid down
            # before the query arrived; conflicts are between the survivors.
            assert absorb is not None and negative_seed is not None
            frontier = _apply_absorption(
                activations,
                frontier,
                absorb,
                negative_seed,
                absorbed_nodes,
                absorptions,
                hop,
                threshold,
            )

        if conflict_on:
            assert negative is not None and conflict is not None
            frontier = _apply_conflicts(
                activations,
                frontier,
                negative,
                conflict,
                fired,
                conflicts,
                hop,
                threshold,
            )

        if stop_reason == "max_nodes":
            break

        is_duplicate: Callable[[str], bool] | None = None
        if dedup_on:
            assert similarity is not None and dedup is not None
            active_nodes = sorted(activations)
            tau = adaptive_threshold(active_nodes, similarity, dedup)
            taus.append(tau)
            cleared: set[str] = set()

            def is_duplicate(
                candidate: str,
                *,
                _active: list[str] = active_nodes,
                _tau: float = tau,
                _cleared: set[str] = cleared,
            ) -> bool:
                if candidate in suppressed:
                    return True
                if candidate in _cleared:
                    return False
                survivor = find_survivor(candidate, _active, similarity, _tau)
                if survivor is None:
                    _cleared.add(candidate)
                    return False
                suppressed[candidate] = survivor
                record_support(supporters, candidate, survivor, source_of)
                return True

        arrivals = _distribute(
            graph,
            frontier,
            activations,
            config.damping,
            contributors,
            config.split_alpha,
            is_duplicate,
            masses,
        )
        # The threshold applies to a node's *accumulated* arrival, never to a
        # single contribution: two weak paths that meet must be allowed to add
        # up. That is where the multi-hop answer lives. Mass scales the gate
        # per node (D11): a heavy atom demands more converging evidence.
        frontier = {
            node: energy
            for node, energy in arrivals.items()
            if energy >= threshold * masses.get(node, 1.0)
        }
        hop += 1

    return PropagationResult(
        activations=activations,
        injected_energy=injected,
        threshold=threshold,
        hops_used=hops_used,
        stop_reason=stop_reason,
        votes=votes_of(supporters),
        supporters=merge_support(supporters),
        suppressed=suppressed,
        dedup_thresholds=tuple(taus),
        conflicts=tuple(conflicts),
        absorptions=tuple(absorptions),
        disputes=tuple(disputes),
    )


def _dedup_seeds(
    seeds: Mapping[str, float],
    similarity: SimilarityFn,
    dedup: DedupConfig,
    source_of: Mapping[str, str] | None,
    suppressed: dict[str, str],
    supporters: dict[str, set[str]],
) -> tuple[dict[str, float], float]:
    """Suppress near-duplicate SEEDS before injection (`include_seeds`).

    Two contacts that say the same thing must not each hold a seed slot: the
    2026-08-14 A1 duplication measurement showed injected twins are the
    dominant redundancy damage channel, and neighbour-level suppression never
    reaches them. The duplicate is dropped BEFORE the proportional split, so
    its share flows to the survivors through the renormalisation (energy is
    conserved) and the surviving idea is voted - the exact neighbour
    contract. The strongest contact always survives; order is strongest
    first with id tie-breaks, so the outcome is platform-stable.
    """
    ordered = sorted(seeds, key=lambda node: (-seeds[node], node))
    tau = adaptive_threshold(ordered, similarity, dedup)
    kept: list[str] = []
    for candidate in ordered:
        survivor = find_survivor(candidate, kept, similarity, tau)
        if survivor is None:
            kept.append(candidate)
            continue
        suppressed[candidate] = survivor
        record_support(supporters, candidate, survivor, source_of)
    return {node: seeds[node] for node in kept}, tau


def _apply_polarity(
    activations: dict[str, Activation],
    frontier: Mapping[str, float],
    negative_atoms: frozenset[str],
    config: PolarityConfig,
    disputed_nodes: set[str],
    disputes: list[DisputeRecord],
    hop: int,
    threshold: float,
) -> dict[str, float]:
    """Fire every newly active negative-polarity atom on the query's energy.

    Each atom fires AT MOST ONCE - its accumulated energy is final at
    activation, exactly the negative-seed contract. The atom destroys
    `coefficient` of what arrived (D34, owner's choice: proportional, full by
    default); damped below the threshold it stops spreading, so the opposing
    claim's evidence dies at the atom instead of flowing on through it.
    """
    damped: set[str] = set()
    for node in sorted(frontier):
        if node not in negative_atoms or node in disputed_nodes:
            continue
        if node not in activations:
            continue
        before = activations[node].energy
        destroyed = config.coefficient * before
        if destroyed <= 0.0:
            continue
        disputed_nodes.add(node)
        damped.add(node)
        after = before - destroyed
        activations[node] = replace(activations[node], energy=after)
        disputes.append(
            DisputeRecord(
                node=node,
                hop=hop,
                absorbed=destroyed,
                energy_before=before,
                energy_after=after,
            )
        )
    updated: dict[str, float] = {}
    for node, energy in frontier.items():
        if node not in damped:
            updated[node] = energy
            continue
        current = activations[node].energy
        if current >= threshold:
            updated[node] = current
    return updated


def _apply_absorption(
    activations: dict[str, Activation],
    frontier: Mapping[str, float],
    absorb: Mapping[str, float],
    config: NegativeSeedConfig,
    absorbed_nodes: set[str],
    absorptions: list[AbsorptionRecord],
    hop: int,
    threshold: float,
) -> dict[str, float]:
    """Fire the absorbing field on every newly active node inside it.

    Each node absorbs AT MOST ONCE - the field is consumed at activation, and
    a node's accumulated energy is final at that moment anyway. Untouched
    frontier nodes keep their energy and their right to spread; an absorbed
    node damped below the threshold is removed and stops spreading.
    """
    damped: set[str] = set()
    for node in sorted(frontier):
        if node in absorbed_nodes or node not in absorb or node not in activations:
            continue
        field_energy = absorb[node]
        before = activations[node].energy
        destroyed = min(before, config.coefficient * field_energy)
        if destroyed <= 0.0:
            continue
        absorbed_nodes.add(node)
        damped.add(node)
        after = before - destroyed
        activations[node] = replace(activations[node], energy=after)
        absorptions.append(
            AbsorptionRecord(
                node=node,
                hop=hop,
                absorbed=destroyed,
                field=field_energy,
                energy_before=before,
                energy_after=after,
            )
        )
    updated: dict[str, float] = {}
    for node, energy in frontier.items():
        if node not in damped:
            updated[node] = energy
            continue
        current = activations[node].energy
        if current >= threshold:
            updated[node] = current
    return updated


def _apply_conflicts(
    activations: dict[str, Activation],
    frontier: Mapping[str, float],
    negative: Mapping[str, Mapping[str, float]],
    conflict: ConflictConfig,
    fired: set[tuple[str, str]],
    conflicts: list[ConflictRecord],
    hop: int,
    threshold: float,
) -> dict[str, float]:
    """Fire every pending negative pair with both endpoints active.

    Each pair fires AT MOST ONCE per run - opposite charges annihilate once;
    a later arrival at a damped node does not reopen a settled conflict.
    Pairs fire in canonical `(node_a, node_b)` sorted order so chained
    conflicts (a-b, b-c) resolve deterministically on every platform. Returns
    the updated frontier: a frontier node DAMPED below the threshold is
    removed and stops spreading, which is what "applied per hop" means.
    Untouched frontier nodes keep their energy and their right to spread -
    seeds are never threshold-checked at injection, and a conflict pass that
    fired nothing must not change that.
    """
    damped: set[str] = set()
    for node_a in sorted(negative):
        if node_a not in activations:
            continue
        for node_b in sorted(negative[node_a]):
            if node_b <= node_a or node_b not in activations:
                continue
            pair = (node_a, node_b)
            if pair in fired:
                continue
            energy_a = activations[node_a].energy
            energy_b = activations[node_b].energy
            strength = negative[node_a][node_b]
            new_a, new_b, absorbed = neutralize(
                energy_a, energy_b, strength, conflict.coefficient
            )
            if absorbed <= 0.0:
                continue
            fired.add(pair)
            damped.update(pair)
            activations[node_a] = replace(activations[node_a], energy=new_a)
            activations[node_b] = replace(activations[node_b], energy=new_b)
            conflicts.append(
                ConflictRecord(
                    node_a=node_a,
                    node_b=node_b,
                    strength=strength,
                    hop=hop,
                    absorbed_each=absorbed,
                    energy_a_before=energy_a,
                    energy_a_after=new_a,
                    energy_b_before=energy_b,
                    energy_b_after=new_b,
                )
            )
    updated: dict[str, float] = {}
    for node, energy in frontier.items():
        if node not in damped:
            updated[node] = energy
            continue
        current = activations[node].energy
        if current >= threshold:
            updated[node] = current
    return updated


def _inject(seeds: Mapping[str, float], seed_energy: float) -> dict[str, float]:
    """Split the injected energy across the seed contacts, weighted by strength."""
    if not seeds:
        raise ValueError("at least one seed node is required")
    total = sum(seeds.values())
    if total <= 0.0:
        raise ValueError("seed weights must sum to a positive value")
    return {node: seed_energy * weight / total for node, weight in seeds.items()}


def _distribute(
    graph: Graph,
    frontier: Mapping[str, float],
    activated: Mapping[str, Activation],
    damping: float,
    contributors: defaultdict[str, list[str]],
    split_alpha: float = 1.0,
    is_duplicate: Callable[[str], bool] | None = None,
    masses: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Forward `damping` of each frontier node's energy to its live neighbours.

    Suppressed edges (weight `0.0`), already activated neighbours, and
    neighbours `is_duplicate` flags as near-duplicates of an active node are
    left out of the denominator, so their share is redistributed over the
    remaining neighbours instead of leaking out of the run. Dedup
    redistributes energy; only negative seeds, contradictions and
    negative-polarity atoms are allowed to destroy it.

    Shares are proportional to `weight**split_alpha`; the exponent sharpens or
    keeps the documented proportional split (`1.0`) but never changes the
    forwarded total. With mass (D11) a node forwards `damping ** (1 / mu)` -
    heavy atoms carry further, light ones fade sooner; the forwarded/kept
    split shifts, the energy ledger does not.
    """
    arrivals: dict[str, float] = defaultdict(float)
    for node, energy in frontier.items():
        mass = masses.get(node, 1.0) if masses else 1.0
        effective = damping if mass == 1.0 else damping ** (1.0 / mass)
        outgoing = energy * effective
        live = {
            target: weight**split_alpha
            for target, weight in graph.neighbors(node).items()
            if weight > 0.0
            and target not in activated
            and target not in frontier
            and not (is_duplicate is not None and is_duplicate(target))
        }
        total_weight = sum(live.values())
        if total_weight <= 0.0:
            continue
        for target, weight in live.items():
            arrivals[target] += outgoing * weight / total_weight
            contributors[target].append(node)
    return dict(arrivals)
