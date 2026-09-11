"""Tunable parameters of the propagation core.

No magic numbers are allowed inside the algorithm modules: every knob lives here
as a documented dataclass field, so the developer UI can build its controls from
this object and every mechanism stays individually switchable for ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

EdgeLayer = Literal["semantic", "entity", "structural", "derivation", "learned"]
"""The five edge layers of the hybrid graph; layer choices live here, not in core."""


@dataclass(frozen=True)
class LayerWeights:
    """Relative strength of each edge layer when merging into one adjacency.

    These are THE home of the Phase 1 hand weights - no other module may
    restate them. A weight of `0.0` disables the layer entirely: its edges and
    any nodes only it mentions never enter the merged graph, which is what
    keeps every layer individually switchable for ablation.

    Attributes:
        semantic: Cosine-similarity edges; seed contact and fallback only.
            Hopping along paraphrases returns repetition, not new information.
        entity: Shared entity / concept edges - the main hop fuel.
        structural: Same document, same section, adjacent chunk.
        derivation: Chunk -> proposition containment links (D10, owner's
            2026-08-14 layer choice) - the bridge between the two node
            layers. Provisional hand weight, subject to the grid; `0.0`
            cuts the layers apart (the ablation switch).
        learned: Hebbian usage-reinforced edges. Disabled by default in
            Phase 1; the layer must never mutate the base graph.
    """

    semantic: float = 0.5
    entity: float = 1.0
    structural: float = 0.3
    derivation: float = 1.0
    learned: float = 0.0

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if value < 0.0:
                raise ValueError(
                    f"layer weight {spec.name}={value!r} must not be negative"
                )

    def weight_of(self, layer: EdgeLayer) -> float:
        """Weight of `layer`; field names and layer names coincide by design."""
        return float(getattr(self, layer))


@dataclass(frozen=True)
class StructuralEdgeConfig:
    """Raw within-layer weights of the structural edge builder.

    These are relation weights INSIDE the structural layer; the layer's overall
    strength against other layers stays in `LayerWeights.structural`. The three
    relations are strictly nested (adjacent pairs are also same-section pairs,
    which are also same-document pairs), so they are not independent evidence:
    per pair the strongest enabled relation WINS - weights never sum. A weight
    of `0.0` disables that relation; a pair with no enabled relation is omitted
    entirely, never emitted at `0.0` (that value is reserved for
    dedup-suppressed edges). Defaults are provisional hand values, subject to
    the same grid search as the layer weights.

    Attributes:
        adjacent: Consecutive chunks of the same document in reading order.
        same_section: Chunks sharing a non-None section within one document.
        same_document: Any two chunks of one document. Off by default: it
            builds an O(n^2) clique per document, and proportional splitting
            turns dense cliques straight into the known hub penalty.
    """

    adjacent: float = 1.0
    same_section: float = 0.6
    same_document: float = 0.0

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if value < 0.0:
                raise ValueError(
                    f"structural relation weight {spec.name}={value!r} "
                    "must not be negative"
                )


@dataclass(frozen=True)
class SemanticEdgeConfig:
    """Settings of the semantic (cosine kNN) edge builder.

    The semantic layer is deliberately weak - seed contact and fallback only -
    so a small `k` keeps it sparse. Defaults are provisional hand values,
    subject to the same grid search as the layer weights.

    Attributes:
        k: Neighbours considered per node. A pair is emitted once when either
            endpoint ranks the other in its top-k (union kNN).
        min_similarity: Emission floor; a pair needs `similarity > floor` to be
            emitted (strict, so a cosine of exactly 0.0 never leaks in as a
            fake suppressed edge). The non-negative bound on this floor is the
            mechanism that keeps negative cosine out of the graph, whose edge
            weights must never be negative.
    """

    k: int = 5
    min_similarity: float = 0.0

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if not 0.0 <= self.min_similarity < 1.0:
            raise ValueError("min_similarity must lie in [0, 1)")


@dataclass(frozen=True)
class EntityEdgeConfig:
    """Settings of the entity (shared entity / concept) edge builder.

    The edge weight itself carries no knob: for every entity shared by two
    chunks the pair gains `1 / df(entity)`, where `df` is the number of chunks
    mentioning that entity - a rare entity is strong evidence of a real link,
    a ubiquitous one is barely any. There is deliberately no `min_shared`
    threshold: the rarity sum already fades weak overlap, and a second cutoff
    would be a knob with no evidence behind it.

    Attributes:
        max_df_ratio: Entities mentioned by more than `max_df_ratio * n_chunks`
            chunks are dropped before any pair is built. The 1/df damping
            bounds a stopword entity's *weight* but not its *edge count* - an
            entity in 80% of n chunks still emits ~0.32*n^2 pairs, and dense
            cliques feed the known hub penalty (the same rationale that keeps
            `same_document` off by default). `1.0` disables the guard.
            Default measured on the 2026-08-14 MuSiQue grid: the original
            hand value of 0.5 produced 2.5M entity edges (mean degree 438)
            and ground the propagation to dust; 0.02 won the grid.

            The builder floors the resulting ceiling at 2 (see
            `edges/entity.py`). Below that the layer cannot emit a single
            edge, because an entity needs two chunks to pair - so at this
            ratio every corpus under 100 chunks got an EMPTY entity layer,
            silently, until 2026-08-26. The floor cannot move a measured
            number: the smallest sealed index holds 3336 chunks.
    """

    max_df_ratio: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.max_df_ratio <= 1.0:
            raise ValueError("max_df_ratio must lie in (0, 1]")


@dataclass(frozen=True)
class EntityExtractionConfig:
    """Settings of the hybrid (spaCy + LLM) entity extraction pipeline.

    spaCy handles the bulk; only chunks where it finds fewer than
    `min_entities` entities are routed to the LLM. Passing no LLM client (or
    `min_entities=0`) disables the hybrid entirely - the mandatory ablation
    switch for the LLM path.

    Attributes:
        spacy_model: Pipeline name to load. The default is the multilingual
            WikiNER model (Turkish + English corpora are both in scope).
        labels: Entity labels kept after NER. Deliberately the UNION of the
            WikiNER scheme (PER/ORG/LOC/MISC) and the OntoNotes scheme used by
            the English models (PERSON/GPE/...), so swapping the model never
            silently drops every entity. Numeric and temporal labels (DATE,
            CARDINAL, PERCENT, ...) are excluded on purpose: a shared "2019"
            is not hop fuel, it floods document frequency. A custom set that
            matches neither scheme extracts nothing - and the LLM fallback
            would then mask the mistake, one paid call per chunk.
        min_entities: Chunks where spaCy yields fewer entities than this go to
            the LLM. Provisional hand value; the default of 1 sends only
            spaCy-blind chunks.
    """

    spacy_model: str = "xx_ent_wiki_sm"
    labels: frozenset[str] = frozenset(
        {
            "PER",
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "NORP",
            "FAC",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "MISC",
        }
    )
    min_entities: int = 1

    def __post_init__(self) -> None:
        if not self.spacy_model:
            raise ValueError("spacy_model must not be empty")
        if self.min_entities < 0:
            raise ValueError("min_entities must not be negative")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Settings of the embedding model wrapper (index time, outside core/).

    Attributes:
        model: Sentence-transformers model name. The default is the Phase 1
            decision: multilingual, Turkish + English both in scope.
        batch_size: Encoding batch size.
        device: Explicit device string, or `None` to auto-resolve in the
            fixed order CUDA -> MPS -> CPU.
    """

    model: str = "intfloat/multilingual-e5-large"
    batch_size: int = 32
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass(frozen=True)
class LLMConfig:
    """Settings of the LLM provider used at index time (never inside core/).

    One OpenAI-compatible chat-completions code path covers the local-first
    default (Ollama) and the optional free APIs (Groq, OpenRouter, ...): they
    all speak the same protocol, only `base_url`, `model` and the key differ.

    Attributes:
        base_url: API root ending before `/chat/completions`. The default is
            Ollama's local OpenAI-compatible endpoint, which needs no key.
        model: Model name as the provider expects it.
        api_key_env: NAME of the environment variable holding the API key
            (e.g. "GROQ_API_KEY"), never the key itself - secrets stay out of
            source and out of every log. `None` sends no Authorization header.
        timeout_seconds: Per-request network timeout.
        temperature: Sampling temperature; extraction wants determinism, so
            the default is 0.
        max_tokens: Completion length cap per request.
        max_retries: Additional attempts after the first failed request.
        retry_backoff_seconds: Base of the exponential backoff between
            retries (`backoff * 2**attempt`).
    """

    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.1:8b"
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.temperature < 0.0:
            raise ValueError("temperature must not be negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.retry_backoff_seconds < 0.0:
            raise ValueError("retry_backoff_seconds must not be negative")


@dataclass(frozen=True)
class MassConfig:
    """Node mass (D11) - inertia for the energy ball, per layer.

    Mass is proportional to length and normalised WITHIN each node layer:
    `mu(v) = clamp((length_v / mean_length(layer_v)) ** exponent, floor,
    cap)`. Cross-layer raw length would leave the proposition layer
    permanently dark (the documented trap); per-layer normalisation is what
    lets short propositions compete among themselves.

    Two mechanical effects, the two halves of the design memory's sentence
    "a long atom activates late but carries further":

    1. Activation gate: an arrival survives only when
       `energy >= threshold * mu(v)` - a heavy node demands more converging
       evidence before it lights up. Seeds and residue stay gate-exempt at
       injection, as ever.
    2. Carry: an activated node forwards `damping ** (1 / mu(v))` of its
       energy - heavier nodes forward more and the ball rolls further.
       The forwarded/kept split changes; the energy ledger does not.

    Mass never enters edge weights or the proportional split: each node's
    mass gates only its own activation and scales only its own forwarding,
    which is the structural form of the design rule "mass is inert on
    cross-layer links".

    Attributes:
        enabled: Master switch, OFF by default: the measured Phase 1 winner
            ran massless, so turning mass on is a measurement decision, not
            a code default (the learned-layer pattern). `False` reproduces
            today's behaviour exactly.
        exponent: Sensitivity of mass to relative length; `0.0` flattens
            every mass to 1.0 (a second ablation path), `1.0` is plain
            proportionality. Provisional hand default, to be measured
            (open question #7).
        floor: Lower clamp on mass - a tiny fragment must not become
            near-massless and light up from dust. Provisional.
        cap: Upper clamp on mass - one giant chunk must not become
            unactivatable. Provisional.
    """

    enabled: bool = False
    exponent: float = 1.0
    floor: float = 0.5
    cap: float = 2.0

    def __post_init__(self) -> None:
        if self.exponent < 0.0:
            raise ValueError("exponent must not be negative")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError("floor must lie in (0, 1]")
        if self.cap < 1.0:
            raise ValueError("cap must be at least 1")


@dataclass(frozen=True)
class PropositionConfig:
    """Settings of proposition extraction (`nodes/propositions.py`, D10).

    Extraction is an index-time LLM pass - one call per chunk - so the cost
    lives with the index, never with the query. The extraction-cost question
    (#2) stays open until measured on a real corpus; the harness therefore
    keeps the stage opt-in.

    Attributes:
        max_per_chunk: Hard cap on propositions kept per chunk - a rambling
            completion must not flood the layer. Provisional hand default.
        min_chars: Minimum character length of a kept proposition; shorter
            lines are fragments, not atomic facts. Provisional hand default.
        tag_polarity: Ask the SAME extraction call to prefix negated facts
            with `NEG:`, which become permanent negative-polarity atoms (D34).
            This is the polarity-DETECTION answer to open question #11
            (owner's 2026-08-15 choice: LLM piggyback - zero extra calls,
            catches implicit negation that cue-word rules miss). `False`
            selects the polarity-free prompt: the ablation switch, and the
            byte-identical pre-#11 behaviour.

            **Default flipped to `False` on 2026-08-16, and the flip is the
            measurement's verdict, not a preference.** Both prompt versions
            were audited on the same 3.336-passage corpus. The first tagged
            142 of 26.058 propositions and only 47.2% carried an actual
            negation cue, while 623 untagged ones did - and the cue proxy
            cannot even see the worse failure the dumps showed by eye, a
            model inventing denials the source never made ("the series was
            not canceled due to audience demand"). Tightening the prompt to
            forbid that fixed precision (75.0%) by nearly switching tagging
            off: 4 tags in 29.566 propositions, misses now 160x the catches.
            Neither operating point is usable, so the piggyback answer to #11
            is refuted in both directions: polarity DETECTION needs its own
            pass, not a clause inside the extraction prompt. The negative-atom
            MECHANISM (`core/polarity.py`) is unaffected - it consumes labels
            and never produces them.
    """

    max_per_chunk: int = 12
    min_chars: int = 15
    tag_polarity: bool = False

    def __post_init__(self) -> None:
        if self.max_per_chunk < 1:
            raise ValueError("max_per_chunk must be at least 1")
        if self.min_chars < 1:
            raise ValueError("min_chars must be at least 1")


@dataclass(frozen=True)
class PropagationConfig:
    """Settings for a single spreading-activation run.

    Attributes:
        seed_energy: Total energy injected into the graph by one query.
        damping: Fraction of its energy a node forwards to its neighbours; the
            rest stays behind. Decay is multiplicative, never subtractive, so a
            weak edge fades faster than a strong one.
        threshold_ratio: Stop condition, expressed relative to the injected
            energy rather than as an absolute number. Energy arriving at a node
            below `threshold_ratio * seed_energy` dies there. Relative because
            thermal memory and query profiles both change the injected total.
        max_hop: Hard overflow guard on propagation depth. The threshold is the
            real stop condition; this only prevents surprises. Raised from 6
            to 8 on 2026-08-16 after the depth data came in: across four
            sealed 1000-question runs every query stopped on `threshold`, but
            the deepest observed hop was 6 - equal to the cap itself (2Wiki,
            2 queries). "The brake never fired" was true and yet there was no
            margin left, and a deeper corpus would have bound it silently.
            Raising it costs nothing, because the threshold stops first in
            every measured run; what it buys is headroom between the observed
            depth and the ceiling.
        max_nodes: Hard overflow guard on the size of the activated set. Same
            status as `max_hop`: safety brake, not a `top-k` in disguise.
        split_alpha: Exponent applied to edge weights when a node splits its
            outgoing energy (share of neighbour i is `w_i**alpha / sum of
            w**alpha`). `1.0` is the documented pure proportional split; values
            above 1 concentrate energy on the strongest edges. This is the
            known-risks softening for the hub penalty: on dense graphs with a
            narrow similarity band the plain split grinds energy into dust
            across hundreds of near-equal neighbours. The forwarded total is
            unchanged - the exponent reshapes shares, never the energy ledger.
        mass: Node-mass settings (D11); disabled by default - see
            `MassConfig`. Nested here so profiles and retrieval configs
            carry it without new plumbing.
    """

    seed_energy: float = 10.0
    damping: float = 0.60
    threshold_ratio: float = 0.15
    max_hop: int = 8
    max_nodes: int = 512
    split_alpha: float = 1.0
    mass: MassConfig = field(default_factory=MassConfig)

    def __post_init__(self) -> None:
        if self.seed_energy <= 0.0:
            raise ValueError("seed_energy must be positive")
        if not 0.0 < self.damping < 1.0:
            raise ValueError("damping must lie strictly between 0 and 1")
        if not 0.0 <= self.threshold_ratio < 1.0:
            raise ValueError("threshold_ratio must lie in [0, 1)")
        if self.max_hop < 0:
            raise ValueError("max_hop must not be negative")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")
        if self.split_alpha <= 0.0:
            raise ValueError("split_alpha must be positive")

    @property
    def threshold(self) -> float:
        """Absolute energy floor implied by `threshold_ratio` for this run."""
        return self.threshold_ratio * self.seed_energy


@dataclass(frozen=True)
class DedupConfig:
    """Settings of query-time dynamic redundancy suppression (dedup -> vote).

    During propagation, a candidate neighbour that is a near-duplicate of an
    already active node is suppressed for the rest of the run: its edge share
    is renormalised over the surviving neighbours (energy is redistributed,
    never destroyed) and the surviving idea's vote count is incremented.
    Repetition becomes evidence instead of noise - dropping duplicates the MMR
    way would throw that signal away.

    The duplicate threshold is ADAPTIVE, computed per hop from the similarity
    distribution of the currently active set ("similar" depends on the query,
    so no fixed cosine cut can be right for every question). The computed value
    is recorded in the result so a UI can show it, as the design requires.

    Attributes:
        enabled: Master switch; `False` makes propagation behave exactly as if
            no similarity function had been supplied (the ablation switch).
        sigma: Width of the adaptive cut: `tau = mean + sigma * std` over the
            active set's pairwise similarities. Only similarities far above
            what the active set considers normal count as duplication.
        floor: Absolute lower bound on `tau`. A flat or low similarity
            distribution must never push the cut into ordinary-relatedness
            territory and start suppressing genuinely distinct nodes. The
            default is the measured safe point of the 2026-08-14 MuSiQue
            dose-response sweep (tour 11b): on that pre-deduplicated corpus
            S@5 deltas vs no-dedup were floor .80 -> -.018 and .85 -> -.009
            (both significant harm), .90 -> -.002 and .95 -> -.001 (ties).
            MuSiQue carries no real redundancy, so the sweep bounds the
            mechanism's cost, not its benefit. The benefit was measured on
            2026-08-14 with injected exact duplicates (A1): with seed-level
            dedup and the elastic refill, floor .95 recovered +.007 S@5 at a
            10% duplication dose and +.027 at 40% (both CI-significant),
            while costing a statistical tie on the clean corpus; .90 gave
            less benefit under duplication AND significant harm when clean,
            so .95 stays the default.
        min_pairs: Minimum number of observed pairwise similarities before the
            adaptive formula is trusted; below it `tau` falls back to `floor`.
        include_seeds: Also suppress near-duplicate SEEDS at injection time.
            The 2026-08-14 A1 duplication measurement showed the dominant
            redundancy damage channel is two contact slots taken by the same
            passage (source + copy tie in the ranking); neighbour-level
            suppression never touches those twins because they are injected,
            not distributed to. The duplicate seed's share flows to the
            survivors through the proportional split (energy conserved) and
            the surviving idea is voted, exactly the neighbour contract.
            `False` restores injection-blind behaviour (the ablation switch).
        distinct_sources: Two seed slots of ONE query part may not land on the
            same source. `include_seeds` catches twins by cosine, which is the
            right test for a copied passage and the wrong one for a two-layer
            index: two propositions of the same passage are different
            sentences, so they are not near-duplicates, yet a colour that
            seeds both explores one passage instead of two. Measured
            2026-08-16 on `musique_prop200`: 287 of 534 colours (54%) had both
            seeds on one passage - zero of 534 on the chunk-only control - and
            the coloured web paid -.0524 CI [-.0854, -.0203] P=.001 for it.
            The source key comes from `Node.source_id` (a proposition inherits
            its parent's), so no id-string convention leaks into retrieval.
            On a single-layer index every chunk is its own source and the rule
            is a no-op, which is why it ships ON: it is the missing half of an
            existing rule, not a new mechanism. `False` is the ablation switch.
    """

    enabled: bool = True
    sigma: float = 2.0
    floor: float = 0.95
    min_pairs: int = 8
    include_seeds: bool = True
    distinct_sources: bool = True

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ValueError("sigma must not be negative")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError("floor must lie in (0, 1]")
        if self.min_pairs < 1:
            raise ValueError("min_pairs must be at least 1")


@dataclass(frozen=True)
class ConflictConfig:
    """Negative-charge neutralisation between contradicting atoms (D15).

    When both endpoints of a pre-marked negative edge are active, each side
    loses `coefficient * strength * min(E_a, E_b)` - equal charge quantities
    annihilate (owner's choice, 2026-08-14). Applied per hop, inside
    propagation, so a neutralised atom also stops spreading. This is the one
    mechanism besides negative seeds that DESTROYS energy; every event lands
    in the result's conflict ledger.

    Attributes:
        enabled: Master switch; `False` makes propagation behave exactly as if
            no negative edges had been supplied (the ablation switch).
        coefficient: Global damping scale on the absorbed amount. `1.0` is
            full annihilation at full NLI confidence - a hand default, subject
            to measurement once NLI edges exist on a real corpus.
    """

    enabled: bool = True
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.coefficient <= 1.0:
            raise ValueError("coefficient must lie in (0, 1]")


@dataclass(frozen=True)
class NegativeSeedConfig:
    """Energy-absorbing negative seeds - "excluding X" as physics, not filter.

    A post-filter only removes the unwanted node from the result; chunks that
    arrived BECAUSE they neighbour X survive it. The negative seed instead
    spreads its own field over X's whole region with the ordinary propagation
    rules (owner's choice, 2026-08-14), and positive energy entering that
    region is destroyed - the paths into the region die, not just the node.
    Absorption is the second and last mechanism allowed to destroy energy
    (with conflicts); every event lands in the result's absorption ledger.

    Attributes:
        enabled: Master switch; `False` behaves exactly as if no negative
            queries had been supplied (the ablation switch).
        seed_width: First-contact atoms per negative query, mirroring the
            positive contact rule.
        energy_ratio: Negative budget as a fraction of `seed_energy` - the
            owner chose a separate knob over hard symmetry; `1.0` (the
            default) IS the symmetric case, and the knob exists for tuning.
        coefficient: Scale on the absorbed amount: a node holding field `A`
            destroys `min(E, coefficient * A)` of an arriving energy `E`,
            once, at activation time.
    """

    enabled: bool = True
    seed_width: int = 5
    energy_ratio: float = 1.0
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.seed_width < 1:
            raise ValueError("seed_width must be at least 1")
        if self.energy_ratio <= 0.0:
            raise ValueError("energy_ratio must be positive")
        if not 0.0 < self.coefficient <= 1.0:
            raise ValueError("coefficient must lie in (0, 1]")


@dataclass(frozen=True)
class OutputConfig:
    """Thresholds of the honesty outputs (theme clusters, gap warnings, D35).

    A theme cluster is a connected component of the ACTIVATED subgraph,
    computed at query time (owner's choice, 2026-08-14) - consistent with the
    project's stance that query-time structure beats index-time labels. A
    cluster counts as DENSE when it passes BOTH knobs below; two dense
    clusters with no connection between them raise a gap warning (D18) - the
    "no bridge" test is the component split itself, which costs nothing.

    The refusal report (D35) is caller-triggered: the library builds it on
    demand and never decides when confidence is "low" - that threshold is the
    caller's policy (D17 wins the D35 wording conflict; owner's choice).

    Attributes:
        min_cluster_nodes: Node-count floor for a cluster to count as dense.
        min_cluster_energy_share: Fraction of the run's total energy a
            cluster must hold to count as dense. Both knobs together keep
            single stray nodes from raising false gap warnings.
    """

    min_cluster_nodes: int = 3
    min_cluster_energy_share: float = 0.15

    def __post_init__(self) -> None:
        if self.min_cluster_nodes < 1:
            raise ValueError("min_cluster_nodes must be at least 1")
        if not 0.0 <= self.min_cluster_energy_share <= 1.0:
            raise ValueError("min_cluster_energy_share must lie in [0, 1]")


@dataclass(frozen=True)
class NLIEdgeConfig:
    """Index-time NLI contradiction marking (D26) - `edges/nli.py`.

    Attributes:
        contradiction_threshold: Minimum contradiction confidence (max over
            both premise/hypothesis directions) for a candidate pair to become
            a negative edge. Conservative by default: a false negative edge
            wrongly destroys energy at query time, so precision beats recall
            here. Hand default; the model choice and threshold are open
            question #10, to be measured.
    """

    contradiction_threshold: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 < self.contradiction_threshold <= 1.0:
            raise ValueError("contradiction_threshold must lie in (0, 1]")


@dataclass(frozen=True)
class NLIModelConfig:
    """Settings of the real NLI model wrapper (`spiyweb/nli.py`).

    The model answers open question #10 (owner's 2026-08-15 choice): a small
    multilingual DeBERTa fine-tuned on XNLI + 2.7M NLI pairs - the strongest
    model of its size class, and it fits an 8 GB GPU with room to spare. The
    wrapper lives OUTSIDE `core/` and behind the `NLIModel` Protocol, so the
    choice stays a config value, never an import.

    Attributes:
        model_name: Hugging Face id of the sequence-classification NLI model.
            The wrapper locates the contradiction class through the model's
            own `id2label`, so any 3-way NLI head works here.
        batch_size: Pairs scored per forward pass. Provisional hand default
            sized for an 8 GB GPU; raise it on bigger hardware.
        max_length: Token truncation bound per (premise, hypothesis) pair.
            Propositions are short by construction; chunks get truncated with
            the documented blur.
        device: Explicit torch device, or `None` for the fixed resolution
            order CUDA -> MPS -> CPU.
    """

    model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    batch_size: int = 16
    max_length: int = 512
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8")


@dataclass(frozen=True)
class NLICandidateConfig:
    """Candidate-pair selection for index-time NLI (`evaluation/index.py`).

    Contradiction is only worth scoring between texts that talk about the
    same thing, so candidates are the high-cosine pairs of the corpus - the
    proposition layer when the index has one (contradiction is sharp on
    propositions), the chunk layer otherwise (documented blur, D26).

    Attributes:
        top_k: Neighbours considered per node when pairing (union kNN).
            Provisional hand default.
        min_similarity: Cosine floor a pair must clear to become a candidate.
            High on purpose: NLI cost scales with the candidate count, and
            low-similarity pairs contradict by talking past each other.
            Provisional hand default.
        max_pairs: Hard cap on scored pairs - the overflow guard that keeps a
            dense corpus from turning the NLI stage into an open-ended bill.
            Highest-similarity pairs win the cut. Provisional hand default.
        require_shared_subject: Both texts must name the same subject before
            the pair is scored. Measured need (2026-08-16, open question
            #10): cosine alone declared 5.418 of 20.000 scanned pairs
            contradictory on an encyclopaedic corpus with no natural
            contradictions, and all fifteen strongest ones were ONE error -
            two same-kind, different-entity texts (two radio stations, two
            villages, two high schools), where NLI silently assumes premise
            and hypothesis share a subject. Raising the threshold cannot fix
            it: the false positives arrive at .9995 while the single genuine
            contradiction found sat at .9990. The cost of leaving it off was
            also measured - the shipped .90 cut ate 28% of a query's energy
            and dropped S@5 by -.0187 CI [-.0270, -.0104] P=.000 - so the
            requirement ships ON. `False` is the ablation switch.
        subject_prefix_chars: How much of a text counts as its SUBJECT
            region. The shared entity must appear inside the leading slice of
            both texts, which is what separates "two texts about the National
            Assembly of Pakistan" from "two radio stations that both mention
            Jackson": in the second, the shared name is an object, never the
            subject. A character window is an approximation with two known
            failure modes - a long opening clause can push the true subject
            out of the window, and a short one can pull an object in. `40` is
            where the audit's own texts sit: "Jackson" arrives around
            character 50 of the composed passage, after the title and the
            "is a radio station licensed to" opening.
        max_subject_df_ratio: Share of the corpus an entity may appear in and
            still qualify as a subject. Same intuition as the entity layer's
            `max_df_ratio`, applied to a different job: a name carried by
            hundreds of passages identifies a category, not a subject. On the
            2026-08-16 audit corpus "Canadian" (1.40%) and "Texas" (1.07%)
            were exactly the names keeping two Calgary radio stations and two
            Texan hamlets paired, while "National Assembly of Pakistan"
            (0.03%) is the kind of name that identifies one thing. Measured
            effect on the recorded edge set at window 40: 5.418 -> 655 with
            the window alone, -> 393 with this cut at .005. The RECALL cost
            is unmeasured and real: a genuine contradiction about a corpus-
            wide subject would be dropped. Precision wins the trade here
            because a false negative edge destroys query energy (the shipped
            cut cost -.0187 S@5), while a missed one costs nothing. `1.0`
            disables the cut and keeps the window test alone.
    """

    top_k: int = 5
    min_similarity: float = 0.80
    max_pairs: int = 20000
    require_shared_subject: bool = True
    subject_prefix_chars: int = 40
    max_subject_df_ratio: float = 0.005

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= self.min_similarity < 1.0:
            raise ValueError("min_similarity must lie in [0, 1)")
        if self.max_pairs < 1:
            raise ValueError("max_pairs must be at least 1")
        if self.subject_prefix_chars < 1:
            raise ValueError("subject_prefix_chars must be at least 1")
        if not 0.0 < self.max_subject_df_ratio <= 1.0:
            raise ValueError("max_subject_df_ratio must lie in (0, 1]")


@dataclass(frozen=True)
class LearnedLayerConfig:
    """Settings of the Hebbian learned layer (`edges/learned.py`).

    The layer strengthens edges that actually carried energy and thins the
    rest - the spider thickens the thread it uses. It NEVER mutates the base
    graph: it emits its own `"learned"` edge list for `Graph.from_layers`,
    and `LayerWeights.learned` (default 0.0 = off) is the read-side ablation
    switch. All values here are PROVISIONAL hand defaults - the forgetting
    coefficient is open question #9, to be measured.

    Attributes:
        enabled: Write-side switch; `False` makes `reinforce()` a no-op, so
            an existing layer can be frozen without discarding it.
        learning_rate: Scale of one reinforcement. The increment is
            energy-proportional WITH saturation (owner's 2026-08-14 choice):
            `learning_rate * min(1.0, carried / injected_energy)`, then the
            strength is capped at `max_strength` - strong paths thicken
            faster, but never without bound.
        forgetting: Ageing multiplier applied to EVERY stored edge at the
            start of each `reinforce()` call. Mandatory: without it the layer
            collapses onto whatever was asked most often (known risk).
        max_strength: Hard cap on a single edge's learned strength.
    """

    enabled: bool = True
    learning_rate: float = 0.1
    forgetting: float = 0.99
    max_strength: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.forgetting <= 1.0:
            raise ValueError("forgetting must lie in (0, 1]")
        if self.max_strength <= 0.0:
            raise ValueError("max_strength must be positive")


@dataclass(frozen=True)
class ThermalConfig:
    """Thermal conversation memory (D22/D32) - follow-ups land on warm ground.

    A fraction of the previous turn's accumulated energy persists as residue
    and is injected alongside the new query's seeds, so a follow-up question
    starts in the region the conversation already warmed. The relative stop
    threshold scales with the injected TOTAL (seed energy + residue) - that
    scaling is the reason the threshold is relative in the first place
    (D5/D27). Reset is hybrid (D32): the caller's `reset()` by default, plus
    an optional topic-change auto-detection behind `auto_reset` - off by
    default, the two code paths' cost was accepted knowingly.

    Attributes:
        enabled: Master switch; `False` makes the session behave exactly like
            stateless `retrieve()` calls (the ablation switch).
        residue_ratio: Fraction of each activated node's energy that persists
            into the next turn. The design band is 20-30%; kept low on
            purpose - on a topic change the residue warms the WRONG region.
            Provisional hand default, to be measured.
        auto_reset: Optional topic-change auto-detection (owner's 2026-08-14
            signal choice: contact overlap). Before injecting residue, the
            new query's index contacts are compared against the warm node
            set; when the overlapping fraction falls below `min_overlap` the
            session resets itself. Embedding-free and language-independent -
            it measures similarity to the previous ACTIVE SET, not to the
            previous question's wording.
        min_overlap: The auto-reset cut: reset fires when
            `|contacts ∩ warm| / |contacts| < min_overlap`. At the default
            seed width this means "reset only when NO contact is warm" - the
            most conservative reading of the design's "similarity to the
            previous active set is low". Provisional hand default, to be
            measured.
    """

    enabled: bool = True
    residue_ratio: float = 0.25
    auto_reset: bool = False
    min_overlap: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.residue_ratio < 1.0:
            raise ValueError("residue_ratio must lie strictly between 0 and 1")
        if not 0.0 <= self.min_overlap <= 1.0:
            raise ValueError("min_overlap must lie in [0, 1]")


@dataclass(frozen=True)
class PolarityConfig:
    """Negative-knowledge atoms (D34) - the corpus's "no" as physics.

    Embeddings do not carry negation: a query asserting "X is Y" lands right
    next to the atom that says "X is not Y". A negative-polarity atom
    (`Node.polarity == -1`, marked at index time OUTSIDE core/) therefore
    absorbs the energy that reaches it - the opposing claim's evidence dies
    there instead of reinforcing it - and every absorption lands in the
    result's dispute ledger, from which the "corpus disputes this" warning is
    built. This is the third and last mechanism allowed to destroy energy
    (with conflicts and negative seeds).

    Attributes:
        enabled: Master switch; `False` makes negative-polarity atoms behave
            exactly like ordinary nodes (the ablation switch, as D34
            requires).
        coefficient: Fraction of the arriving energy the atom destroys, once,
            at activation (owner's 2026-08-14 choice: proportional with full
            absorption as the default). `1.0` removes the atom's region from
            the ranking entirely; the dispute ledger still reports it.
    """

    enabled: bool = True
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.coefficient <= 1.0:
            raise ValueError("coefficient must lie in (0, 1]")


@dataclass(frozen=True)
class ConsolidationConfig:
    """Settings of offline base-graph consolidation pruning (D23).

    Consolidation removes edges that never carried energy across the recorded
    runs - the offline counterpart of the learned layer's per-round
    forgetting. Node merging is deliberately deferred to Phase 2 (it is
    irreversible); pruning itself is REVERSIBLE by design: `prune_layers`
    returns the removed edges alongside the kept ones, and feeding both back
    restores the original graph.

    Attributes:
        min_runs: Minimum number of recorded propagation runs before pruning
            removes anything. An unused edge may simply guard a question
            nobody has asked yet, so the evidence must accumulate first;
            below this count `prune_layers` keeps every edge. Provisional
            hand default, to be measured.
    """

    min_runs: int = 100

    def __post_init__(self) -> None:
        if self.min_runs < 1:
            raise ValueError("min_runs must be at least 1")


@dataclass(frozen=True)
class CorpusLintConfig:
    """Thresholds of the corpus-lint diagnostic (D37) - Phase 2's own product.

    Every value here is a REPORTING threshold, never a mechanism: nothing in
    this config changes what retrieval does, only what the lint pass calls
    worth mentioning. That is why they can be tuned freely, and why a corpus
    owner who disagrees with one can move it without touching a measured
    number.

    Attributes:
        min_orphan_nodes: Smallest island worth naming. `2` because a
            single unconnected atom is reported separately, as `isolated`.
        min_hub_degree: Below this many live neighbours a node is simply not
            a hub, whatever its share works out to.
        hub_share_floor: A hub is reported when its STRONGEST neighbour
            receives no more than this fraction of what it forwards. Not a
            degree cutoff: eight hundred neighbours behind one dominant edge
            waste nothing, and a degree-based rule would report them anyway.
        split_alpha: The exponent the share is computed with. Mirrors
            `PropagationConfig.split_alpha` so the number reported is the
            share energy would actually take - set them together or the lint
            describes a propagation that is not the one being run.
        duplicate_weight: Raw semantic-layer cosine at or above which two
            atoms are called near-identical. A STATIC proxy: query-time
            duplicate detection is dynamic and adaptive (D6), and this pass
            does not pretend to reproduce it.
        min_duplicates_per_source: Near-duplicate pairs inside one document
            before the document itself is reported. Within-source repetition
            never becomes a vote (D7), so it is pure cost.
        min_contradictions_per_source: Marked contradiction pairs before a
            source is named in the contradiction map.
        max_per_kind: Findings kept per kind, worst first. A corpus with ten
            thousand orphans needs the worst twenty and a count, not ten
            thousand rows.
        max_nodes_per_finding: Atoms listed inside one finding, so a report
            of a large island stays readable.
    """

    min_orphan_nodes: int = 2
    min_hub_degree: int = 20
    hub_share_floor: float = 0.05
    split_alpha: float = 1.0
    duplicate_weight: float = 0.95
    min_duplicates_per_source: int = 1
    min_contradictions_per_source: int = 1
    max_per_kind: int = 20
    max_nodes_per_finding: int = 20

    def __post_init__(self) -> None:
        if self.min_orphan_nodes < 2:
            raise ValueError("min_orphan_nodes must be at least 2")
        if self.min_hub_degree < 2:
            raise ValueError("min_hub_degree must be at least 2")
        if not 0.0 < self.hub_share_floor < 1.0:
            raise ValueError("hub_share_floor must lie strictly between 0 and 1")
        if self.split_alpha <= 0.0:
            raise ValueError("split_alpha must be positive")
        if not 0.0 < self.duplicate_weight <= 1.0:
            raise ValueError("duplicate_weight must lie in (0, 1]")
        if self.max_per_kind < 1:
            raise ValueError("max_per_kind must be at least 1")
        if self.max_nodes_per_finding < 1:
            raise ValueError("max_nodes_per_finding must be at least 1")


ATTACH_DIR = ".spiyweb"
"""Where a running `spiyweb` monitor leaves its marker and where an
application's records are appended while that marker is fresh. Relative to
the process's working directory: the monitor is opened in the project folder,
and the application runs from the same folder."""

ATTACH_STALE_S = 10.0
"""A marker older than this many seconds belongs to a monitor that crashed or
was closed. The library stops writing on its own; nothing has to clean up."""


@dataclass(frozen=True)
class TraceConfig:
    """Settings of the query trace layer (D38) - what an application recorded.

    A trace is a recorded CALL, not a re-runnable one: the viewer it feeds
    reads what already happened, so it never loads a second copy of the graph
    and the vector store into memory. That only works while the record stands
    on its own, which is why `include_edges` and `include_texts` default on
    and turning either off is a documented downgrade rather than a saving.

    Attributes:
        enabled: Whether `SpiywebIndex` records its queries at all. On by
            default and in memory only - the ring buffer costs a few
            megabytes and nothing leaves the process.
        capacity: How many records the in-memory ring buffer keeps. The
            oldest is dropped once it is full, so a long-running application
            never grows without bound.
        directory: Where to ALSO append the records as JSONL. `None` - the
            default - means the disk is never touched: passage text lands in
            the file, so writing has to be an explicit choice.
        include_texts: Whether each traced atom carries the text it was
            indexed as. Off makes the record unreadable without the corpus;
            it exists for callers who must not copy passages at all.
        text_chars: Truncate traced text to this many characters. `0` - the
            default - keeps it whole.
        include_edges: Whether the activated subgraph's edges are recorded.
            Off makes the record undrawable without the graph.
        max_edges: Overflow guard on the recorded edge count, strongest
            first. `0` means no limit. A record that hit the guard says so
            (`edges_truncated`) rather than quietly showing a thinner web.
        attach_dir: The one exception to "disk on request": while somebody
            runs the `spiyweb` monitor in this folder, its fresh marker file
            under this directory is the request, and records are appended
            to `<attach_dir>/traces.jsonl` so the monitor can play them.
            Passage text lands there, in a folder that ignores itself from
            git. `None` switches the mechanism off - the production setting.
            Ignored when `directory` is set.
        attach_stale_s: How old the marker may be and still count as a
            listening monitor. One `stat` per query, never cached.
    """

    enabled: bool = True
    capacity: int = 200
    directory: str | Path | None = None
    include_texts: bool = True
    text_chars: int = 0
    include_edges: bool = True
    max_edges: int = 4000
    attach_dir: str | Path | None = ATTACH_DIR
    attach_stale_s: float = ATTACH_STALE_S

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        if self.text_chars < 0:
            raise ValueError("text_chars must not be negative")
        if self.max_edges < 0:
            raise ValueError("max_edges must not be negative")
        if self.attach_stale_s <= 0:
            raise ValueError("attach_stale_s must be positive")


@dataclass(frozen=True)
class WatchConfig:
    """Settings of the terminal monitor - bare `spiyweb`.

    Not part of the query contract and not in `spiyweb.__all__`: the monitor
    is an interface over the trace layer, and these are its knobs.

    Attributes:
        attach_dir: The folder the marker and the traces live in; must agree
            with `TraceConfig.attach_dir` of the application being watched.
        stale_s: Reported marker age past which an application would stop
            writing - the library's own limit is `TraceConfig.attach_stale_s`.
        heartbeat_s: How often the marker is touched. Must beat `stale_s`
            with room to spare.
        poll_ms: How often the trace file and the keyboard are checked.
        hop_delay_ms: Time spent revealing one hop of a record. `0` plays
            the last frame only.
        hold_ms: Pause on a finished hop before the next one starts.
        map_enabled: Whether the ring map is drawn beside the ranking.
        map_min_width: Below this many columns the map is dropped.
        map_rows: Height of the ring map.
        ranking_width: Columns reserved for the ranking beside the map.
        label_chars: Atom ids on the map and in the ranking are cut here.
        max_rows: Ranking rows shown before "+n more".
        bar_width: Width of the energy bars.
        pet_enabled: Whether the spider sits in the welcome box.
        sleep_after_s: Without a record for this long the status reads
            "no app attached", the marker notwithstanding.
        find_max_depth: How deep `/find` walks below the working directory.
        find_max_files: How many files `/find` reads before it stops.
        find_max_file_bytes: Files larger than this are skipped by `/find`.
    """

    attach_dir: str | Path = ATTACH_DIR
    stale_s: float = ATTACH_STALE_S
    heartbeat_s: float = 2.0
    poll_ms: int = 40
    hop_delay_ms: int = 650
    hold_ms: int = 350
    map_enabled: bool = True
    map_min_width: int = 72
    map_rows: int = 15
    ranking_width: int = 72
    label_chars: int = 12
    max_rows: int = 12
    bar_width: int = 22
    pet_enabled: bool = True
    sleep_after_s: float = 60.0
    find_max_depth: int = 6
    find_max_files: int = 5000
    find_max_file_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if self.stale_s <= 0 or self.heartbeat_s <= 0:
            raise ValueError("stale_s and heartbeat_s must be positive")
        if self.heartbeat_s * 2 > self.stale_s:
            raise ValueError("heartbeat_s must be at most half of stale_s")
        if self.poll_ms < 1:
            raise ValueError("poll_ms must be at least 1")
        if self.hop_delay_ms < 0 or self.hold_ms < 0:
            raise ValueError("hop_delay_ms and hold_ms must not be negative")
        if self.map_rows < 5:
            raise ValueError("map_rows must be at least 5")
        if self.max_rows < 1 or self.bar_width < 1 or self.label_chars < 1:
            raise ValueError("max_rows, bar_width and label_chars must be positive")
        if self.find_max_depth < 0 or self.find_max_files < 1:
            raise ValueError("find limits must be sensible")


@dataclass(frozen=True)
class RetrievalConfig:
    """Settings of one end-to-end `retrieve()` call.

    Bundles the seed-contact width with the propagation it feeds on purpose:
    a query profile (D13 - precise / explore / compare) is exactly a
    "damping + threshold + seed width" package, so the future `profiles.py`
    becomes a factory of these objects rather than a parallel config tree.

    Attributes:
        seed_width: Number of first-contact atoms the query touches (Phase 1
            decision: 5). The seed energy is split among them proportionally
            to cosine similarity; the split itself lives in the core.
        propagation: Settings of the spreading-activation run the seeds feed.
        contact_overfetch: Elastic contact refill (2026-08-14 A1 decision).
            When dedup is active, the index is searched
            `seed_width * contact_overfetch` deep and the seed slots are
            filled with the first DISTINCT ideas: a contact that duplicates
            an already selected one is skipped (and voted), and its slot goes
            to the NEXT distinct contact. Without the refill a duplicated
            corpus collapses both slots onto one passage and the web explores
            half as many ideas - the dominant damage channel the A1
            duplication measurement exposed. `1` disables the refill (the
            ablation switch); the value is inert while dedup is off.
    """

    seed_width: int = 5
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    contact_overfetch: int = 3

    def __post_init__(self) -> None:
        if self.seed_width < 1:
            raise ValueError("seed_width must be at least 1")
        if self.contact_overfetch < 1:
            raise ValueError("contact_overfetch must be at least 1")


def _winning_propagation() -> PropagationConfig:
    """Measured winner of the 2026-08-14 grid campaign (see below)."""
    return PropagationConfig(threshold_ratio=0.01, split_alpha=3.0)


@dataclass(frozen=True)
class ColoredRetrievalConfig:
    """Settings of one coloured multi-seed `retrieve_colored()` call (D12).

    These defaults are the MEASURED WINNER of the 2026-08-14 MuSiQue
    grid/ablation campaign (tours 1-12): the question is decomposed into
    colours by a dedicated decomposition model and chained level by level
    (colour i's top-passage answer feeds colour i+1's query) - S@5 .512 at
    ~2.6 LLM calls per question, the first statistically significant win
    over the iterative baseline's .463 at ~4 calls (paired bootstrap CI
    [+.033, +.065], tour 12). They deliberately differ from
    `PropagationConfig`'s own defaults: the core defaults carry the
    canonical worked example (CLAUDE.md §2.6) and stay untouched; the
    measured operating point lives here.

    Attributes:
        seed_width: First-contact atoms PER COLOUR. Narrower than the plain
            5-seed contact on purpose: with 2-4 colours the total contact
            count would otherwise fill the whole S@5 window with seeds.
        propagation: Propagation settings of the coloured run. The default
            carries the grid winner (threshold_ratio 0.01, split_alpha 3.0).
        max_colors: Hard cap on the number of colours kept from the LLM
            decomposition; MuSiQue questions never chain more than 4 facts.
            Tour 12 measured clamping tighter than the decomposition itself
            as neutral-to-harmful (clamp3 .507, clamp2 .431), so the cap
            stays a safety guard, not a tuning knob.
        chain_mode: How intermediate answers feed later colours.
            "sequential" (tour 10/12 winner): each colour's top-passage
            answer is extracted and appended to the NEXT colour's query,
            level by level (n_colors - 1 extra calls). "single" (tour 9,
            S@5 .460): one extraction from colour 0's top passage, appended
            to every later colour (1 extra call). "none" (tour 7 ablation,
            S@5 .413): decomposition only.
        decomposition_model: Model for the question -> colours split. The
            tour-12 winner is qwen3.5:9b (S@5 .512 vs llama3.1:8b's .469;
            gold hop-count agreement 72.9% vs 35.7%). Extraction stays on
            the default `LLMConfig` model - tour 12 measured qwen
            extraction as no gain (.507).
        decomposition_no_think: Route the decomposition through Ollama's
            native endpoint with thinking disabled. qwen3.5 is a thinking
            model: through the OpenAI-compatible endpoint it burns the whole
            token budget on reasoning and returns empty content.
        max_answer_words: Word cap on the extracted intermediate answer; a
            rambling extraction would drown the sub-query it is appended to.
        contact_overfetch: Elastic contact refill, per colour - the same
            mechanism `RetrievalConfig.contact_overfetch` documents. With a
            seed width of 2 a single duplicated passage captures BOTH slots
            of a colour, so the refill matters most here. `1` disables it;
            inert while dedup is off.
    """

    seed_width: int = 2
    propagation: PropagationConfig = field(default_factory=_winning_propagation)
    max_colors: int = 4
    chain_mode: str = "sequential"
    decomposition_model: str = "qwen3.5:9b"
    decomposition_no_think: bool = True
    max_answer_words: int = 10
    contact_overfetch: int = 3

    def __post_init__(self) -> None:
        if self.seed_width < 1:
            raise ValueError("seed_width must be at least 1")
        if self.contact_overfetch < 1:
            raise ValueError("contact_overfetch must be at least 1")
        if self.max_colors < 1:
            raise ValueError("max_colors must be at least 1")
        if self.chain_mode not in ("none", "single", "sequential"):
            raise ValueError(
                f"chain_mode {self.chain_mode!r} must be "
                "'none', 'single' or 'sequential'"
            )
        if not self.decomposition_model:
            raise ValueError("decomposition_model must not be empty")
        if self.max_answer_words < 1:
            raise ValueError("max_answer_words must be at least 1")


@dataclass(frozen=True)
class IterativeBaselineConfig:
    """Settings of the IRCoT-style iterative retrieval baseline.

    The baseline retrieves once with the question, then loops: an LLM extends
    a chain of thought over everything collected so far, the first generated
    sentence becomes the next retrieval query, and retrieved paragraphs are
    unioned in. It exists because `top-k` is not the only honest competitor -
    LLM query rewriting is strong and cheap, and the Phase 1 gate requires
    beating both.

    Attributes:
        per_step_k: Paragraphs retrieved per round, including the initial one.
        max_steps: LLM rounds after the initial dense retrieval. `0` turns the
            baseline into plain `top-k` of the question - the ablation switch.
        stop_phrase: Case-insensitive substring that ends the loop when it
            appears in the LLM's sentence (IRCoT's answer trigger).
        max_collected: Cap on the size of the union across rounds.
    """

    per_step_k: int = 5
    max_steps: int = 4
    stop_phrase: str = "answer is"
    max_collected: int = 20

    def __post_init__(self) -> None:
        if self.per_step_k < 1:
            raise ValueError("per_step_k must be at least 1")
        if self.max_steps < 0:
            raise ValueError("max_steps must not be negative")
        if not self.stop_phrase:
            raise ValueError("stop_phrase must not be empty")
        if self.max_collected < self.per_step_k:
            raise ValueError("max_collected must be at least per_step_k")


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings of the MuSiQue evaluation harness - the weighted objective's
    only home.

    The Phase 1 objective is `0.65 * support recall@k + 0.35 * Novelty@k`.
    Both terms are recalls against the same gold supporting set, so the
    weights act on a common [0, 1] scale - that shared denominator IS the
    normalisation rule the objective needs (closed open question #1).
    Novelty@k counts gold paragraphs the web returns in its top-k that the
    plain dense top-k (the fixed reference) does not return at all
    (closed open question #8).

    Attributes:
        dataset_url: Source of the MuSiQue-Ans dev split. The default is the
            Hugging Face mirror of the official file (byte-identical,
            CC BY 4.0); plain HTTPS, no hub client needed.
        sample_size: Questions sampled from the dev split, drawn
            deterministically from the lexicographically sorted question ids.
            The default 1000 matches the HippoRAG-comparable regime (pooled
            dedup corpus of the sampled questions' candidate paragraphs).
            `0` means the full split.
        sample_seed: Seed of the sampling RNG - part of the experiment
            identity, recorded in the index receipt.
        k_values: Cutoffs of the report table. The primary number is S@5:
            k=2 cannot hold a 4-hop question's four gold paragraphs, and 5 is
            HippoRAG's headline cutoff.
        accuracy_weight: Weight of support recall@k in the objective.
        novelty_weight: Weight of Novelty@k in the objective.
    """

    dataset_url: str = (
        "https://huggingface.co/datasets/dgslibisey/MuSiQue/"
        "resolve/main/musique_ans_v1.0_dev.jsonl"
    )
    sample_size: int = 1000
    sample_seed: int = 42
    k_values: tuple[int, ...] = (2, 5, 10)
    accuracy_weight: float = 0.65
    novelty_weight: float = 0.35

    def __post_init__(self) -> None:
        if not self.dataset_url:
            raise ValueError("dataset_url must not be empty")
        if self.sample_size < 0:
            raise ValueError("sample_size must not be negative (0 means all)")
        if not self.k_values:
            raise ValueError("k_values must not be empty")
        if any(k < 1 for k in self.k_values):
            raise ValueError("every k in k_values must be at least 1")
        if list(self.k_values) != sorted(set(self.k_values)):
            raise ValueError("k_values must be strictly ascending")
        for name in ("accuracy_weight", "novelty_weight"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value!r} must lie in [0, 1]")
        if not math.isclose(
            self.accuracy_weight + self.novelty_weight, 1.0, abs_tol=1e-9
        ):
            raise ValueError("accuracy_weight and novelty_weight must sum to 1.0")
