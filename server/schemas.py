"""Wire types. Field names match `web/src/lib/types.ts` one for one.

No camelCase translation layer: a rename between the two sides is a silent
bug waiting to happen, and there is nothing to gain from it here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from spiyweb.config import ColoredRetrievalConfig

# The inspector opens on the MEASURED WINNER, not on the library defaults -
# threshold .01 and alpha 3 are what the grid campaign selected, and a form
# that starts somewhere else invites conclusions about a configuration nobody
# ships. Those numbers used to be retyped here as literals; when `max_hop`
# moved from 6 to 8 this file, the Streamlit sidebar and the React form all
# still said 6, three copies of a number with one owner. Reading the winner
# from the library costs nothing (`spiyweb.config` is plain dataclasses) and
# makes the next change land everywhere at once.
_WINNER = ColoredRetrievalConfig().propagation


class ApiError(BaseModel):
    code: str
    message: str
    hint: str | None = None


# --- indexes ---------------------------------------------------------------


class IndexSummary(BaseModel):
    name: str
    dataset: str | None
    corpus_chunks: int | None
    propositions: int | None
    questions: int | None
    llm_model: str | None
    nli_edges: int | None
    has_results: bool
    has_per_query: bool
    modified_at: str


class LayerCount(BaseModel):
    layer: str
    edges: int
    present: bool


class ArtifactInfo(BaseModel):
    name: str
    exists: bool
    bytes: int


class IndexDetail(IndexSummary):
    meta: dict[str, object]
    layers: list[LayerCount]
    artifacts: list[ArtifactInfo]
    nodes: int


class AtomHit(BaseModel):
    id: str
    title: str
    snippet: str


# --- inspect ---------------------------------------------------------------


class QuerySpec(BaseModel):
    mode: str = "atom"  # "atom" | "text"
    node: str | None = None
    text: str | None = None
    model: str = "intfloat/multilingual-e5-large"
    device: str = "cpu"


class LayerWeightsDto(BaseModel):
    semantic: float = 0.5
    entity: float = 1.0
    structural: float = 0.3
    derivation: float = 1.0
    learned: float = 0.0

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.semantic,
            self.entity,
            self.structural,
            self.derivation,
            self.learned,
        )


class PropagationDto(BaseModel):
    seed_energy: float = _WINNER.seed_energy
    damping: float = _WINNER.damping
    threshold_ratio: float = _WINNER.threshold_ratio
    max_hop: int = _WINNER.max_hop
    max_nodes: int = _WINNER.max_nodes
    split_alpha: float = _WINNER.split_alpha
    mass_enabled: bool = _WINNER.mass.enabled


class AblationsDto(BaseModel):
    dedup: bool = True
    dedup_sigma: float = 2.0
    dedup_floor: float = 0.95
    dedup_include_seeds: bool = True
    conflict: bool = True
    polarity: bool = True


class ViewDto(BaseModel):
    max_nodes: int = 300
    max_edges: int = 1500
    edge_mode: str = "contributors"
    label_top_n: int = 15


class InspectRequest(BaseModel):
    index: str
    query: QuerySpec
    profile: str | None = None
    propagation: PropagationDto = Field(default_factory=PropagationDto)
    weights: LayerWeightsDto = Field(default_factory=LayerWeightsDto)
    ablations: AblationsDto = Field(default_factory=AblationsDto)
    view: ViewDto = Field(default_factory=ViewDto)
    seed_width: int = 5
    contact_overfetch: int = 3
    k: int = 5
    sample_size: int = 1000
    sample_seed: int = 42


class SceneNodeDto(BaseModel):
    id: str
    x: float
    y: float
    rx: float
    ry: float
    energy: float
    hop: int
    votes: int
    kind: str
    node_layer: str
    source_id: str
    polarity: int
    disputed: bool
    label: str
    title: str
    tooltip: str


class SceneEdgeDto(BaseModel):
    source: str
    target: str
    x1: float
    y1: float
    x2: float
    y2: float
    rx1: float
    ry1: float
    rx2: float
    ry2: float
    weight: float
    layer: str
    layers: list[str]
    kind: str
    tooltip: str


class RingDto(BaseModel):
    hop: int
    radius: float


class SceneDto(BaseModel):
    nodes: list[SceneNodeDto]
    edges: list[SceneEdgeDto]
    rings: list[RingDto] = []
    legend: dict[str, str]
    layer_order: list[str]
    dropped_nodes: int
    dropped_edges: int
    caption: str
    max_hop: int


class DestroyedDto(BaseModel):
    conflict: float
    conflict_events: int
    negative_seed: float
    negative_seed_events: int
    polarity: float
    polarity_events: int
    total: float


class LedgerDto(BaseModel):
    injected: float
    held: float
    dissipated: float
    destroyed: DestroyedDto
    residual: float
    residual_share: float
    mismatch: float
    tolerance: float
    balanced: bool
    exact: bool
    notes: list[str]
    dedup_cuts: int
    contact_cuts: int
    dedup_taus: list[float]
    contact_tau: float | None


class ComparisonRowDto(BaseModel):
    rank: int
    node_id: str
    title: str
    snippet: str
    score: float
    hop: int | None
    votes: int
    in_other: bool
    badges: list[str]


class ComparisonDto(BaseModel):
    web: list[ComparisonRowDto]
    baseline: list[ComparisonRowDto]
    only_in_web: list[str]
    only_in_baseline: list[str]
    overlap: list[str]
    contact_tau: float | None
    dedup_enabled: bool


class ActivationPathDto(BaseModel):
    node: str
    steps: list[str]
    hop: int
    energy: float
    converging: int
    rendered: str


class ThemeClusterDto(BaseModel):
    nodes: list[str]
    energy: float
    energy_share: float
    top_node: str


class WarningDto(BaseModel):
    kind: str
    message: str
    nodes: list[str]


class RefusalDto(BaseModel):
    stop_reason: str
    hop_depth: int
    deepest_nodes: list[str]
    text: str


class SeedDto(BaseModel):
    node: str
    title: str
    similarity: float
    energy: float


class InspectStats(BaseModel):
    total_energy: float
    node_count: int
    hop_depth: int
    stop_reason: str
    threshold: float
    elapsed_ms: float
    scene_ms: float


class InspectResponse(BaseModel):
    query_label: str
    stats: InspectStats
    scene: SceneDto
    ledger: LedgerDto
    comparison: ComparisonDto
    paths: list[ActivationPathDto]
    clusters: list[ThemeClusterDto]
    warnings: list[WarningDto]
    refusal: RefusalDto | None
    seeds: list[SeedDto]
    texts_available: bool


# --- system ----------------------------------------------------------------


class GpuDto(BaseModel):
    present: bool
    name: str | None
    vram_used_mb: int
    vram_total_mb: int
    vram_share: float
    utilization: int
    temperature: int | None
    over_budget: bool
    budget_share: float
    error: str | None


class OllamaDto(BaseModel):
    reachable: bool
    models: list[str]
    error: str | None


class CacheEntryDto(BaseModel):
    kind: str
    key: str
    approx_bytes: int


class CacheStatus(BaseModel):
    entries: list[CacheEntryDto]
    total_bytes: int
    embedder_loaded: bool


class SystemStatus(BaseModel):
    gpu: GpuDto
    ollama: OllamaDto
    cache: CacheStatus
    checked_at: str


# --- runs ------------------------------------------------------------------


class DutyDto(BaseModel):
    enabled: bool = True
    window_s: int = 2400
    cool_s: int = 300


class RunRequest(BaseModel):
    index: str
    stage: str = "evaluate"
    dataset: str = "musique"
    sample_size: int | None = None
    sample_seed: int | None = None
    force: bool = False
    no_entity_llm: bool = False
    propositions: bool = False
    nli: bool = False
    skip_iterative: bool = False
    web: str = "colored"
    device: str | None = "cpu"
    llm_model: str | None = None
    decomp_model: str | None = None
    max_colors: int | None = None
    chain_mode: str | None = None
    duty: DutyDto = Field(default_factory=DutyDto)


class RunPlan(BaseModel):
    argv: list[str]
    cwd: str
    will_skip: list[str]
    estimated_llm_calls: int | None
    warnings: list[str]
    token: str
    confirm_word: str


class RunStartRequest(BaseModel):
    request: RunRequest
    token: str
    typed: str


class RunStopRequest(BaseModel):
    typed: str


class DutyState(BaseModel):
    enabled: bool
    phase: str
    cycle: int
    window_s: int
    cool_s: int
    seconds_left: float


class RunProgress(BaseModel):
    stage: str
    llm_calls: int
    llm_calls_at_start: int
    llm_calls_per_min: float
    questions_total: int | None
    elapsed_s: float
    eta_s: float | None
    eta_basis: str | None


class RunStatus(BaseModel):
    state: str
    pid: int | None
    argv: list[str]
    cwd: str | None
    index: str | None
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    duty: DutyState | None
    progress: RunProgress | None
    log_out: str | None
    log_err: str | None


class LogTail(BaseModel):
    out: list[str]
    err: list[str]


# --- results ---------------------------------------------------------------


class MetricSet(BaseModel):
    support_recall: float
    novelty: float
    objective: float
    bridge_recall: float


class ResultsSummary(BaseModel):
    index: str
    question_count: int
    primary_k: int
    systems: dict[str, dict[str, MetricSet]]
    objective_by_hop: dict[str, dict[str, float]]
    stop_reasons: dict[str, int]
    combo: dict[str, object]
    iterative_included: bool
    extras: dict[str, object]
    modified_at: str


class PairedDiff(BaseModel):
    rival: str
    mean: float
    ci_low: float
    ci_high: float
    p: float
    significant: bool


class HopScore(BaseModel):
    hops: int
    questions: int
    scores: dict[str, float]


class BootstrapReport(BaseModel):
    index: str
    k: int
    iterations: int
    seed: int
    questions: int
    means: dict[str, float]
    bridge: dict[str, float]
    diffs: list[PairedDiff]
    by_hop: list[HopScore]
