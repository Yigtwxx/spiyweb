/**
 * Wire types, mirroring `server/schemas.py` field for field.
 *
 * Names stay snake_case on purpose. A camelCase translation layer buys
 * nothing here and is a place for silent renames to hide.
 */

export type ApiError = { code: string; message: string; hint: string | null };

export type IndexSummary = {
  name: string;
  dataset: string | null;
  corpus_chunks: number | null;
  propositions: number | null;
  questions: number | null;
  llm_model: string | null;
  nli_edges: number | null;
  has_results: boolean;
  has_per_query: boolean;
  modified_at: string;
};

export type LayerCount = { layer: string; edges: number; present: boolean };
export type ArtifactInfo = { name: string; exists: boolean; bytes: number };

export type IndexDetail = IndexSummary & {
  meta: Record<string, unknown>;
  layers: LayerCount[];
  artifacts: ArtifactInfo[];
  nodes: number;
};

export type AtomHit = { id: string; title: string; snippet: string };

export type QuerySpec = {
  mode: "atom" | "text";
  node?: string | null;
  text?: string | null;
  model?: string;
  device?: "cpu" | "cuda" | "mps";
};

export type LayerWeightsDto = {
  semantic: number;
  entity: number;
  structural: number;
  derivation: number;
  learned: number;
};

export type PropagationDto = {
  seed_energy: number;
  damping: number;
  threshold_ratio: number;
  max_hop: number;
  max_nodes: number;
  split_alpha: number;
  mass_enabled: boolean;
};

export type AblationsDto = {
  dedup: boolean;
  dedup_sigma: number;
  dedup_floor: number;
  dedup_include_seeds: boolean;
  conflict: boolean;
  polarity: boolean;
};

export type ViewDto = {
  max_nodes: number;
  max_edges: number;
  edge_mode: "contributors" | "induced";
  label_top_n: number;
};

export type InspectRequest = {
  index: string;
  query: QuerySpec;
  profile: string | null;
  propagation: PropagationDto;
  weights: LayerWeightsDto;
  ablations: AblationsDto;
  view: ViewDto;
  seed_width: number;
  contact_overfetch: number;
  k: number;
  sample_size: number;
  sample_seed: number;
};

export type NodeKind = "seed" | "bridge" | "activated" | "suppressed";

export type SceneNodeDto = {
  id: string;
  x: number;
  y: number;
  rx: number;
  ry: number;
  energy: number;
  hop: number;
  votes: number;
  kind: NodeKind;
  node_layer: string;
  source_id: string;
  polarity: number;
  disputed: boolean;
  label: string;
  title: string;
  tooltip: string;
};

export type SceneEdgeDto = {
  source: string;
  target: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  rx1: number;
  ry1: number;
  rx2: number;
  ry2: number;
  weight: number;
  layer: string;
  layers: string[];
  kind: "active" | "suppressed";
  tooltip: string;
};

/** One guide circle of the hop layout, radius in normalised units. */
export type RingDto = {
  hop: number;
  radius: number;
};

export type SceneDto = {
  nodes: SceneNodeDto[];
  edges: SceneEdgeDto[];
  /** Sent by the server so the guides and the atoms share one rule. */
  rings?: RingDto[];
  legend: Record<string, string>;
  layer_order: string[];
  dropped_nodes: number;
  dropped_edges: number;
  caption: string;
  max_hop: number;
};

export type DestroyedDto = {
  conflict: number;
  conflict_events: number;
  negative_seed: number;
  negative_seed_events: number;
  polarity: number;
  polarity_events: number;
  total: number;
};

export type LedgerDto = {
  injected: number;
  held: number;
  dissipated: number;
  destroyed: DestroyedDto;
  residual: number;
  residual_share: number;
  mismatch: number;
  tolerance: number;
  balanced: boolean;
  exact: boolean;
  notes: string[];
  dedup_cuts: number;
  contact_cuts: number;
  dedup_taus: number[];
  contact_tau: number | null;
};

export type ComparisonRowDto = {
  rank: number;
  node_id: string;
  title: string;
  snippet: string;
  score: number;
  hop: number | null;
  votes: number;
  in_other: boolean;
  badges: string[];
};

export type ComparisonDto = {
  web: ComparisonRowDto[];
  baseline: ComparisonRowDto[];
  only_in_web: string[];
  only_in_baseline: string[];
  overlap: string[];
  contact_tau: number | null;
  dedup_enabled: boolean;
};

export type ActivationPathDto = {
  node: string;
  steps: string[];
  hop: number;
  energy: number;
  converging: number;
  rendered: string;
};

export type ThemeClusterDto = {
  nodes: string[];
  energy: number;
  energy_share: number;
  top_node: string;
};

export type WarningDto = {
  kind: "gap" | "dispute";
  message: string;
  nodes: string[];
};

export type RefusalDto = {
  stop_reason: string;
  hop_depth: number;
  deepest_nodes: string[];
  text: string;
};

export type SeedDto = {
  node: string;
  title: string;
  similarity: number;
  energy: number;
};

export type InspectStats = {
  total_energy: number;
  node_count: number;
  hop_depth: number;
  stop_reason: "threshold" | "max_hop" | "max_nodes";
  threshold: number;
  elapsed_ms: number;
  scene_ms: number;
};

export type InspectResponse = {
  query_label: string;
  stats: InspectStats;
  scene: SceneDto;
  ledger: LedgerDto;
  comparison: ComparisonDto;
  paths: ActivationPathDto[];
  clusters: ThemeClusterDto[];
  warnings: WarningDto[];
  refusal: RefusalDto | null;
  seeds: SeedDto[];
  texts_available: boolean;
};

export type GpuDto = {
  present: boolean;
  name: string | null;
  vram_used_mb: number;
  vram_total_mb: number;
  vram_share: number;
  utilization: number;
  temperature: number | null;
  over_budget: boolean;
  budget_share: number;
  error: string | null;
};

export type OllamaDto = {
  reachable: boolean;
  models: string[];
  error: string | null;
};
export type CacheEntryDto = { kind: string; key: string; approx_bytes: number };
export type CacheStatus = {
  entries: CacheEntryDto[];
  total_bytes: number;
  embedder_loaded: boolean;
};
export type SystemStatus = {
  gpu: GpuDto;
  ollama: OllamaDto;
  cache: CacheStatus;
  checked_at: string;
};

export type DutyDto = { enabled: boolean; window_s: number; cool_s: number };

export type RunRequest = {
  index: string;
  stage: "download" | "index" | "evaluate" | "report" | "all";
  dataset: "musique" | "2wiki" | "hotpotqa";
  sample_size: number | null;
  sample_seed: number | null;
  force: boolean;
  no_entity_llm: boolean;
  propositions: boolean;
  nli: boolean;
  skip_iterative: boolean;
  web: "colored" | "plain";
  device: "cpu" | "cuda" | "mps" | null;
  llm_model: string | null;
  decomp_model: string | null;
  max_colors: number | null;
  chain_mode: "none" | "single" | "sequential" | null;
  duty: DutyDto;
};

export type RunPlan = {
  argv: string[];
  cwd: string;
  will_skip: string[];
  estimated_llm_calls: number | null;
  warnings: string[];
  token: string;
  confirm_word: string;
};

export type DutyState = {
  enabled: boolean;
  phase: "window" | "cooling";
  cycle: number;
  window_s: number;
  cool_s: number;
  seconds_left: number;
};

export type RunProgress = {
  stage: string;
  llm_calls: number;
  llm_calls_at_start: number;
  llm_calls_per_min: number;
  questions_total: number | null;
  elapsed_s: number;
  eta_s: number | null;
  eta_basis: string | null;
};

export type RunState =
  | "idle"
  | "running"
  | "stopping"
  | "stale"
  | "finished"
  | "failed"
  // A measurement started outside this server, from a terminal. The page must
  // not claim the machine is idle while the GPU is saturated.
  | "external";

export type RunStatus = {
  state: RunState;
  pid: number | null;
  argv: string[];
  cwd: string | null;
  index: string | null;
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  duty: DutyState | null;
  progress: RunProgress | null;
  log_out: string | null;
  log_err: string | null;
};

export type LogTail = { out: string[]; err: string[] };

export type MetricSet = {
  support_recall: number;
  novelty: number;
  objective: number;
  bridge_recall: number;
};

export type ResultsPayload = {
  index: string;
  question_count: number;
  primary_k: number;
  systems: Record<string, Record<string, MetricSet>>;
  objective_by_hop: Record<string, Record<string, number>>;
  stop_reasons: Record<string, number>;
  combo?: Record<string, string | number>;
  iterative_included: boolean;
  questions_with_bridge?: number;
  bridge_contains_gold?: number;
  colors_per_question?: Record<string, number>;
  meta: Record<string, unknown>;
  modified_at: string;
};

export type PairedDiff = {
  rival: string;
  mean: number;
  ci_low: number;
  ci_high: number;
  p: number;
  significant: boolean;
};

export type BootstrapReport = {
  index: string;
  k: number;
  iterations: number;
  seed: number;
  questions: number;
  means: Record<string, number>;
  bridge: Record<string, number>;
  diffs: PairedDiff[];
  by_hop: { hops: number; questions: number; scores: Record<string, number> }[];
};

// --- the trace viewer (Faz 2.5) -------------------------------------------

/**
 * What the server on the other end of this page actually is.
 *
 * One bundle serves two products: the repository's measurement rig, which
 * owns `data/` and supervises benchmark runs, and `spiyweb.viewer`, which
 * ships in the wheel and shows an application its own recorded calls. The
 * page asks rather than guesses — probing for a 404 and inferring from it
 * would break the first time an endpoint was renamed.
 */
export interface Capabilities {
  version: string;
  /** `"rig"`, `"store"` (a live index's ring buffer) or `"file"` (JSONL). */
  mode: string;
  origin: string;
  /** Whether a new question can be asked, or only recorded ones read. */
  live: boolean;
  colored: boolean;
  count: number;
  /** Whether the measurement-run views exist on this server. */
  runs: boolean;
  bundle: boolean;
}

/** One row of the trace list — never the nodes, edges or passage text. */
export interface TraceSummary {
  trace_id: string;
  sequence: number;
  recorded_at: string;
  kind: string;
  query: string;
  node_count: number;
  total_energy: number;
  hops_used: number;
  stop_reason: string;
  elapsed_ms: number;
  dedup_mode: string;
  edge_count: number;
  bridge_count: number;
  balanced: boolean | null;
}

export interface TraceListDto {
  total: number;
  offset: number;
  limit: number;
  traces: TraceSummary[];
}

export interface TraceNodeDto {
  id: string;
  source_id: string;
  layer: string;
  energy: number;
  hop: number;
  votes: number;
  text: string;
  seed_similarity: number | null;
  polarity: number;
  disputed: boolean;
  suppressed_by: string;
}

export interface TracePathDto {
  node: string;
  steps: string[];
  hop: number;
  energy: number;
  converging: number;
}

export interface TraceClusterDto {
  nodes: string[];
  energy: number;
  energy_share: number;
  top_node: string;
  colors: string[];
}

export interface TraceEventDto {
  kind: string;
  node: string;
  other: string;
  amount: number;
  hop: number;
}

/** The flattened energy ledger a record carries with it. */
export interface TraceLedgerDto {
  injected: number;
  held: number;
  dissipated: number;
  destroyed_conflict: number;
  destroyed_negative_seed: number;
  destroyed_polarity: number;
  residual: number;
  mismatch: number;
  tolerance: number;
  balanced: boolean;
  exact: boolean;
  dedup_cuts: number;
  notes: string[];
}

/** One recorded retrieval, complete enough to be drawn on its own. */
export interface TraceRecordDto {
  schema_version: number;
  trace_id: string;
  sequence: number;
  recorded_at: string;
  kind: string;
  query: string;
  parts: Record<string, string>;
  index: string;
  profile: string;
  elapsed_ms: number;
  nodes: TraceNodeDto[];
  edges: { source: string; target: string; weight: number }[];
  paths: TracePathDto[];
  clusters: TraceClusterDto[];
  events: TraceEventDto[];
  bridges: Record<string, string[]>;
  stop_reason: string;
  hops_used: number;
  injected_energy: number;
  threshold: number;
  total_energy: number;
  node_count: number;
  dedup_mode: string;
  dedup_thresholds: number[];
  settings: Record<string, number | string | boolean>;
  ledger: TraceLedgerDto | null;
  edges_truncated: boolean;
}
