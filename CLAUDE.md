# Spiyweb

Graph-based retrieval for RAG. Instead of cutting retrieval off at `top-k`, the
query is injected into a vector graph as an **energy seed** and spreads outward
with decay. Strongly related nodes light up first; weakly related but genuinely
connected nodes light up later, through multiple hops — like a spider web.

**Status:** Phase 1 — research prototype. Public repo, Apache-2.0.

---

## 1. Mental model

| Metaphor | Technical mapping |
|---|---|
| Box of oxygen atoms | Vector space / index |
| One atom | One node — a chunk **or** a proposition |
| Living ball entering the box | Query embedding = seed node |
| Strong bond | High-weight edge |
| Weak bond | Low-weight edge, 2–3 hops out |
| Energy reaching zero | Activation decay → natural termination |
| The spider web | Activated node set, ranked by accumulated energy |

Formally: **spreading activation** over a sparse graph, mathematically equivalent
to **Personalized PageRank** with a damping factor. It reduces to a repeated
sparse matrix–vector product, so it is cheap and numerically stable.

---

## 2. Algorithm invariants

Settled design decisions. These are the identity of the project, not tuning
knobs. Full rationale for each lives in `memory/`.

### 2.1 Propagation

| Rule | Value / behaviour | Why |
|---|---|---|
| Decay | **Multiplicative**: `E_out = E_in * damping` | A weak edge must fade faster than a strong one |
| Damping | `0.60` default, overridden per query profile | Node forwards 60% of its energy, keeps 40% |
| Split | Energy is **divided** among neighbours, **proportional to edge weight** | One neighbour gets ~60%; five equidistant neighbours get ~12% each |
| Accumulation | **Additive** — energy arriving via multiple paths sums | Gives converging evidence for free |
| Dedup renormalisation | After a duplicate's edge is zeroed, the remaining neighbour weights are **renormalised** — the duplicate's share is redistributed | Visible in the worked example; stated here as a rule, not an accident |
| Energy conservation | Dedup **redistributes** (conserves) energy; negative seeds, contradictions and negative-polarity atoms **absorb** (destroy) it; nothing else creates or destroys energy | Makes every mechanism's effect on the energy ledger auditable |
| Stop condition | **Relative energy threshold** — 15% of total injected energy (`1.5` against a seed of `10.0`) — *not* a token budget | Scales consistently when thermal residue or profiles change the injected total; keeps the mechanism pure and context-window independent |
| Safety caps | `max_nodes` and `max_hop` as hard overflow guards | The threshold is the primary stop; these only prevent surprises |
| Node mass | Proportional to length, **normalised within its own layer** (mechanism landed 2026-08-14: `core/mass.py`, gate `threshold·μ` + carry `damping^(1/μ)`; **default OFF** until measured — open question #7) | Without per-layer normalisation the proposition layer silently dies |
| Freshness | **Tie-breaker only**, never a continuous multiplier | A continuous boost quietly promotes recent-but-irrelevant content |

### 2.2 Nodes and edges

| Rule | Value / behaviour |
|---|---|
| Node layers | **Two**: chunk nodes and proposition nodes, linked to each other |
| Edge layers | **Layered hybrid** — `semantic` (cosine, seed contact + fallback only), `entity` (main hop fuel), `structural` (same doc/section/adjacent), `derivation` (chunk→proposition containment), `learned` (usage-reinforced). **A layer can be empty without being broken, and until 2026-08-26 nothing said so:** the structural layer carried zero edges in all four sealed Phase 1 indexes because every benchmark document is a single chunk, so its within-document relations cannot exist; the entity layer carried zero on any corpus under 100 chunks because `max_df_ratio * n` fell below the two mentions an entity needs to pair. The first was correct behaviour on that corpus shape and means `LayerWeights.structural = 0.3` was inert in every measured run; the second was a bug, fixed with a floor of 2. `spiyweb lint` reports both kinds now |
| Learned layer | Hebbian reinforcement lives in a **separate, disableable layer**; the base graph is never mutated |
| Consolidation | Periodic offline **pruning** of never-used edges. Node merging is deferred to Phase 2 (irreversible) |

### 2.3 Redundancy, contradiction, negation

| Rule | Value / behaviour |
|---|---|
| Redundancy | Near-duplicate neighbour: **edge weight → 0**, source idea's **vote count += 1** |
| Seed twins | Duplicate CONTACTS never each hold a seed slot: suppressed at injection (`include_seeds`) and skipped at contact selection with **elastic refill** (`contact_overfetch`) — the freed slot goes to the next distinct idea, the survivor is voted. Measured 2026-08-14 (A1): injected twins were the dominant redundancy damage channel; neighbour-level suppression alone was statistically neutral |
| Distinct sources | Two seed slots of ONE query part may never land on the same source (`DedupConfig.distinct_sources`, on by default, no-op on a single-layer index). Cosine is the right twin test for a copied passage and the wrong one for two propositions of the same passage — different sentences, never near-duplicates, yet the colour explores one passage instead of two |
| Duplicate detection | **Dynamic**, at query time, among currently active nodes |
| Harness note | The measurement harness keeps duplicate suppression **OFF by default** (`--dedup` turns it on) and every `results.json` records which way the run went. Every number sealed in the 2026-08 campaign was produced with the mechanism configured but never delivered — `retrieve()` needs BOTH a `DedupConfig` and a similarity backend, and the harness passed neither. Comparability is why the default did not change |
| Duplicate threshold | **Adaptive**, computed from the active set's similarity distribution — the computed value must be visible in the UI |
| Vote granularity | Per **document/source**, never per chunk |
| Contradiction | Modelled as **negative charge** — opposing atoms damp each other instead of reinforcing |
| Contradiction detection | **Index-time NLI** — a small multilingual NLI model runs inside `edges/` and emits negative edges; `core/` only consumes pre-marked data. Landed 2026-08-15: real wrapper `nli.py` (mDeBERTa-v3 XNLI, provisional #10 choice), high-cosine candidate pairing + `edges_nli.json` stage in the harness (`--nli`, default OFF pending measurement). Candidates must also **name the same subject** (2026-08-16, `shared_subject_pairs`): a rare-enough entity in the leading region of both texts. Cosine alone pairs same-kind different-entity texts and NLI then assumes they share a subject — the measured dominant false positive, and one no threshold separates. **Sensitivity measured 2026-08-16 on WikiContradict's 253 annotated real pairs: 31.6% caught at the shipped cut, 0% of the 63 same-passage ones (edges run between nodes, so an intra-passage contradiction is structurally invisible without the proposition layer), 9.5% end to end after the subject filter.** The mechanism is sound and the detector is not yet a working feature; better detection is Phase 2 work, not a threshold |
| Contradiction surfacing | The library emits a **template-built, LLM-free question with options** for the user |
| No answer given | **Both sides enter the context, flagged as disputed.** Never silently pick a winner |
| Negative requests | "excluding X", "without Y" → **energy-absorbing negative seed**, not a post-filter |
| Negative knowledge | Negated propositions ("X does not…") become permanently **negative-polarity atoms**: they absorb the energy of queries asserting the opposite and emit a "corpus disputes this" warning. Implemented 2026-08-14 (`core/polarity.py` + `PolarityConfig`, proportional absorption, full by default) after the first Phase 1 measurement, as planned. Detection landed 2026-08-15 (#11, owner's choice): the proposition-extraction call itself tags negated facts with a `NEG:` prefix — zero extra calls; `PropositionConfig.tag_polarity` is the ablation switch |
| Supersession | NLI contradiction + ordered timestamps on the same subject = **update, not conflict** — older atom damped, newer marked current, no user question emitted. Freshness tie-break stays separate: supersession requires NLI evidence, not mere recency. Implementation in Phase 2 |

### 2.4 Query shaping

| Rule | Value / behaviour |
|---|---|
| Multi-seed colours | The query is decomposed; each part is a differently **coloured** seed. A node where two colours meet is a **bridge** — that is where a multi-hop answer lives. Runs as an ablation in Phase 1 |
| Query profiles | `precise` / `explore` / `compare` — each carries its own damping, threshold and seed width. The caller picks; never an LLM inside `core/`. **Library default is `explore`** (`DEFAULT_PROFILE`) in `SpiywebIndex.retrieve()` and `ThermalSession` when the caller gives neither a profile nor a config — the bare `RetrievalConfig()` (5 seeds × 15% threshold) cannot forward past hop 0 and is kept only as the canonical-trace object of §2.6. A config the caller supplies is never overlaid; `retrieve()` warns if it cannot spread. Same default in the terminal since 0.1.2 |
| Conversation memory | **20–30%** of the previous turn's energy persists; follow-up questions land on warm ground. Reset is **hybrid**: caller-controlled `reset()` by default, optional topic-change auto-detection behind a config flag |

### 2.5 Output contract

`retrieve()` returns, in one structure:

- ranked nodes with accumulated energy
- vote counts per idea
- **activation paths** — passed to the LLM as explanations, not just debug data
- **theme clusters** — "these are separate themes, they intersect here"
- **confidence score** — total activated energy, activated node count and reached hop depth; the caller decides what counts as "I don't know"
- **corpus gap warnings** — two dense clusters with no bridge between them
- **structural refusal report** — when confidence is low, a template-built,
  LLM-free explanation of *why*: which entity clusters activated, where the
  missing bridge is, where the energy died, what kind of source is missing
- **contradiction records** plus the ready-made user question

The ranking is over NODES. On a two-layer index that is not the same thing as
a ranking over passages, and the difference is measurable: the harness used to
store the first `max_k` nodes, which on `musique_prop200` carried **3.81**
distinct passages instead of five, costing the coloured web
.0118 CI [.0032, .0224]. `--distinct-passages` stores down to the `max_k`-th
distinct passage instead; it is OFF by default because switching it on breaks
comparability with the sealed runs, and every `results.json` records which
window the run used. Whether the ranking itself should aggregate a passage's
propositions is a separate open question (`memory/on-kayit-onerme-enerji.md`).

### 2.6 Worked example (the canonical trace)

Seed `Q = 10.0`, damping `0.60`, threshold `1.5`.

```
HOP 0   A = 10.0 * .9/(.9+.7) = 5.60      C = 10.0 * .7/(.9+.7) = 4.40

HOP 1   A distributes 5.60 * .60 = 3.36   neighbours: A'(.95 DUPLICATE) B(.8) D(.4)
          A' -> edge zeroed, excluded from context, "idea A" VOTES = 2
          renormalise without A' -> B = 3.36 * .8/1.2 = 2.24
                                    D = 3.36 * .4/1.2 = 1.12
        C distributes 4.40 * .60 = 2.64   neighbours: D(.6) E(.3)
                                    D = 2.64 * .6/0.9 = 1.76
                                    E = 2.64 * .3/0.9 = 0.88   -> below threshold, dies

HOP 2   D = 1.12 + 1.76 = 2.88            <- converging evidence
        D distributes 2.88 * .60 = 1.73 -> F = 1.73

HOP 3   F distributes 1.73 * .60 = 1.04   -> below threshold, web stops

RESULT  A 5.60 | C 4.40 | D 2.88 | B 2.24 | F 1.73     (+ idea A: 2 votes)
```

The ledger rounds hop 0 to two decimals and carries that rounding downwards;
the unrounded values are `A 5.625 | C 4.375 | D 2.875 | B 2.25 | F 1.725`, and
those are what `tests/test_propagate.py` pins. The ordering is identical, which
is the part that matters.

`D` is never the single most similar node to `Q`, yet it ranks third. **That gap
is the entire value proposition of this project.** Any change that destroys it
is a regression, whatever the benchmark says.

---

## 3. Architecture

```
src/spiyweb/
├── core/                 # pure computation - ZERO I/O, zero heavy dependencies
│   ├── graph.py          # sparse adjacency, multi-layer edge merging, node layers
│   ├── mass.py           # node mass (D11): per-layer normalised inertia
│   ├── propagate.py      # damping, proportional split, accumulation, threshold, mass
│   ├── dedup.py          # dynamic redundancy suppression: edge -> 0, vote += 1
│   ├── colors.py         # multi-seed coloured activation, bridge detection
│   ├── conflict.py       # contradiction as negative charge
│   ├── negative.py       # negative seeds: the energy-absorbing field
│   └── polarity.py       # negative-knowledge atoms (D34): dispute records
├── edges/                # hybrid edge builders
│   ├── semantic.py       # cosine kNN (seed contact + fallback only)
│   ├── entity.py         # shared entity / concept edges (main hop fuel)
│   ├── structural.py     # same document, same section, adjacent chunk
│   ├── derivation.py     # chunk -> proposition containment links (D10)
│   ├── nli.py            # index-time NLI: emits negative (contradiction) edges
│   └── learned.py        # Hebbian reinforcement, separate and disableable
├── nodes/
│   ├── chunks.py         # chunk-level nodes
│   └── propositions.py   # proposition-level nodes (LLM extraction at index time)
├── entities.py           # hybrid entity extraction: spaCy bulk, LLM for ambiguous chunks
├── embedding.py          # e5 wrapper: role prefixes baked in, device CUDA -> MPS -> CPU
├── nli.py                # real NLI wrapper (mDeBERTa XNLI): lazy deps, spiyweb[nli] extra
├── llm.py                # LLM provider abstraction: Ollama default, free APIs optional
├── prompts.py            # prompt templates, kept apart from pipeline logic
├── profiles.py           # precise / explore / compare propagation profiles
├── thermal.py            # ThermalSession: conversation warmth across turns
├── config.py             # dataclass: damping, threshold, max_hop, max_nodes, weights
├── cli.py                # `spiyweb` command: version / index / query / lint
├── terminal.py           # ANSI colour and bars, zero dependencies
├── wizard.py             # bare `spiyweb`: guided menu, never in a pipe
├── store.py              # numpy + FAISS single-file vector store (outside core/)
│                         #   + FAISS-backed twin of the semantic edge builder
├── output.py             # result structure, paths, clusters, confidence, conflicts
├── lint.py               # corpus lint (D37): orphans, hubs, duplicates, conflicts
├── indexing.py           # corpus-agnostic build_index + artifact loaders
├── session.py            # SpiywebIndex: open an index, ask, get text back
├── trace.py              # recorded calls (D38): self-contained, JSONL-able
├── ledger.py             # energy ledger: held / dissipated / destroyed
├── scene.py              # render-agnostic layout/scene, numpy only (D2.2)
├── retrieve.py           # seed injection -> propagate -> structured result
└── evaluation/           # renamed from eval/ — avoids shadowing the Python builtin
    ├── datasets.py       # MuSiQue loader: download, deterministic sample, dedup pool
    ├── metrics.py        # support recall, Novelty@k, bridge recall, weighted S@k
    │                     #   + passage folding: propositions score as their parent
    ├── baseline.py       # plain top-k + IRCoT-style iterative retrieval
    ├── cache.py          # deterministic prompt-hash LLM cache (reproducible runs)
    ├── stats.py          # paired bootstrap CI - the protocol's interval, one copy
    ├── index.py          # corpus -> vectors + entities + edge-layer artifacts
    └── run.py            # CLI: download / index / evaluate / report
```

There is no browser face in the tree any more. The React front end, the
FastAPI viewer package (`spiyweb.viewer`, `inspect_url()`, `spiyweb view`,
the `[web]` extra) and the `server/` measurement rig were **removed on
2026-09-11** — the owner has a different interface in mind. What stayed is
everything that interface will need and that never depended on a browser:
the trace layer (`trace.py`), the energy ledger (`ledger.py`) and the
render-agnostic scene (`scene.py`, numpy only, `spiyweb[view]`).

### Boundary rules — the single most important thing in this file

1. **`core/` knows nothing about the outside world.** No vector store, no LLM, no
   embedding model, no filesystem, no network. Arrays and a config in, numbers
   out. This is what keeps every later phase from becoming a rewrite.
2. **`edges/` and `nodes/` are plural on purpose.** Layer choices live in config,
   not in code. Adding a layer must never require touching `core/`.
3. **`evaluation/` is the Phase 1 product.** It becomes the regression suite
   later. It is never throwaway code. (Named `evaluation`, not `eval`, to avoid
   shadowing the Python builtin.)
4. **`pip install spiyweb` pulls in nothing, and nothing is imported
   eagerly.** `dependencies = []` is the packaging-level face of the
   core-purity rule; numpy, FAISS, torch and spaCy all arrive through
   extras, and `import spiyweb` touches none of them. The wheel job
   measures this on the artifact, not on the checkout. Any interface that
   replaces the removed browser face lives under the same rule: it may be
   an extra, it may be a separate package, it is never a dependency — and
   whatever it draws, it draws from `scene.py`, `ledger.py` and `trace.py`
   rather than from a second copy of the layout, so one query produces one
   picture no matter what asks for it.
5. **Contradiction question templates live outside `core/`.** The library ships
   them so callers do not rewrite them, but the core only produces structured
   conflict data.

---

## 4. Phase 1 settings

| Item | Decision |
|---|---|
| Objective | **65% multi-hop accuracy + 35% serendipity** (weighted) |
| Serendipity metric | **Novelty@k** — nodes the web returns that `top-k` never returns, and that are still relevant |
| Extra metric | **Bridge-node recall** — where does the required intermediate document rank? Standard `recall@k` does not reward this |
| Benchmark | **MuSiQue** |
| Baselines | **`top-k`** and **iterative retrieval** (LLM query rewriting) — the Phase 1 gate requires beating **both** |
| Reference comparison | **HippoRAG** results reported alongside — informative, not a gate criterion |
| Entity extraction | **spaCy + LLM hybrid** — spaCy for the bulk, LLM for ambiguous cases |
| Embedding | **multilingual-e5-large** (Turkish + English) |
| Vector store | **numpy + FAISS**, single file, no server |
| Layer weights | Start by hand: `semantic .5 / entity 1.0 / structural .3`, then a small grid search |
| Seed width | **5 atoms** |
| Environment | **Python 3.11 + uv** |
| Platforms | **macOS + Windows + Linux** — no OS-specific paths or calls; device order CUDA → MPS → CPU |
| LLM provider | **Local-first (Ollama)**; free APIs (Gemini, OpenRouter, Groq) optional via config — the provider abstraction lives outside `core/` |
| Packaging | **`src/spiyweb/` layout**; `eval/` → **`evaluation/`**; no browser face in the wheel (removed 2026-09-11) |
| Repo / license | **Public from day one, Apache-2.0** |

Because the repo is public from day one, `CLAUDE.local.md` must stay in
`.gitignore` permanently.

---

## 5. Roadmap

| Phase | Contents | Gate to leave it |
|---|---|---|
| **1. Simple working version** | Graph, propagation, dedup, conflicts, eval harness, dev UI | Beat both baselines on MuSiQue by a meaningful margin |
| **2. Browser face (UI)** | **CLOSED 2026-08-26.** Shipped: a stable public API (101 declared names, snapshot-tested, zero dependencies proved on the artifact), corpus-agnostic indexing and `SpiywebIndex`, a self-contained trace layer, a browser face that ran from the installed wheel (`inspect_url()`; **removed 2026-09-11**, the owner has a different interface in mind), the `spiyweb` terminal command, version `0.1.0`, and **corpus lint** — the plan-B diagnostic of orphans, hubs, duplicates, contradictions and empty layers. Not done: the TestPyPI upload (the owner's token), and four research-queue items with no signal — **contradiction detection** (measured in Phase 1 at 31.6% recall on 253 annotated pairs, 0% on same-passage ones, 9.5% end to end; the mechanism is correct and the detector is not yet a working feature, and fixing it needs the proposition layer rather than a threshold), the learned layer's useful form, query-time latency, and colour-count calibration | Signal, for any queue item |
| **3. Framework / ecosystem** | Ingestion, LLM calls, orchestration; distributable as a skill | — |

The order above is a **tentative revision** from the owner and may change. Two
things do not change: gates are triggered by signals, never by the calendar; and
Phase 3 abstractions must never leak backwards into Phase 1 or 2 code.

---

## 6. Code standards

- **Python**, type annotations mandatory on every function signature.
- `ruff` for lint and format. All identifiers, comments and docstrings in
  **English**.
- Device selection order is fixed everywhere a tensor is placed:
  **CUDA → MPS → CPU**.
- No magic numbers in code. Every tunable lives in `config.py` as a dataclass
  field with a documented default — the UI builds its sliders from these.
- Every mechanism in section 2 must be **individually disableable** from config.
  Ablation is impossible otherwise, and ablation is how this project proves
  itself.
- No secrets in source. Read from environment (`os.environ.get`) or `.env`.
  Never log credentials, not even at `DEBUG`.

### Verification

```
ruff check .
ruff format --check .
pytest
```

All three must be clean before any change is considered done.

---

## 7. Things not to do

- Do not add an LLM call to `core/`.
- Do not replace multiplicative decay with a fixed subtraction "for simplicity".
- Do not turn duplicate suppression into plain MMR — dropping duplicates loses
  the vote signal, which is the project's most original idea.
- Do not compute node mass from raw length across layers; propositions are short
  by definition and would never activate.
- Do not silently resolve a contradiction by picking the stronger side.
- Do not let the learned layer write into the base graph.
- Do not evaluate on a private corpus and call it evidence.
- Do not commit, push, or open PRs unless explicitly asked.

---

## 8. Known open risks

- **Prior art.** PPR-based graph RAG already exists (HippoRAG, GraphRAG,
  LightRAG, RAPTOR). Differentiation rests on dynamic redundancy-to-vote
  conversion, coloured multi-seed bridging, and the honesty outputs — not on the
  propagation itself.
- **Hub penalty.** Proportional splitting punishes information-dense nodes.
  Softening option: weight by `sim ** alpha`.
- **Repetition is not truth.** Vote counts measure corpus support. Document-level
  counting limits but does not eliminate this.
- **Index cost.** Proposition extraction and hybrid entity extraction both need
  LLM calls at index time. Fine on a fixed benchmark corpus; must be measured
  before it is pointed at a real one.
- **Learned-layer drift.** Reinforcement without a forgetting factor collapses
  the graph toward whatever was asked most often.

The original design document (`docs/specs/2026-08-10-spiyweb-design.md`, frozen
2026-08-25 as a historical snapshot) and the decision log (`memory/`) are kept
**locally and out of the repository** — this file is the public ground truth,
and where the frozen spec disagrees with it, this file wins. A reference to
either from here points at something a clone will not contain; that is
deliberate, not a broken link.
