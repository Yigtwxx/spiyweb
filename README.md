# Spiyweb

[![ci](https://github.com/Yigtwxx/spiyweb/actions/workflows/ci.yml/badge.svg)](https://github.com/Yigtwxx/spiyweb/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Graph-based retrieval for RAG, built on spreading activation — like a spider
web.**

Instead of cutting retrieval off at `top-k`, the query is injected into a
vector graph as an **energy seed** and spreads outward with decay. Strongly
related nodes light up first; weakly related but genuinely connected nodes
light up later, through multiple hops. The answer is built from the whole web,
not from one cluster of near-duplicates.

> **Status: Phase 1 — measured on three benchmarks, and the gate is not
> passed yet.** The web beats both baselines on MuSiQue and 2WikiMultihopQA
> and **loses to iterative retrieval on HotpotQA**. The numbers are below,
> including the one that does not flatter the project. The design has been
> public from day one so the idea can be judged — and challenged — early.

---

## Why

Classic RAG retrieval has two structural weaknesses:

1. **A hard cutoff has no notion of indirect relevance.** When an answer is
   distributed across several documents, no single one of them may be among
   the most similar chunks — so `top-k` never sees it.
2. **Repetition is treated as content.** Ten near-identical chunks fill ten
   context slots and tell the model one thing, badly.

## How it works

Formally: **spreading activation** over a sparse multi-layer graph,
mathematically equivalent to Personalized PageRank with a damping factor. It
reduces to repeated sparse matrix–vector products — cheap and numerically
stable.

```
HOP 0   Q = 10.0  ->  first contact (cosine):   A = 5.60    C = 4.40
HOP 1   A -> B 2.24, D 1.12      C -> D 1.76, E 0.88 (dies)
HOP 2   D = 1.12 + 1.76 = 2.88   <- converging evidence
HOP 3   energy below threshold, the web stops on its own

RESULT  A 5.60 | C 4.40 | D 2.88 | B 2.24 | F 1.73
```

`D` is never the single most similar node to the query, yet it ranks third —
because two independent weak paths converged on it. That promotion is the
entire value proposition.

## Try it

Three commands, from an empty directory to what the retrieval did:

```bash
pip install "spiyweb[index]"
python -m spacy download en_core_web_sm

spiyweb index docs/ my-index          # a directory of .txt/.md -> an index
spiyweb query my-index "what happened afterwards"
spiyweb lint my-index                 # what is wrong with the CORPUS
```

`lint` is the diagnostic that needs no query: it reads the graph's shape and
reports islands nothing bridges, hubs that grind arriving energy into dust,
near-identical passages that will compete for one seed slot, and the sources
that contradict the rest of the corpus.

`spiyweb version` says which extras are installed and prints the `pip install`
line for the ones that are not — it works on a bare install, which is exactly
when you need it to.

The core itself is pure Python with no dependencies at all. Similarities come
from the caller; `core/` never computes them.

```bash
git clone https://github.com/Yigtwxx/spiyweb && cd spiyweb
uv sync --group dev && uv run pytest
```

```python
from spiyweb import Graph, propagate

graph = Graph.from_edges(
    [
        ("A", "A_dup", 0.0),  # near duplicate, edge suppressed by dedup
        ("A", "B", 0.8),
        ("A", "D", 0.4),
        ("C", "D", 0.6),
        ("C", "E", 0.3),
        ("D", "F", 0.5),
    ]
)
result = propagate(graph, seeds={"A": 0.9, "C": 0.7})

[(node, round(energy, 3)) for node, energy in result.ranked()]
# [('A', 5.625), ('C', 4.375), ('D', 2.875), ('B', 2.25), ('F', 1.725)]
result.activations["D"].contributors  # ('A', 'C') — converging evidence
result.stop_reason  # 'threshold' — the web stopped itself
```

`E` is missing because 0.875 of energy reached it against a floor of 1.5, and
`A_dup` is missing because its edge was suppressed and its share redistributed.
Neither outcome came from a result-count parameter.

## Bring your own corpus

```bash
pip install "spiyweb[index]"
```

```python
import spiyweb
from spiyweb.indexing import DocumentInput, TextUnit, build_index
from spiyweb.indexing import SentenceTransformerEmbedder, load_spacy_pipeline

docs = [
    DocumentInput(source_id="handbook", units=(TextUnit(text=part) for part in parts))
    for parts in my_documents
]
build_index(
    docs,
    "data/mydocs",
    embedder=SentenceTransformerEmbedder(),
    entity_pipeline=load_spacy_pipeline(),
)

with spiyweb.open_index("data/mydocs") as index:
    answer = index.retrieve("who signed off on the change?", profile="precise")

    for passage in answer.passages:
        print(f"{passage.energy:5.2f}  {passage.votes} votes  {passage.text[:70]}")

    print(answer.profile)  # which profile ran; "explore" when you named none
    print(answer.confidence)  # total energy, node count, hop depth
    print(answer.dedup_mode)  # which duplicate rules actually ran
    for path in answer.paths():  # how the energy reached each node
        print(path)
```

There is no `k`. The web stops when its energy falls below the threshold -
that self-termination is the argument against `top-k`, so a `k=` parameter
here would quietly reintroduce the thing being argued against. Slice
`answer.passages` if you want fewer.

`profile` is optional. Name none and pass no `config`, and the library runs
`explore` (`spiyweb.DEFAULT_PROFILE`) - the bare `RetrievalConfig()` carries
the canonical worked example above and cannot spread past five seeds, so it
is what you get only when you build it yourself, and `retrieve()` warns if
what you built cannot leave hop 0.

`open_index` wires duplicate suppression correctly, which is not a detail:
the mechanism needs a config AND a similarity backend, and this project's own
measurement campaign ran with it silently off for want of the second half.
`answer.dedup_mode` is the receipt.

## Public API

`import spiyweb` is the query-time contract: everything in `spiyweb.__all__`
and nothing else. `spiyweb.indexing` is the index-time contract, on the same
terms; it imports with nothing installed, and only the FAISS-bound names ask
for `pip install "spiyweb[store]"`. Anything reached through a submodule path
(`spiyweb.core.*`, `spiyweb.evaluation.*`) is internal and may change without
notice. Both surfaces are snapshot-tested, and changes to them are written
down in [CHANGELOG.md](CHANGELOG.md).

## What is different

Spreading activation over graphs is not new (HippoRAG, GraphRAG, LightRAG,
RAPTOR). Spiyweb's claimed differentiators are elsewhere:

- **Redundancy becomes a vote, not noise.** When a near-duplicate is found
  during propagation, its edge is severed and its energy share redistributed —
  and the surviving idea's **vote count** goes up. Repetition turns into
  corpus-support evidence instead of burning context slots. To our knowledge
  this dynamic dedup-to-vote conversion has no published equivalent.
- **Honesty outputs.** Every retrieval returns a **confidence score** (total
  energy, node count, hop depth), **corpus-gap warnings** (two dense clusters
  with no bridge), **contradiction records** with a ready-made, LLM-free
  question for the user, and **activation paths** as explanations the LLM can
  cite. The retriever can say "I don't know, and here is why."
- **Coloured multi-seed bridging.** A decomposed query injects differently
  coloured seeds; a node where two colours meet is a **bridge** — exactly
  where a multi-hop answer lives.
- **The web stops itself.** Termination is a relative energy threshold, not a
  "return N results" parameter.

## Traces

Every query an open index answers is recorded as a self-contained trace:
the activated subgraph, the passages' text, the energy ledger and the
settings the call ran with. Nothing else is needed to read it back - no
index, no store, no dependency.

Traces are held in memory (the last 200) and cost no disk unless asked:

```python
index = spiyweb.open_index("my-index", trace=spiyweb.TraceConfig(directory="traces"))
```

That writes JSONL, and a machine that holds no index can read it back:

```python
from spiyweb import load_traces

for record in load_traces("traces/traces.jsonl"):
    print(record.query, record.profile, record.ledger)
```

The layout of a recorded call - hop rings, layers, a side-by-side against
plain `top-k` - lives in `spiyweb.scene` (numpy only, `spiyweb[view]`), so
one query produces one picture no matter what draws it. The browser face
that used to draw it was removed after 0.1.2; what replaces it is open.

## Phase 1 plan

| Item | Decision |
|---|---|
| Benchmark | MuSiQue (multi-hop) |
| Gate | Beat **both** baselines — plain `top-k` and iterative retrieval — by a meaningful margin |
| Reference | HippoRAG results reported alongside |
| Metrics | 65% multi-hop accuracy + 35% Novelty@k, plus bridge-node recall |
| Embedding | multilingual-e5-large |
| Store | numpy + FAISS, single file |
| LLM (index-time only) | Local-first (Ollama); free APIs optional |
| Environment | Python 3.11 + uv · macOS / Windows / Linux |

Phase 1 also succeeds if it *fails* clearly: a reliable negative number is a
better outcome than a polished library built on an unmeasured assumption.

## Results so far

S@5 = 0.65 · support recall + 0.35 · Novelty@5. 1000 questions per run,
paired bootstrap intervals, winning configuration applied **unchanged** to
every dataset after the first.

| dataset | SPIYWEB | iterative (IRCoT-style) | plain top-k | verdict |
|---|---|---|---|---|
| MuSiQue (tuning, seed 42) | **.5094** | .4631 | .3090 | passes, +.046 CI [+.030, +.062] |
| MuSiQue (confirmation, seed 123) | **.5073** | .4420 | .3046 | passes, +.065 CI [+.048, +.082] |
| 2WikiMultihopQA | **.7130** | .687 | .468 | passes, +.026 CI [+.016, +.037] |
| **HotpotQA** | .6228 | **.6428** | .5623 | **fails**: −.020 CI [−.032, −.009] |

![Phase 1 measurement record](Stats/phase1_results.png)

**[`RESULTS.md`](RESULTS.md) is the complete record** — one table with every
number Phase 1 produced: the gate, the metric decomposition, the per-hop
breakdown, all five failed rescue rounds, every mechanism ablation, the
contradiction-detection measurements, and the seven limits that go with them.

The gate asks for both baselines, so HotpotQA is a failure, not a footnote.
The diagnosis is that the advantage is **depth-dependent**: 74.5% of
HotpotQA questions already have all their gold in the dense top-5, where
Novelty@5 is 0 by construction and spreading can only displace.

**Five pre-registered attempts to close that gap were all recorded as
negative**, and Phase 1 closed on that. Four worked in the ranking layer
(a confidence gate, blending, novelty-free slots); the fifth moved inside
propagation, letting the decomposition's colour count pick the query profile.
It won +.0025 CI [+.0005,+.0050] on the tuning set, was not confirmed on the
seed-123 set, and changed HotpotQA by **exactly zero** — not one question's
window moved. One earlier attempt helped HotpotQA and hurt the deeper sets,
which is exactly the trade the gate is meant to refuse.

So the honest headline is a result with a condition attached: the advantage
over both baselines is real on the deeper sets and does not transfer to a
benchmark that is entirely 2-hop.

Cost: ~2.3 LLM calls per question at query time, against roughly 4 for the
iterative baseline.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — condensed engineering ground truth: the settled
  invariants, the architecture boundaries, and the rules a change must not
  break. The full design specification and the decision log (every choice
  with its rationale and its rejected alternatives) are kept privately.

## Contributing

Design feedback is welcome right now — especially prior art for the
dedup-to-vote mechanism. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE) © 2026 Yigit Erdogan
