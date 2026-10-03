"""The measurement pipeline: download -> index -> evaluate -> report.

`python -m spiyweb.evaluation.run all` produces the Phase 1 number. Every
stage is resumable (artifacts skip themselves), every per-question outcome is
persisted to `per_query.jsonl` (the raw material for hop-stratified analysis
and for open question #4's stop_reason data), and the report renders the
three-system table with HippoRAG's published numbers as a clearly labelled
reference row (D29: reported, never reproduced here, never a gate).

Only `main()` constructs real components (embedder, spaCy pipeline, LLM
client); every stage function takes them injected, so the whole pipeline runs
under fakes in CI.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from spiyweb.config import (
    ColoredRetrievalConfig,
    ConflictConfig,
    DedupConfig,
    EvaluationConfig,
    IterativeBaselineConfig,
    LayerWeights,
    RetrievalConfig,
)
from spiyweb.core.conflict import conflict_adjacency
from spiyweb.evaluation.baseline import iterative_retrieve, topk_retrieve
from spiyweb.evaluation.datasets import (
    HOTPOTQA_DEV_URL,
    TWOWIKI_DEV_URL,
    download_musique,
    load_2wiki,
    load_dataset,
    load_hotpotqa,
)
from spiyweb.evaluation.decompose import (
    decompose_question,
    extract_intermediate_answer,
)
from spiyweb.evaluation.index import (
    IndexPaths,
    build_index,
    load_graph,
    load_nli_edges,
    load_similarity,
    load_store,
)
from spiyweb.evaluation.metrics import (
    bridge_recall_at_k,
    nodes_for_k_passages,
    novelty_at_k,
    support_recall_at_k,
    weighted_objective,
)
from spiyweb.retrieve import QUESTION_COLOR, _dedup_mode, retrieve, retrieve_colored

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from spiyweb.core.dedup import SimilarityFn
    from spiyweb.core.graph import Graph
    from spiyweb.embedding import Embedder
    from spiyweb.evaluation.datasets import MusiqueDataset, MusiqueQuestion
    from spiyweb.llm import LLMClient
    from spiyweb.retrieve import ColoredRetrievalResult, SeedSource

# Published reference, never reproduced here and never a gate (D29).
# Source: HippoRAG (ColBERTv2), arXiv:2405.14831, Table 2, MuSiQue column.
HIPPORAG_REFERENCE = {"R@2": 0.409, "R@5": 0.519}

DENSE_OVERFETCH = 5
"""How much deeper the dense side is fetched under the passage window.

A dense top-10 on a two-layer index can be ten propositions of three
passages; asking for `max_k` alone would hand the window nothing to work
with. Five times is generous against the measured worst case (ten stored
nodes carried 3.81 distinct passages) and costs nothing - the store search
is exact and in memory."""


def evaluate_questions(
    dataset: MusiqueDataset,
    paths: IndexPaths,
    *,
    embedder: Embedder,
    llm: LLMClient | None = None,
    decomp_llm: LLMClient | None = None,
    retrieval_config: RetrievalConfig | None = None,
    colored_config: ColoredRetrievalConfig | None = None,
    iterative_config: IterativeBaselineConfig | None = None,
    eval_config: EvaluationConfig | None = None,
    weights: LayerWeights | None = None,
    web_mode: str = "colored",
    iterative: bool = True,
    conflict: ConflictConfig | None = None,
    dedup: DedupConfig | None = None,
    distinct_passages: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Run all systems over every question; write per-query records + results.

    The default web system is the coloured, sequentially chained one - the
    measured winner of the 2026-08-14 campaign, tour 12 (~2.6 LLM calls per
    question); pass `web_mode="plain"` for the single-seed web. `llm=None`
    skips the iterative baseline (the report says so instead of showing an
    empty column) and forces the plain web, since decomposition needs an LLM
    too. `decomp_llm` is the decomposition model's client (tour-12 winner:
    qwen3.5:9b); when omitted, `llm` decomposes as well.

    `iterative=False` drops ONLY the baseline and leaves the coloured web
    intact. Dropping the baseline used to mean passing `llm=None`, which also
    silently demoted the winner to the plain web - so a run meant to save the
    baseline's ~4 calls per question measured a different system than the one
    it named.

    The work is phased, not per-question: every LLM phase completes before
    the next embedding phase starts, so on an 8 GB GPU the LLM server and the
    embedder never fight over VRAM within a phase.

    `conflict` activates the contradiction mechanism (D14/D26) over the
    index's pre-marked negative edges (`edges_nli.json`); an index without
    that artifact contributes no edges, so passing a config is then a no-op.

    `distinct_passages` stores a ranking down to the node that completes its
    `max_k`-th distinct PASSAGE, instead of cutting at `max_k` nodes. On a
    chunk-only index the two are identical; on a two-layer index they are
    not, and the difference was measured on 2026-08-16: ten stored nodes
    carried 3.81 distinct passages, so `passages_at_k` was scoring a
    handicapped ranking and the coloured web lost .0118 CI [.0032,.0224] to
    the cut alone. Default `False` keeps the sealed runs comparable.

    `dedup` activates redundancy suppression (D6): the harness then also
    builds the node-pair similarity `retrieve()` requires, since the
    mechanism stays off unless BOTH arrive. Default `None` = off, which is
    how every sealed number from 2026-08-14 onwards was produced - stated
    here because "the config exists" was never the same as "the mechanism
    ran", and the closing report has to say which one it means. Turning it on
    changes what the web returns, so a run that enables it is not comparable
    with the sealed ones.
    """
    if web_mode not in ("colored", "plain"):
        raise ValueError(f"web_mode {web_mode!r} must be 'colored' or 'plain'")
    cfg = eval_config if eval_config is not None else EvaluationConfig()
    store = load_store(paths)
    graph = load_graph(paths, weights)
    # Built only when asked: the matrix is the whole corpus in memory, and a
    # run that does not dedup has no use for it.
    similarity = load_similarity(paths) if dedup is not None else None
    if dedup is not None:
        log("dedup on: near-duplicate neighbours lose their edge and gain a vote")
    negative: dict[str, dict[str, float]] | None = None
    if conflict is not None:
        marked = load_nli_edges(paths)
        negative = conflict_adjacency(marked) if marked else None
        log(f"conflict mechanism on: {len(marked)} pre-marked negative edges")
    max_k = max(cfg.k_values)
    questions = dataset.questions

    colored_active = web_mode == "colored" and llm is not None
    if web_mode == "colored" and llm is None:
        log("no LLM client - decomposition is impossible, using the plain web")

    log(f"embedding {len(questions)} questions ...")
    question_vectors = embedder.embed_queries(
        [question.question for question in questions]
    )

    # With the passage window on, a ranking is stored down to the node that
    # completes its `max_k`-th distinct passage instead of being cut at
    # `max_k` nodes. The dense side has to be fetched deeper for the same
    # reason - a two-layer index answers a top-10 request with propositions.
    def window(ranked: list[str]) -> list[str]:
        if not distinct_passages:
            return ranked[:max_k]
        return nodes_for_k_passages(ranked, max_k)

    dense_depth = max_k * DENSE_OVERFETCH if distinct_passages else max_k
    dense_of = {
        question.id: window(topk_retrieve(vector, store, dense_depth))
        for question, vector in zip(questions, question_vectors, strict=True)
    }

    web_of: dict[str, list[str]] = {}
    web_extras_of: dict[str, dict[str, object]] = {}
    ccfg = colored_config if colored_config is not None else ColoredRetrievalConfig()
    extraction_calls = 0
    if colored_active:
        extraction_calls = _run_colored_web(
            questions,
            dataset,
            store,
            graph,
            embedder=embedder,
            llm=llm,
            decomp_llm=decomp_llm if decomp_llm is not None else llm,
            config=ccfg,
            max_k=max_k,
            web_of=web_of,
            extras_of=web_extras_of,
            window=window,
            negative=negative,
            conflict=conflict if negative is not None else None,
            similarity=similarity,
            dedup=dedup,
            question_vectors={
                question.id: vector
                for question, vector in zip(questions, question_vectors, strict=True)
            },
            log=log,
        )
    else:
        for question, vector in zip(questions, question_vectors, strict=True):
            web_result = retrieve(
                vector,
                store,
                graph,
                retrieval_config,
                similarity=similarity,
                dedup=dedup,
                negative=negative,
                conflict=conflict if negative is not None else None,
            )
            web_of[question.id] = window([node for node, _ in web_result.ranked()])
            web_extras_of[question.id] = {
                "stop_reason": web_result.propagation.stop_reason,
                "hops_used": web_result.propagation.hops_used,
                # Open question #4 needs the DISTANCE to the brake, not just
                # whether it fired: `stop_reason` alone cannot say whether
                # `max_nodes` was never approached or missed by one atom.
                "n_activated": len(web_result.propagation.activations),
                "seeds": dict(web_result.seeds),
            }

    records: list[dict[str, object]] = []
    for number, question in enumerate(questions, start=1):
        record: dict[str, object] = {
            "id": question.id,
            "hops": question.hops,
            "gold": list(question.gold_ids),
            "bridge_gold": list(question.bridge_gold_ids),
            "topk": dense_of[question.id],
            "web": web_of[question.id],
            **web_extras_of[question.id],
            "iterative": None,
            "iterative_steps": None,
            "stopped_early": None,
        }
        if llm is not None and iterative:
            trace = iterative_retrieve(
                question.question,
                embedder,
                store,
                dataset.texts,
                dataset.titles,
                llm,
                iterative_config,
            )
            record["iterative"] = window(list(trace.ranked))
            record["iterative_steps"] = list(trace.steps)
            record["stopped_early"] = trace.stopped_early
        records.append(record)
        # Guarded on the same condition as the work it reports. It used to
        # read `llm is not None`, so a run with the baseline switched off
        # still announced "iterative baseline 200/200 questions" - a log line
        # describing work that never happened.
        if llm is not None and iterative and number % 100 == 0:
            log(f"iterative baseline {number}/{len(questions)} questions")

    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.per_query_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    results = aggregate(records, cfg)
    # Part of the receipt, not a detail: every number sealed between
    # 2026-08-14 and 2026-08-16 was produced with this off, and nothing in
    # the artifact said so. A run whose ledger omits the switch cannot be
    # compared with one that has it.
    results["dedup"] = asdict(dedup) if dedup is not None else None
    # The config records what was ASKED for; this records what could
    # actually run. They came apart once already - a `DedupConfig`
    # without a similarity backend is a switched-off mechanism - and
    # the artifact is where that has to be visible, because the
    # artifact is what gets compared across runs.
    results["dedup_mode"] = _dedup_mode(similarity, dedup)
    results["distinct_passages"] = distinct_passages
    if colored_active:
        results["combo"] = _colored_receipt(ccfg)
        results["combo"]["llm_calls_per_question"] = round(
            1 + extraction_calls / len(records), 2
        )
        results["questions_with_bridge"] = sum(
            1 for record in records if record["n_bridges"]
        )
        results["bridge_contains_gold"] = sum(
            1 for record in records if record["bridge_hit"]
        )
        results["non_none_answers"] = sum(
            1 for record in records if any(record["intermediate_answers"])
        )
        results["colors_per_question"] = {
            str(count): total
            for count, total in sorted(
                Counter(int(record["n_colors"]) for record in records).items()
            )
        }
    paths.results_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    log(f"wrote {paths.per_query_jsonl.name} and {paths.results_json.name}")


def _run_colored_web(
    questions: Sequence[MusiqueQuestion],
    dataset: MusiqueDataset,
    store: SeedSource,
    graph: Graph,
    *,
    embedder: Embedder,
    llm: LLMClient,
    decomp_llm: LLMClient,
    config: ColoredRetrievalConfig,
    max_k: int,
    web_of: dict[str, list[str]],
    extras_of: dict[str, dict[str, object]],
    window: Callable[[list[str]], list[str]] | None = None,
    negative: Mapping[str, Mapping[str, float]] | None = None,
    conflict: ConflictConfig | None = None,
    similarity: SimilarityFn | None = None,
    dedup: DedupConfig | None = None,
    on_result: Callable[[str, ColoredRetrievalResult], None] | None = None,
    profile_for: Callable[[int], ColoredRetrievalConfig] | None = None,
    question_vectors: Mapping[str, Sequence[float]] | None = None,
    log: Callable[[str], None],
) -> int:
    """The coloured pipeline, phase by phase (decompose -> embed -> chain ->
    propagate); fills `web_of` and `extras_of` in place and returns the number
    of intermediate-answer extraction calls made.

    `on_result` is a measurement seam: an ablation that needs the ENERGIES
    rather than the order gets the shipped pipeline's own result object,
    instead of rebuilding the chain in a script - a rebuilt chain is a
    different system, and comparing against it would prove nothing.

    `profile_for` is the second measurement seam: given a question's COLOUR
    COUNT it returns the config that question propagates under (D13 profiles
    chosen automatically instead of by the caller). It is consulted only for
    the propagation call - decomposition and intermediate-answer extraction
    keep using `config`, so every LLM prompt stays byte-identical and the
    profile is the single thing that varies.

    `question_vectors` (question id -> the undivided question's embedding)
    feeds the question colour; it is only read when
    `config.question_color_width` is positive."""
    vector_of: dict[str, list[float]] = {}

    def embed_unique(texts: list[str]) -> None:
        fresh = sorted({text for text in texts if text not in vector_of})
        if fresh:
            for text, vector in zip(fresh, embedder.embed_queries(fresh), strict=True):
                vector_of[text] = vector

    def extract_from_top_contact(text: str) -> tuple[str, bool]:
        """Extract an answer from `text`'s top contact; (answer, call_made)."""
        contacts = store.search(vector_of[text], config.seed_width)
        seeds = {node: score for node, score in contacts if score > 0.0}
        if not seeds:
            # retrieve_colored will raise its descriptive error for this
            # question anyway; extracting from a non-contact would be noise.
            return "", False
        top_node = max(seeds.items(), key=lambda item: item[1])[0]
        # On a two-layer index the strongest contact can be a PROPOSITION
        # (`d01024:0#p0`), and the dataset only knows passages - propositions
        # live in their own artifact. A proposition stands for its parent
        # passage everywhere else in the harness (that is the rule the metrics
        # use), so it stands for it here too. Without the fold this raised
        # `KeyError` and killed the run at the first proposition contact.
        passage = top_node.split("#", 1)[0]
        answer = extract_intermediate_answer(
            llm,
            dataset.titles[passage],
            dataset.texts[passage],
            text,
            config.max_answer_words,
        )
        return answer, True

    log("decomposing questions into colours (1 LLM call each, cached) ...")
    subqueries: dict[str, list[str]] = {}
    for number, question in enumerate(questions, start=1):
        subqueries[question.id] = decompose_question(
            question.question, decomp_llm, config.max_colors
        )
        if number % 100 == 0:
            log(f"decomposed {number}/{len(questions)} questions")

    chained_text: dict[tuple[str, int], str] = {
        (question.id, 0): subqueries[question.id][0] for question in questions
    }
    answers_of: dict[str, list[str]] = {question.id: [] for question in questions}
    extraction_calls = 0

    if config.chain_mode == "sequential":
        # Tour-10/12 winner: colour i's top-passage answer feeds colour i+1,
        # level by level. Each level is its own embed phase then LLM phase.
        max_levels = max(len(subs) for subs in subqueries.values())
        for level in range(max_levels):
            active = [
                question
                for question in questions
                if len(subqueries[question.id]) > level
            ]
            if not active:
                break
            log(f"chain level {level}: embedding {len(active)} queries ...")
            embed_unique([chained_text[(question.id, level)] for question in active])
            log(f"chain level {level}: extracting answers (cached where seen) ...")
            for question in active:
                if len(subqueries[question.id]) <= level + 1:
                    continue  # last colour: nothing left to chain
                text = chained_text[(question.id, level)]
                answer, call_made = extract_from_top_contact(text)
                extraction_calls += int(call_made)
                answers_of[question.id].append(answer)
                nxt = subqueries[question.id][level + 1]
                chained_text[(question.id, level + 1)] = (
                    f"{nxt} {answer}" if answer else nxt
                )
    else:
        log("embedding primary sub-queries ...")
        embed_unique([subs[0] for subs in subqueries.values()])
        if config.chain_mode == "single":
            # Tour-9 chain: one extraction from colour 0's top passage,
            # appended to every later colour's query.
            log("extracting intermediate answers (1 LLM call each, cached) ...")
            for number, question in enumerate(questions, start=1):
                subs = subqueries[question.id]
                if len(subs) < 2:
                    continue
                answer, call_made = extract_from_top_contact(subs[0])
                extraction_calls += int(call_made)
                answers_of[question.id].append(answer)
                if number % 100 == 0:
                    log(f"extracted {number}/{len(questions)} answers")
        log("embedding chained later-colour queries ...")
        for question in questions:
            answers = answers_of[question.id]
            answer = answers[0] if answers else ""
            for index, sub in enumerate(subqueries[question.id]):
                chained = sub if index == 0 or not answer else f"{sub} {answer}"
                chained_text[(question.id, index)] = chained
        embed_unique(list(chained_text.values()))

    log("running coloured propagation ...")
    for number, question in enumerate(questions, start=1):
        colored_queries = {
            f"c{index}": vector_of[chained_text[(question.id, index)]]
            for index in range(len(subqueries[question.id]))
        }
        active = config if profile_for is None else profile_for(len(colored_queries))
        result = retrieve_colored(
            colored_queries,
            store,
            graph,
            active,
            similarity=similarity,
            dedup=dedup,
            negative=negative,
            conflict=conflict,
            question=(
                question_vectors[question.id]
                if question_vectors is not None and active.question_color_width
                else None
            ),
        )
        # Measurement seam: an ablation that needs the ENERGIES (not just the
        # order) must see the shipped pipeline's own result, never a
        # reconstruction of it - a rebuilt chain is a different system.
        if on_result is not None:
            on_result(question.id, result)
        ranked_nodes = [node for node, _ in result.ranked()]
        web_of[question.id] = (
            window(ranked_nodes) if window is not None else ranked_nodes[:max_k]
        )
        primary = result.colored.per_color[next(iter(result.seeds_by_color))]
        extras_of[question.id] = {
            "stop_reason": primary.stop_reason,
            "hops_used": result.confidence.hop_depth,
            # `max_nodes` guards ONE propagation run, so the brake's distance
            # is the widest colour, not the merged web (#4). Both are kept:
            # the merged count is what reaches the context window.
            "n_activated": result.confidence.node_count,
            "n_activated_max_color": max(
                len(per.activations) for per in result.colored.per_color.values()
            ),
            "seeds": {
                color: dict(seeds) for color, seeds in result.seeds_by_color.items()
            },
            # Query PARTS only: the question colour is not a decomposition
            # result, and counting it would shift every colour histogram.
            "n_colors": sum(
                1 for color in result.seeds_by_color if color != QUESTION_COLOR
            ),
            "n_bridges": len(result.bridges),
            "bridge_hit": any(node in question.gold_ids for node in result.bridges),
            "subqueries": list(subqueries[question.id]),
            "intermediate_answers": list(answers_of[question.id]),
        }
        if number % 250 == 0:
            log(f"propagated {number}/{len(questions)} questions")
    return extraction_calls


def _colored_receipt(config: ColoredRetrievalConfig) -> dict[str, object]:
    """The experiment receipt of the coloured web, embedded in results.json."""
    variant = {
        "sequential": "sequential_chained_colors",
        "single": "chained_colors_llm_answer",
        "none": "llm_decomposition_fewshot",
    }[config.chain_mode]
    return {
        "variant": variant,
        "chain_mode": config.chain_mode,
        "decomposition_model": config.decomposition_model,
        "seed_width": config.seed_width,
        "ranking": "sum",
        "threshold_ratio": config.propagation.threshold_ratio,
        "split_alpha": config.propagation.split_alpha,
        "max_colors": config.max_colors,
        "max_answer_words": config.max_answer_words,
        "question_color_width": config.question_color_width,
    }


def aggregate(
    records: Sequence[dict[str, object]], config: EvaluationConfig | None = None
) -> dict[str, object]:
    """Reduce per-query records to the report's numbers - pure, re-runnable."""
    cfg = config if config is not None else EvaluationConfig()
    if not records:
        raise ValueError("no per-query records to aggregate")

    systems = ["topk", "web"]
    if all(record["iterative"] is not None for record in records):
        systems.append("iterative")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def metrics_of(
        subset: Sequence[dict[str, object]], system: str, k: int
    ) -> dict[str, float]:
        recalls: list[float] = []
        novelties: list[float] = []
        bridges: list[float] = []
        for record in subset:
            retrieved = record[system]
            reference = record["topk"]
            recalls.append(support_recall_at_k(retrieved, record["gold"], k))
            novelties.append(novelty_at_k(retrieved, reference, record["gold"], k))
            bridges.append(bridge_recall_at_k(retrieved, record["bridge_gold"], k))
        recall = mean(recalls)
        novelty = mean(novelties)
        return {
            "support_recall": recall,
            "novelty": novelty,
            "objective": weighted_objective(recall, novelty, cfg),
            "bridge_recall": mean(bridges),
        }

    primary_k = 5 if 5 in cfg.k_values else cfg.k_values[-1]

    by_hop: dict[str, dict[str, object]] = {}
    for hops in sorted({int(record["hops"]) for record in records}):
        subset = [record for record in records if record["hops"] == hops]
        by_hop[str(hops)] = {
            "questions": len(subset),
            **{
                system: metrics_of(subset, system, primary_k)["objective"]
                for system in systems
            },
        }

    return {
        "question_count": len(records),
        "primary_k": primary_k,
        "systems": {
            system: {str(k): metrics_of(records, system, k) for k in cfg.k_values}
            for system in systems
        },
        "objective_by_hop": by_hop,
        "stop_reasons": dict(Counter(str(record["stop_reason"]) for record in records)),
        # The brakes' headroom, not just whether they fired (#4). A run whose
        # deepest query sits ON `max_hop` has passed without a margin, and
        # `stop_reasons` alone would report that as a clean threshold stop.
        "brake_headroom": _brake_headroom(records),
        "iterative_included": "iterative" in systems,
    }


def _brake_headroom(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Deepest hop and widest web observed, for calibrating the safety caps.

    Fields absent from older `per_query.jsonl` records are reported as
    `None` rather than zero: "not recorded" and "never activated" are
    different answers to open question #4.
    """

    def peak(field: str) -> int | None:
        values = [
            int(record[field]) for record in records if record.get(field) is not None
        ]
        return max(values) if values else None

    return {
        "max_hops_used": peak("hops_used"),
        "max_activated": peak("n_activated"),
        "max_activated_per_color": peak("n_activated_max_color"),
    }


def render_report(results: dict[str, object]) -> str:
    """Markdown report from `aggregate`'s output."""
    systems: dict[str, dict[str, dict[str, float]]] = results["systems"]
    primary_k = results["primary_k"]

    lines = [
        "# Spiyweb - MuSiQue report",
        "",
        f"{results['question_count']} questions; "
        f"the primary number is S@{primary_k} "
        "(0.65 * support recall + 0.35 * novelty).",
        "",
        "## Systems",
        "",
        "| system | k | support recall | novelty | S | bridge recall |",
        "|---|---|---|---|---|---|",
    ]
    for system, per_k in systems.items():
        for k, values in per_k.items():
            lines.append(
                f"| {system} | {k} "
                f"| {values['support_recall']:.3f} "
                f"| {values['novelty']:.3f} "
                f"| {values['objective']:.3f} "
                f"| {values['bridge_recall']:.3f} |"
            )
    lines += [
        "",
        "Novelty is measured against plain `top-k` at the same cutoff, so "
        "`top-k`'s own novelty is 0 by construction.",
    ]
    if not results["iterative_included"]:
        lines += ["", "The iterative baseline was skipped in this run."]

    # Stated in every report, both ways round. Redundancy-to-vote is the
    # project's most original claim, and for the whole 2026-08 campaign it
    # was configured but never actually running here - a reader had no way
    # to tell from the artifact. Absent key = an older run, which is exactly
    # the case this line exists to make visible.
    passages = results.get("distinct_passages", "unrecorded")
    if passages == "unrecorded":
        lines += [
            "",
            "Ranking window: NOT RECORDED - this run predates the "
            "`--distinct-passages` switch, so rankings were cut at max_k "
            "NODES.",
        ]
    else:
        lines += [
            "",
            "Ranking window: "
            + (
                "max_k distinct PASSAGES (not comparable with runs cut at max_k nodes)."
                if passages
                else "max_k nodes, as every sealed run."
            ),
        ]

    dedup = results.get("dedup", "unrecorded")
    if dedup == "unrecorded":
        lines += [
            "",
            "Duplicate suppression: NOT RECORDED - this run predates the "
            "`--dedup` switch, so the mechanism was off.",
        ]
    elif dedup is None:
        lines += [
            "",
            "Duplicate suppression (D6): OFF. Near-duplicate neighbours kept "
            "their edges and cast no votes; comparable with the sealed runs.",
        ]
    else:
        lines += [
            "",
            f"Duplicate suppression (D6): ON, {dedup}. Not comparable with "
            "runs made before it was switched on.",
        ]

    lines += [
        "",
        "## Reference (reported, not reproduced)",
        "",
        "HippoRAG (ColBERTv2), arXiv:2405.14831 Table 2, MuSiQue: "
        f"R@2 {HIPPORAG_REFERENCE['R@2']:.3f}, "
        f"R@5 {HIPPORAG_REFERENCE['R@5']:.3f}. "
        "Informative only - different sample, never a gate (D29).",
        "",
        f"## Objective by hop count (S@{primary_k})",
        "",
    ]
    hop_systems = list(systems)
    lines.append("| hops | questions | " + " | ".join(hop_systems) + " |")
    lines.append("|---" * (2 + len(hop_systems)) + "|")
    by_hop: dict[str, dict[str, object]] = results["objective_by_hop"]
    for hops, row in by_hop.items():
        cells = " | ".join(f"{row[system]:.3f}" for system in hop_systems)
        lines.append(f"| {hops} | {row['questions']} | {cells} |")

    lines += [
        "",
        "## Web stop reasons",
        "",
        "| reason | count |",
        "|---|---|",
    ]
    stop_reasons: dict[str, int] = results["stop_reasons"]
    for reason, count in sorted(stop_reasons.items()):
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    return "\n".join(lines)


def _real_embedder(device: str | None) -> Embedder:
    from spiyweb.config import EmbeddingConfig
    from spiyweb.embedding import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(EmbeddingConfig(device=device))


def _llm_cache_path(paths: IndexPaths, model: str | None) -> Path:
    """Model-specific cache file; the rule lives on the layout."""
    return paths.llm_cache_for(model)


def _real_llm(
    paths: IndexPaths, model: str | None, *, native_no_think: bool = False
) -> LLMClient:
    from spiyweb.config import LLMConfig
    from spiyweb.evaluation.cache import CachedLLMClient
    from spiyweb.llm import NativeOllamaClient, OpenAICompatClient

    config = LLMConfig(model=model) if model is not None else LLMConfig()
    client: LLMClient = (
        NativeOllamaClient(config, think=False)
        if native_no_think
        else OpenAICompatClient(config)
    )
    return CachedLLMClient(client, _llm_cache_path(paths, model))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spiyweb.evaluation.run",
        description="MuSiQue evaluation pipeline (Phase 1 measurement).",
    )
    parser.add_argument(
        "stage", choices=["download", "index", "evaluate", "report", "all"]
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/musique"))
    parser.add_argument(
        "--dataset",
        choices=["musique", "2wiki", "hotpotqa"],
        default="musique",
        help="benchmark to run; 2wiki and hotpotqa are cross-dataset "
        "generalisation checks and run the winner configuration AS IS "
        "(protocol: no tuning on them) - pair with --data-dir data/2wiki "
        "or data/hotpotqa. hotpotqa is the UNTOUCHED third set: after "
        "tour 13, 2wiki no longer counts as blind",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-entity-llm",
        action="store_true",
        help="ablation: spaCy-only entity extraction (default is full hybrid)",
    )
    parser.add_argument(
        "--propositions",
        action="store_true",
        help="index the proposition node layer (D10): one LLM call per "
        "passage at index time plus the derivation edge layer; off by "
        "default because the extraction cost is an open question (#2)",
    )
    parser.add_argument(
        "--nli",
        action="store_true",
        help="index-time NLI contradiction stage (D26): score high-cosine "
        "pairs with the configured NLI model and write edges_nli.json; "
        "evaluate then activates the conflict mechanism over those edges. "
        "Off by default (model choice and threshold are open question #10)",
    )
    parser.add_argument(
        "--skip-iterative",
        action="store_true",
        help="skip the iterative baseline and nothing else - the coloured "
        "web still runs. To run with no LLM at all (Ollama down), combine "
        "it with --web plain",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="turn duplicate suppression ON (D6: a near-duplicate neighbour "
        "loses its edge and the surviving idea gains a vote). OFF by default "
        "because every sealed run was measured that way and switching it on "
        "breaks comparability - not because the mechanism is optional. The "
        "results ledger and the report state which way the run went",
    )
    parser.add_argument(
        "--distinct-passages",
        action="store_true",
        help="store rankings down to the max_k-th distinct PASSAGE instead of "
        "the max_k-th node. Only matters on a two-layer index, where the "
        "node cut costs the web ~.012 S@5; OFF by default because turning it "
        "on breaks comparability with the sealed runs",
    )
    parser.add_argument(
        "--web",
        choices=["colored", "plain"],
        default="colored",
        help="web system to evaluate: the coloured, sequentially chained "
        "winner (default; ~2.6 LLM calls/question) or the plain "
        "single-seed web",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Ollama model for entity fallback, iterative baseline and "
        "answer extraction (default: LLMConfig's)",
    )
    parser.add_argument(
        "--decomp-model",
        default=None,
        help="model for the question -> colours decomposition "
        "(default: ColoredRetrievalConfig's, the tour-12 winner)",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=None,
        help="cap on colours kept from the decomposition (default: config's)",
    )
    parser.add_argument(
        "--chain-mode",
        choices=["none", "single", "sequential"],
        default=None,
        help="intermediate-answer chaining: sequential (tour-12 winner), "
        "single (tour-9 ablation) or none (decomposition only)",
    )
    parser.add_argument(
        "--question-color-width",
        type=int,
        default=None,
        help="seed the undivided question as one more colour with this many "
        "contacts (gate round #6); 0 = off, the default",
    )
    args = parser.parse_args(argv)

    colored_defaults = ColoredRetrievalConfig()
    colored_cfg = ColoredRetrievalConfig(
        max_colors=(
            args.max_colors
            if args.max_colors is not None
            else colored_defaults.max_colors
        ),
        chain_mode=(
            args.chain_mode
            if args.chain_mode is not None
            else colored_defaults.chain_mode
        ),
        decomposition_model=(
            args.decomp_model
            if args.decomp_model is not None
            else colored_defaults.decomposition_model
        ),
        question_color_width=(
            args.question_color_width
            if args.question_color_width is not None
            else colored_defaults.question_color_width
        ),
    )

    defaults = EvaluationConfig()
    cfg = EvaluationConfig(
        sample_size=(
            args.sample_size if args.sample_size is not None else defaults.sample_size
        ),
        sample_seed=(
            args.sample_seed if args.sample_seed is not None else defaults.sample_seed
        ),
    )
    paths = IndexPaths(root=args.data_dir)
    if args.dataset == "musique":
        dataset_path, loader, dataset_url = (
            paths.dataset_jsonl,
            load_dataset,
            cfg.dataset_url,
        )
    elif args.dataset == "2wiki":
        dataset_path, loader, dataset_url = (
            paths.twowiki_dev_json,
            load_2wiki,
            TWOWIKI_DEV_URL,
        )
    else:
        dataset_path, loader, dataset_url = (
            paths.hotpot_dev_json,
            load_hotpotqa,
            HOTPOTQA_DEV_URL,
        )

    if args.stage in ("download", "all"):
        download_musique(dataset_path, dataset_url)
        print(f"dataset ready at {dataset_path}")

    if args.stage in ("index", "evaluate", "all"):
        dataset = loader(dataset_path, cfg)
        # One embedder for both stages: loading e5-large twice would double
        # a multi-minute model load for nothing.
        embedder = _real_embedder(args.device)

    if args.stage in ("index", "all"):
        from spiyweb.config import LLMConfig
        from spiyweb.entities import load_spacy_pipeline

        llm_model = args.llm_model if args.llm_model is not None else LLMConfig().model
        nli_model = None
        nli_model_name = None
        if args.nli:
            from spiyweb.config import NLIModelConfig
            from spiyweb.nli import TransformersNLIModel

            nli_cfg = NLIModelConfig(device=args.device)
            nli_model = TransformersNLIModel(nli_cfg)
            nli_model_name = nli_cfg.model_name
        build_index(
            dataset,
            paths,
            embedder=embedder,
            entity_pipeline=load_spacy_pipeline(),
            llm=(
                _real_llm(paths, args.llm_model)
                if args.propositions or not args.no_entity_llm
                else None
            ),
            llm_model=llm_model
            if args.propositions or not args.no_entity_llm
            else None,
            entity_llm=not args.no_entity_llm,
            propositions=args.propositions,
            nli_model=nli_model,
            nli_model_name=nli_model_name,
            force=args.force,
        )

    if args.stage in ("evaluate", "all"):
        # An LLM is needed for the coloured web AND for the baseline; it is
        # only dispensable when neither is wanted. Tying it to
        # `--skip-iterative` alone demoted the winner to the plain web behind
        # the caller's back.
        use_llm = args.web == "colored" or not args.skip_iterative
        decomp_llm = None
        if use_llm and args.web == "colored":
            decomp_llm = _real_llm(
                paths,
                colored_cfg.decomposition_model,
                native_no_think=colored_cfg.decomposition_no_think,
            )
        evaluate_questions(
            dataset,
            paths,
            embedder=embedder,
            llm=_real_llm(paths, args.llm_model) if use_llm else None,
            decomp_llm=decomp_llm,
            iterative=not args.skip_iterative,
            colored_config=colored_cfg,
            eval_config=cfg,
            web_mode=args.web,
            # The conflict mechanism follows the artifact: an index carrying
            # edges_nli.json gets it, any other index is a no-op here.
            conflict=ConflictConfig() if paths.nli_json.exists() else None,
            # Explicit, never inferred from an artifact: dedup changes what
            # the web returns, so it may only be on because someone asked.
            dedup=DedupConfig() if args.dedup else None,
            distinct_passages=args.distinct_passages,
        )

    if args.stage in ("report", "all"):
        results = json.loads(paths.results_json.read_text(encoding="utf-8"))
        print(render_report(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
