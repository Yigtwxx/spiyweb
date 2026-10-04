"""The whole measurement pipeline, in miniature, under fakes.

The claim under test: index -> evaluate -> report actually runs end to end
and produces the numbers the design promises - on a rigged 3-document corpus
where the bridge document is semantically invisible and only the entity hop
can lift it, the web MUST outscore plain top-k on the weighted objective.
If this miniature ever stops showing that gap, the pipeline (not the corpus)
broke. Also pinned: stage resume, --force, the hybrid entity LLM path, and
the e5 rule that passages never go through the query embedding path.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("numpy")
pytest.importorskip("faiss")

from spiyweb.config import DedupConfig, EntityEdgeConfig, EvaluationConfig
from spiyweb.evaluation.datasets import load_dataset
from spiyweb.evaluation.index import (
    IndexPaths,
    build_index,
    load_graph,
    load_store,
)
from spiyweb.evaluation.run import evaluate_questions, main, render_report

if TYPE_CHECKING:
    from pathlib import Path

# One 2-hop question. Corpus keys sort to: Alpha=d00000, Bravo=d00001,
# Noise=d00002. Alpha and Bravo are gold; Alpha is the bridge document.
RECORD = {
    "id": "2hop__q1",
    "question": "the question",
    "answer": "the answer",
    "answerable": True,
    "paragraphs": [
        {
            "idx": 0,
            "title": "Alpha",
            "paragraph_text": "alpha text",
            "is_supporting": True,
        },
        {
            "idx": 1,
            "title": "Bravo",
            "paragraph_text": "bravo text",
            "is_supporting": True,
        },
        {
            "idx": 2,
            "title": "Noise",
            "paragraph_text": "noise text",
            "is_supporting": False,
        },
    ],
    "question_decomposition": [
        {"id": 0, "question": "step 0", "answer": "x", "paragraph_support_idx": 0},
        {"id": 1, "question": "step 1", "answer": "y", "paragraph_support_idx": 1},
    ],
}

# A twin of Alpha under another title: identical vector, so cosine calls it
# a near-duplicate at the shipped floor. Only the dedup test uses it - the
# rigged miniature is otherwise orthogonal, and a corpus with no duplicates
# cannot show whether duplicate suppression ran.
RECORD_WITH_TWIN = {
    **RECORD,
    "paragraphs": [
        *RECORD["paragraphs"],
        {
            "idx": 3,
            "title": "Alpha Mirror",
            "paragraph_text": "alpha text",
            "is_supporting": False,
        },
    ],
}

# Bravo is orthogonal to the question: dense retrieval can never rank it
# above Noise. Only the shared entity can carry energy to it.
PASSAGE_VECTORS = {
    "Alpha\nalpha text": [1.0, 0.0, 0.0],
    "Alpha Mirror\nalpha text": [1.0, 0.0, 0.0],
    "Bravo\nbravo text": [0.0, 1.0, 0.0],
    "Noise\nnoise text": [0.0, 0.0, 1.0],
}
QUERY_VECTORS = {
    "the question": [1.0, 0.0, 0.1],
    "I still need the Bravo document.": [0.0, 1.0, 0.0],
    # The coloured path: decomposition yields these two sub-queries; the
    # second is chained with the extracted intermediate answer before embedding.
    "find Alpha": [1.0, 0.0, 0.0],
    "find Bravo Wardenclyffe": [0.0, 1.0, 0.0],
    # Level-2 text of the three-colour sequential test: colour 1's top passage
    # (Bravo) yields "Static", which chains into colour 2's query.
    "find Noise Static": [0.0, 1.0, 0.0],
}

SPACY_ENTITIES = {
    "Alpha\nalpha text": ["Wardenclyffe"],
    "Alpha Mirror\nalpha text": ["Wardenclyffe"],
    "Bravo\nbravo text": ["Wardenclyffe"],
    "Noise\nnoise text": [],  # below min_entities -> routed to the LLM
}


class FakeEmbedder:
    def __init__(self) -> None:
        self.passage_calls = 0
        self.query_calls = 0

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.query_calls += len(texts)
        return [QUERY_VECTORS[text] for text in texts]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_calls += len(texts)
        for text in texts:
            assert text in PASSAGE_VECTORS, (
                f"passage embedded with unexpected composition {text!r} - "
                "passages must be title + newline + text, and must never go "
                "through the query path"
            )
        return [PASSAGE_VECTORS[text] for text in texts]


class FakeSpan:
    def __init__(self, text: str) -> None:
        self.text = text
        self.label_ = "ORG"


class FakeDoc:
    def __init__(self, entities: list[str]) -> None:
        self.ents = [FakeSpan(entity) for entity in entities]


class FakePipeline:
    def pipe(self, texts: list[str]) -> list[FakeDoc]:
        return [FakeDoc(SPACY_ENTITIES[text]) for text in texts]


class FakeLLM:
    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.script.pop(0)


EVAL_CONFIG = EvaluationConfig(sample_size=0, k_values=(1, 2, 3))


def make_index(tmp_path: Path, record: dict | None = None) -> IndexPaths:
    paths = IndexPaths(root=tmp_path / "musique")
    paths.root.mkdir(parents=True)
    paths.dataset_jsonl.write_text(
        json.dumps(record if record is not None else RECORD) + "\n", encoding="utf-8"
    )
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    build_index(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        entity_pipeline=FakePipeline(),
        llm=FakeLLM(["Static Hiss"]),
        # df(wardenclyffe)=2 of 3 chunks; the default 0.02 ratio (or the old
        # 0.5 hand value) would drop it in a corpus this tiny.
        entity_config=EntityEdgeConfig(max_df_ratio=1.0),
        log=lambda message: None,
    )
    return paths


def test_index_writes_every_artifact_and_the_hybrid_llm_path_runs(
    tmp_path: Path,
) -> None:
    paths = make_index(tmp_path)
    for artifact in (
        paths.vectors_npz,
        paths.entities_json,
        paths.nodes_json,
        paths.meta_json,
        paths.edges_json("semantic"),
        paths.edges_json("entity"),
        paths.edges_json("structural"),
    ):
        assert artifact.exists(), f"missing artifact {artifact.name}"

    entities = json.loads(paths.entities_json.read_text(encoding="utf-8"))
    assert entities["d00002:0"] == ["static hiss"], (
        "the spaCy-blind chunk must carry the LLM's (normalised) entities - "
        "that IS the hybrid path"
    )
    meta = json.loads(paths.meta_json.read_text(encoding="utf-8"))
    assert meta["entity_llm"] is True
    assert meta["corpus_chunks"] == 3


def test_existing_artifacts_skip_their_stages_until_forced(
    tmp_path: Path,
) -> None:
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)

    resumed_embedder = FakeEmbedder()
    resumed_llm = FakeLLM([])
    build_index(
        dataset,
        paths,
        embedder=resumed_embedder,
        entity_pipeline=FakePipeline(),
        llm=resumed_llm,
        entity_config=EntityEdgeConfig(max_df_ratio=1.0),
        log=lambda message: None,
    )
    assert resumed_embedder.passage_calls == 0, (
        "existing vectors must skip the embed stage - that skip is the resume"
    )
    assert resumed_llm.prompts == []

    forced_embedder = FakeEmbedder()
    build_index(
        dataset,
        paths,
        embedder=forced_embedder,
        entity_pipeline=FakePipeline(),
        llm=FakeLLM(["Static Hiss"]),
        entity_config=EntityEdgeConfig(max_df_ratio=1.0),
        force=True,
        log=lambda message: None,
    )
    assert forced_embedder.passage_calls == 3


def test_store_and_graph_load_back_aligned(tmp_path: Path) -> None:
    paths = make_index(tmp_path)
    store = load_store(paths)
    graph = load_graph(paths)

    assert len(store) == 3
    assert graph.nodes == {"d00000:0", "d00001:0"}, (
        "only the entity edge's endpoints carry edges here: the corpus is "
        "orthogonal (no semantic edges) and single-unit (no structural edges)"
    )
    assert graph.nodes <= {chunk_id for chunk_id, _ in store.search([1, 0, 0], 3)}


def test_the_web_outscores_topk_on_the_rigged_miniature(tmp_path: Path) -> None:
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=FakeLLM(["I still need the Bravo document.", "The answer is Bravo."]),
        eval_config=EVAL_CONFIG,
        web_mode="plain",
        log=lambda message: None,
    )

    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert results["question_count"] == 1
    assert results["primary_k"] == 3
    assert results["iterative_included"] is True

    web_at_2 = results["systems"]["web"]["2"]
    topk_at_2 = results["systems"]["topk"]["2"]
    # Hand-traced: web@2 = [Alpha, Bravo] -> recall 1.0, novelty 1/2 (Bravo is
    # gold and absent from dense top-2 [Alpha, Noise]) -> S = .825.
    # topk@2 -> recall 1/2, novelty 0 -> S = .325.
    assert web_at_2["support_recall"] == pytest.approx(1.0)
    assert web_at_2["novelty"] == pytest.approx(0.5)
    assert web_at_2["objective"] == pytest.approx(0.825)
    assert topk_at_2["objective"] == pytest.approx(0.325)
    assert web_at_2["objective"] > topk_at_2["objective"], (
        "the entity hop lifts the semantically invisible gold document - if "
        "this gap vanishes, the pipeline broke, not the corpus"
    )

    per_query = [
        json.loads(line)
        for line in paths.per_query_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert per_query[0]["web"][:2] == ["d00000:0", "d00001:0"]
    assert per_query[0]["stop_reason"] == "threshold"
    assert per_query[0]["iterative_steps"] == [
        "I still need the Bravo document.",
        "The answer is Bravo.",
    ]


def test_the_colored_web_chains_the_answer_on_the_rigged_miniature(
    tmp_path: Path,
) -> None:
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    # Scripted call order of the coloured pipeline: decomposition first,
    # answer extraction second, then the iterative baseline's two rewrites.
    llm = FakeLLM(
        [
            "find Alpha\nfind Bravo",
            "Wardenclyffe",
            "I still need the Bravo document.",
            "The answer is Bravo.",
        ]
    )
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=llm,
        eval_config=EVAL_CONFIG,
        log=lambda message: None,
    )

    assert llm.script == [], "every scripted LLM reply must have been consumed"

    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    web_at_2 = results["systems"]["web"]["2"]
    # Colour 0 lands on Alpha, extraction chains "Wardenclyffe" into colour 1,
    # which lands on Bravo; both colours cross the entity edge and meet.
    assert web_at_2["support_recall"] == pytest.approx(1.0)
    assert web_at_2["novelty"] == pytest.approx(0.5)
    assert results["combo"]["variant"] == "sequential_chained_colors"
    assert results["combo"]["chain_mode"] == "sequential"
    assert results["combo"]["llm_calls_per_question"] == pytest.approx(2.0)
    assert results["questions_with_bridge"] == 1
    assert results["bridge_contains_gold"] == 1
    assert results["non_none_answers"] == 1
    assert results["colors_per_question"] == {"2": 1}

    per_query = [
        json.loads(line)
        for line in paths.per_query_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    record = per_query[0]
    assert set(record["web"][:2]) == {"d00000:0", "d00001:0"}
    assert record["subqueries"] == ["find Alpha", "find Bravo"]
    assert record["intermediate_answers"] == ["Wardenclyffe"]
    assert record["n_colors"] == 2
    assert record["n_bridges"] == 2, "both gold chunks are reached by both colours"
    assert record["seeds"] == {
        "c0": {"d00000:0": 1.0},
        "c1": {"d00001:0": 1.0},
    }, "non-positive contacts must not seed a colour"


def test_sequential_chaining_extracts_one_answer_per_level(
    tmp_path: Path,
) -> None:
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    # Three colours -> two extraction levels: level 1's question must be the
    # CHAINED text (sub + previous answer), not the raw sub-query - that is
    # the whole difference between sequential and single chaining.
    llm = FakeLLM(
        [
            "find Alpha\nfind Bravo\nfind Noise",
            "Wardenclyffe",
            "Static",
            "I still need the Bravo document.",
            "The answer is Bravo.",
        ]
    )
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=llm,
        eval_config=EVAL_CONFIG,
        log=lambda message: None,
    )

    assert llm.script == [], "every scripted LLM reply must have been consumed"
    assert "find Bravo Wardenclyffe" in llm.prompts[2], (
        "level-1 extraction must ask the chained question, not the raw sub"
    )

    per_query = [
        json.loads(line)
        for line in paths.per_query_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    record = per_query[0]
    assert record["intermediate_answers"] == ["Wardenclyffe", "Static"]
    assert record["n_colors"] == 3

    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert results["combo"]["llm_calls_per_question"] == pytest.approx(3.0)
    assert results["non_none_answers"] == 1


def test_the_profile_seam_is_asked_per_question_and_reaches_propagation(
    tmp_path: Path,
) -> None:
    """Gate tour 5 (colour count -> automatic D13 profile) needs the shipped
    coloured pipeline to propagate one question under a per-question config.
    Two things are pinned: the seam is asked with the QUESTION'S colour count,
    and what it returns actually drives propagation - a hook that is consulted
    but ignored would produce a tour of identical cells."""
    from dataclasses import replace

    from spiyweb.config import ColoredRetrievalConfig
    from spiyweb.evaluation.run import _run_colored_web

    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    store = load_store(paths)
    graph = load_graph(paths)
    base = ColoredRetrievalConfig()
    asked: list[int] = []

    def run(profile_for: object | None) -> dict[str, float]:
        llm = FakeLLM(["find Alpha\nfind Bravo", "Wardenclyffe"])
        web_of: dict[str, list[str]] = {}
        energy: dict[str, float] = {}
        _run_colored_web(
            dataset.questions,
            dataset,
            store,
            graph,
            embedder=FakeEmbedder(),
            llm=llm,
            decomp_llm=llm,
            config=base,
            max_k=10,
            web_of=web_of,
            extras_of={},
            profile_for=profile_for,  # type: ignore[arg-type]
            on_result=lambda _, result: energy.update(dict(result.ranked())),
            log=lambda message: None,
        )
        assert llm.script == [], "the profile must not change any LLM prompt"
        assert list(web_of) == ["2hop__q1"]
        return energy

    def starved(n_colors: int) -> ColoredRetrievalConfig:
        asked.append(n_colors)
        # A threshold at 90% of the seed kills every hop, so the entity edge
        # stops delivering: the same nodes carry strictly less energy. A hook
        # that is consulted but ignored would leave the numbers identical.
        return replace(base, propagation=replace(base.propagation, threshold_ratio=0.9))

    shipped = run(None)
    starved_energy = run(starved)

    assert asked == [2], "the seam sees the question's own colour count"
    assert set(starved_energy) == set(shipped)
    assert all(starved_energy[node] < shipped[node] for node in shipped), (
        f"the returned profile must drive propagation: {starved_energy} vs {shipped}"
    )


def test_skipping_the_iterative_baseline_still_produces_the_web_number(
    tmp_path: Path,
) -> None:
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=None,
        eval_config=EVAL_CONFIG,
        log=lambda message: None,
    )
    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert results["iterative_included"] is False
    assert "iterative" not in results["systems"]
    assert "web" in results["systems"]


def test_dropping_the_baseline_keeps_the_coloured_web(tmp_path: Path) -> None:
    """`iterative=False` must not quietly demote the system under test.

    The only way to drop the baseline used to be `llm=None`, which also
    turned the coloured winner into the plain web - so a run that named the
    winner measured something else and said nothing about it.
    """
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        # Decomposition, then one answer extraction - the same scripted order
        # the coloured test uses, minus the baseline's two rewrites.
        llm=FakeLLM(["find Alpha\nfind Bravo", "Wardenclyffe"]),
        eval_config=EVAL_CONFIG,
        web_mode="colored",
        iterative=False,
        log=lambda message: None,
    )
    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert results["iterative_included"] is False
    assert "iterative" not in results["systems"]

    per_query = [
        json.loads(line)
        for line in paths.per_query_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert per_query[0]["iterative"] is None, "the baseline really was skipped"
    assert per_query[0]["n_colors"] >= 1, (
        "the coloured path still ran - n_colors only exists on that branch"
    )


def test_brake_headroom_reports_the_distance_to_the_safety_caps(
    tmp_path: Path,
) -> None:
    """Open question #4 needs headroom, not just whether a brake fired.

    `stop_reasons` says "threshold" whether the deepest query stopped six
    hops below the cap or one hop under it, so the run has to persist the
    peaks themselves.
    """
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=None,
        eval_config=EVAL_CONFIG,
        web_mode="plain",
        log=lambda message: None,
    )

    per_query = [
        json.loads(line)
        for line in paths.per_query_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    activated = per_query[0]["n_activated"]
    assert activated >= 1, "a web that activated nothing is not a web"

    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    headroom = results["brake_headroom"]
    assert headroom["max_activated"] == activated
    assert headroom["max_hops_used"] == per_query[0]["hops_used"]
    # The plain path runs one propagation, so there is no per-colour peak to
    # report - and "not recorded" must not be rounded down to zero.
    assert headroom["max_activated_per_color"] is None


def test_the_dedup_switch_actually_reaches_the_mechanism(tmp_path: Path) -> None:
    """The harness ran the whole 2026-08 campaign with dedup configured but
    never delivered: `retrieve()` keeps it off unless it gets the similarity
    backend too, and the harness passed neither. This pins the fix - the
    switch must change what the web returns, not just what the ledger says.

    The corpus here carries a real twin (a copy of Alpha under another
    title), because the rigged miniature is otherwise orthogonal and nothing
    in it is a duplicate at the shipped floor.
    """
    paths = make_index(tmp_path, record=RECORD_WITH_TWIN)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)

    def web_ranking() -> list[str]:
        line = paths.per_query_jsonl.read_text(encoding="utf-8").splitlines()[0]
        return json.loads(line)["web"]

    common = {
        "embedder": FakeEmbedder(),
        "llm": None,
        "eval_config": EVAL_CONFIG,
        "web_mode": "plain",
        "log": lambda message: None,
    }
    evaluate_questions(dataset, paths, **common)
    without = web_ranking()

    evaluate_questions(dataset, paths, dedup=DedupConfig(), **common)
    assert web_ranking() != without, (
        "the twin must be voted instead of seeded - if the ranking is "
        "identical the mechanism never ran"
    )

    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert results["dedup"]["floor"] == DedupConfig().floor, (
        "the ledger records the config the run actually used"
    )
    assert results["dedup_mode"] == "full", (
        "the config says what was asked for; dedup_mode says what could "
        "run, and a run whose artifact carries only the first cannot be "
        "told apart from one where the mechanism was silently off"
    )
    assert "Duplicate suppression (D6): ON" in render_report(results)


def test_the_passage_window_stores_enough_nodes_to_fill_the_cutoffs(
    tmp_path: Path,
) -> None:
    """A two-layer ranking cut at max_k NODES can carry three passages, and
    `passages_at_k` can only fold what it is given. The switch stores down to
    the max_k-th distinct passage instead; the default must not move."""
    from spiyweb.evaluation.metrics import nodes_for_k_passages

    ranking = ["d0:0", "d0:0#p1", "d0:0#p2", "d1:0", "d1:0#p0", "d2:0"]
    assert nodes_for_k_passages(ranking, 3) == ranking, (
        "the prefix must reach the node that completes the third passage"
    )
    assert nodes_for_k_passages(ranking, 2) == ranking[:4]
    assert nodes_for_k_passages(ranking, 9) == ranking, (
        "a ranking too short to supply k passages is a finding, not an error"
    )
    assert nodes_for_k_passages(["a", "b", "c"], 2) == ["a", "b"], (
        "a chunk-only ranking must behave exactly as the plain cut did"
    )

    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=None,
        eval_config=EVAL_CONFIG,
        web_mode="plain",
        distinct_passages=True,
        log=lambda message: None,
    )
    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert results["distinct_passages"] is True, "the ledger records the window"
    assert "distinct PASSAGES" in render_report(results)


def test_the_report_states_the_default_window(tmp_path: Path) -> None:
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=None,
        eval_config=EVAL_CONFIG,
        web_mode="plain",
        log=lambda message: None,
    )
    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert results["distinct_passages"] is False
    assert "max_k nodes, as every sealed run" in render_report(results)
    del results["distinct_passages"]
    assert "Ranking window: NOT RECORDED" in render_report(results)


def test_the_report_states_that_dedup_was_off(tmp_path: Path) -> None:
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=None,
        eval_config=EVAL_CONFIG,
        web_mode="plain",
        log=lambda message: None,
    )
    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert results["dedup"] is None
    assert "Duplicate suppression (D6): OFF" in render_report(results), (
        "silence is what produced the ambiguity in the first place; the "
        "report must say it either way"
    )


def test_an_older_results_file_is_reported_as_unrecorded(tmp_path: Path) -> None:
    # Runs sealed before the switch existed have no key at all. "Not
    # recorded" and "off" happen to mean the same thing here, but the report
    # must not claim to know something the artifact never said.
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=None,
        eval_config=EVAL_CONFIG,
        web_mode="plain",
        log=lambda message: None,
    )
    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    del results["dedup"]  # exactly what a pre-2026-08-16 artifact looks like
    assert "NOT RECORDED" in render_report(results)


def test_the_report_renders_every_promised_section(tmp_path: Path) -> None:
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=FakeLLM(["I still need the Bravo document.", "The answer is Bravo."]),
        eval_config=EVAL_CONFIG,
        web_mode="plain",
        log=lambda message: None,
    )
    results = json.loads(paths.results_json.read_text(encoding="utf-8"))
    report = render_report(results)

    assert "| web | 2 |" in report
    assert "| iterative |" in report
    assert "HippoRAG" in report and "not reproduced" in report
    assert "Objective by hop count" in report
    assert "## Web stop reasons" in report
    assert "| threshold | 1 |" in report


def test_the_report_stage_of_the_cli_prints_the_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_index(tmp_path)
    dataset = load_dataset(paths.dataset_jsonl, EVAL_CONFIG)
    evaluate_questions(
        dataset,
        paths,
        embedder=FakeEmbedder(),
        llm=None,
        eval_config=EVAL_CONFIG,
        log=lambda message: None,
    )

    exit_code = main(["report", "--data-dir", str(paths.root)])
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "# Spiyweb - MuSiQue report" in printed
    assert "The iterative baseline was skipped in this run." in printed
