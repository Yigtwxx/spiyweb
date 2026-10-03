"""Progress read off a background job's lines - and pinned to the real logs."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from spiyweb import cli, indexing
from spiyweb.progress import (
    EDGE_LAYERS,
    IndexProgress,
    JobProgress,
    bar_percent,
    classify,
    clock_text,
)


def _sample(node: ast.expr) -> str | None:
    """The line a log call would print, with every placeholder filled by a
    dummy number - what the stage rules have to recognise."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue) and (
                isinstance(value.value, ast.Name) and value.value.id == "layer"
            ):
                parts.append(EDGE_LAYERS[0])  # the loop over the edge builders
            else:
                parts.append("7")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _sample(node.left), _sample(node.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.IfExp):
        return _sample(node.body)
    return None


def _logged(function: object, name: str) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))  # type: ignore[arg-type]
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
            and node.args
            and not any(k.arg == "file" for k in node.keywords)
        ):
            sample = _sample(node.args[0])
            assert sample is not None, ast.unparse(node)
            lines.append(sample)
    return lines


def test_every_line_build_index_logs_is_recognised_as_a_stage() -> None:
    lines = _logged(indexing.build_index, "log")
    assert len(lines) >= 15, "the AST walk found the log calls"
    unknown = [line for line in lines if classify(line) is None]
    assert unknown == [], f"add a rule in progress.RULES for: {unknown}"


def test_the_edge_layers_are_the_ones_build_index_writes_in_that_order() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(indexing.build_index)))
    builders = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "edge_builders"
    )
    assert isinstance(builders, ast.Dict)
    keys = tuple(k.value for k in builders.keys if isinstance(k, ast.Constant))
    assert keys == EDGE_LAYERS, "the edges k/4 counter must count these"


def test_every_stdout_line_of_the_index_verb_is_recognised() -> None:
    lines = [line for line in _logged(cli._index, "print") if line.strip()]
    assert lines, "the verb prints its progress"
    unknown = [line for line in lines if classify(line) is None]
    assert unknown == [], f"add a rule in progress.RULES for: {unknown}"


@pytest.mark.parametrize(
    ("line", "stage"),
    [
        ("3 document(s), 12 unit(s) -> my-index", "read"),
        ("extracting propositions: 12 passages, one LLM call each ...", "propositions"),
        ("embedding 12 passages ...", "embed"),
        ("vectors exist, skipping the embed stage", "embed"),
        ("4 of 12 passages fall below min_entities=0; LLM fallback is OFF", "entities"),
        ("semantic layer: 40 edges", "edges"),
        ("derivation edges exist, skipping", "edges"),
        ("done: 12 chunk(s), 80 edge(s) across 3 layer(s)", "done"),
        ("Batches: 100%|##########| 1/1", None),
        ("some library chatter", None),
    ],
)
def test_classify_maps_a_line_to_its_stage(line: str, stage: str | None) -> None:
    assert classify(line) == stage, line


@pytest.mark.parametrize(
    ("line", "percent"),
    [
        ("Batches:  45%|####5     | 9/20 [00:01<00:01,  8.10it/s]", 45),
        ("model.safetensors: 100%|##########| 2.24G/2.24G [01:10<00:00]", 100),
        ("embedding 12 passages ...", None),
        ("coverage is 45% of the corpus", None),
    ],
)
def test_bar_percent_reads_tqdm_bars_only(line: str, percent: int | None) -> None:
    assert bar_percent(line) == percent


def test_clock_text_reads_like_a_stopwatch() -> None:
    assert [clock_text(s) for s in (0, 5, 65, 3725)] == [
        "0:00",
        "0:05",
        "1:05",
        "1:02:05",
    ]


def test_index_progress_counts_stages_and_edge_layers() -> None:
    progress = IndexProgress(label="indexing docs", started=0.0)
    assert progress.parts(1.0) == ["indexing docs", "starting", "0:01"]
    assert not progress.feed("3 document(s), 12 unit(s) -> out")
    assert progress.parts(2.0)[1] == "stage 1/4 read"
    progress.feed("loading the embedding model ...")
    assert progress.parts(3.0)[1] == "loading the embedding model", "no number"
    progress.feed("embedding 12 passages ...")
    assert progress.feed("Batches:  50%|#####     | 1/2"), "a bar is consumed"
    assert progress.parts(5.0) == ["indexing docs", "stage 2/4 embed", "0:05", "50%"]
    progress.feed("extracting entities from 12 passages (spaCy bulk) ...")
    assert progress.percent is None, "a new stage forgets the old bar"
    progress.feed("semantic layer: 40 edges")
    progress.feed("entity edges exist, skipping")
    assert progress.parts(9.0)[1] == "stage 4/4 edges 2/4"


def test_the_proposition_layer_adds_a_stage() -> None:
    progress = IndexProgress(label="x", started=0.0, propositions=True)
    progress.feed("embedding 12 passages ...")
    assert progress.parts(0.0)[1] == "stage 3/5 embed"


def test_a_plain_job_shows_its_name_its_clock_and_its_bar() -> None:
    progress = JobProgress(label="installing [index]", started=10.0)
    assert not progress.feed("Collecting numpy")
    assert progress.feed("Downloading  30%|###       |")
    assert progress.parts(75.0) == ["installing [index]", "1:05", "30%"]
