"""Traces: what actually happened, kept without keeping the index.

Phase 2.4. The viewer this feeds (Phase 2.5) reads a RECORDED call, never a
new one, and that is the whole reason the record has to stand on its own: a
trace that needs the graph and the text map re-loaded to be drawn is a second
copy of the index in disguise, which is exactly the shape D38 rejected.

So the load-bearing assertion here is self-containment - a record carries the
activated subgraph's edges and the passages' text - and the second one is the
ring buffer's promise that a long-running application never grows without
bound. The fifty-call test is Phase 2.4's stated exit criterion, written as a
test rather than a script so it keeps holding.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from spiyweb.config import ColoredRetrievalConfig, TraceConfig
from spiyweb.ledger import build_ledger
from spiyweb.trace import SCHEMA_VERSION, TraceRecord, TraceStore, load_traces

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spiyweb.session import SpiywebIndex


def test_tracing_is_on_in_memory_without_being_asked_for(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """The default is the in-memory ring buffer; nothing touches the disk."""
    index = open_tiny()
    assert index.traces.enabled
    assert index.traces.path is None

    index.retrieve("who raised the tower")

    assert len(index.traces) == 1
    assert index.traces.latest() is not None


def test_tracing_switches_off_from_config(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """Every mechanism is individually disableable (CLAUDE.md §6)."""
    index = open_tiny(TraceConfig(enabled=False))
    index.retrieve("who raised the tower")
    assert len(index.traces) == 0
    assert index.traces.latest() is None


def test_a_trace_carries_the_activated_subgraph_and_its_text(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """Self-containment: the viewer must not need the index to draw this."""
    index = open_tiny()
    answer = index.retrieve("what happened afterwards")
    record = index.traces.latest()
    assert record is not None

    drawn = {node.id for node in record.nodes}
    assert drawn >= {passage.node_id for passage in answer.passages}
    assert all(node.text for node in record.nodes if node.energy > 0.0)
    assert record.edges, "an activated subgraph with no edges cannot be drawn"
    for edge in record.edges:
        assert edge.source in drawn
        assert edge.target in drawn
        assert edge.weight > 0.0


def test_a_trace_reports_the_same_numbers_as_the_answer(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """The record is a copy of the run, not a second, differently-shaped one."""
    index = open_tiny()
    answer = index.retrieve("who raised the tower")
    record = index.traces.latest()
    assert record is not None

    energies = {node.id: node.energy for node in record.nodes if node.energy > 0.0}
    assert energies == pytest.approx(dict(answer.result.ranked()))
    assert record.stop_reason == answer.result.propagation.stop_reason
    assert record.dedup_mode == answer.dedup_mode
    assert record.hops_used == answer.confidence.hop_depth
    assert record.node_count == answer.confidence.node_count
    assert record.total_energy == pytest.approx(answer.confidence.total_energy)
    votes = answer.votes()
    for node in record.nodes:
        if node.energy > 0.0:
            expected = votes.get(node.source_id, votes.get(node.id, 1))
            assert node.votes == expected
    assert {path.node for path in record.paths} == set(energies)
    assert record.query == "who raised the tower"
    assert record.index == str(index.layout.root)


def test_a_trace_carries_its_own_energy_ledger(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """The reader has no graph, so the ledger cannot be its job to compute."""
    index = open_tiny()
    answer = index.retrieve("who raised the tower")
    record = index.traces.latest()
    assert record is not None and record.ledger is not None

    assert answer.config is not None
    expected = build_ledger(
        answer.result.propagation, index.graph, answer.config.propagation
    )
    assert record.ledger.injected == pytest.approx(expected.injected)
    assert record.ledger.held == pytest.approx(expected.held)
    assert record.ledger.dissipated == pytest.approx(expected.dissipated)
    assert record.ledger.destroyed == pytest.approx(expected.destroyed.total)
    assert record.ledger.balanced is expected.balanced
    # Nothing in this fixture destroys energy, so the book must close exactly.
    assert record.ledger.balanced, record.ledger.notes


def test_a_coloured_ledger_is_the_sum_of_its_colours(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """One book per colour: the merged view never distributed anything."""
    index = open_tiny()
    answer = index.retrieve_colored(
        {"tower": "who raised the tower", "after": "what happened afterwards"}
    )
    record = index.traces.latest()
    assert record is not None and record.ledger is not None

    books = [
        build_ledger(result, index.graph, ColoredRetrievalConfig().propagation)
        for _, result in sorted(answer.colored.per_color.items())
    ]
    assert record.ledger.injected == pytest.approx(sum(b.injected for b in books))
    assert record.ledger.held == pytest.approx(sum(b.held for b in books))
    assert record.ledger.balanced is all(b.balanced for b in books)


def test_a_trace_round_trips_through_json(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """JSONL is the on-disk form; a lossy dump is a broken viewer."""
    index = open_tiny()
    index.retrieve("what happened afterwards")
    record = index.traces.latest()
    assert record is not None

    payload = json.loads(json.dumps(record.to_dict()))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert TraceRecord.from_dict(payload) == record


def test_the_ring_buffer_forgets_the_oldest(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """A long-running application must not grow without bound."""
    index = open_tiny(TraceConfig(capacity=10))
    for number in range(50):
        index.retrieve(f"question number {number}")

    records = index.traces.records()
    assert len(records) == 10
    assert [record.sequence for record in records] == list(range(40, 50))
    assert records[-1] is index.traces.latest()


def test_fifty_calls_leave_fifty_correct_traces(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """Phase 2.4's exit criterion, as a test rather than a throwaway script."""
    index = open_tiny(TraceConfig(capacity=200))
    answers = [index.retrieve(f"question number {number}") for number in range(50)]

    records = index.traces.records()
    assert len(records) == 50
    assert len({record.trace_id for record in records}) == 50
    assert [record.sequence for record in records] == list(range(50))
    for answer, record in zip(answers, records, strict=True):
        assert record.query == answer.query
        assert record.node_count == answer.confidence.node_count
        assert {node.id for node in record.nodes if node.energy > 0.0} == {
            node for node, _ in answer.result.ranked()
        }
        assert record.elapsed_ms >= 0.0
        assert TraceRecord.from_dict(record.to_dict()) == record


def test_nothing_reaches_the_disk_until_a_directory_is_given(
    open_tiny: Callable[..., SpiywebIndex], tmp_path: Path
) -> None:
    """Passage text hits the disk here, so writing is an explicit choice (D38)."""
    quiet = open_tiny()
    quiet.retrieve("who raised the tower")
    assert not list(tmp_path.iterdir())

    loud = open_tiny(TraceConfig(directory=tmp_path))
    assert loud.traces.path is not None
    for number in range(3):
        loud.retrieve(f"question number {number}")

    lines = loud.traces.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert load_traces(loud.traces.path) == loud.traces.records()


def test_text_can_be_left_out_of_the_record(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """The switch exists because the record is a copy of the corpus otherwise."""
    index = open_tiny(TraceConfig(include_texts=False))
    index.retrieve("who raised the tower")
    record = index.traces.latest()
    assert record is not None
    assert all(node.text == "" for node in record.nodes)


def test_text_can_be_truncated(open_tiny: Callable[..., SpiywebIndex]) -> None:
    index = open_tiny(TraceConfig(text_chars=8))
    index.retrieve("who raised the tower")
    record = index.traces.latest()
    assert record is not None
    assert all(len(node.text) <= 8 for node in record.nodes)


def test_a_coloured_run_traces_its_bridges(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """Colours are the differentiator; a trace that drops them drops the point."""
    index = open_tiny()
    answer = index.retrieve_colored(
        {"tower": "who raised the tower", "after": "what happened afterwards"}
    )
    record = index.traces.latest()
    assert record is not None
    assert record.kind == "colored"
    assert record.parts == {
        "tower": "who raised the tower",
        "after": "what happened afterwards",
    }
    assert dict(record.bridges) == {
        node: tuple(colors) for node, colors in answer.bridges.items()
    }
    assert TraceRecord.from_dict(record.to_dict()) == record


def test_suppressed_atoms_stay_in_the_record(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """A cut edge is the vote mechanism firing; the picture must show it."""
    index = open_tiny()
    answer = index.retrieve("who raised the tower")
    record = index.traces.latest()
    assert record is not None
    cuts = {
        **answer.result.contact_suppressed,
        **answer.result.propagation.suppressed,
    }
    traced = {
        node.id: node.suppressed_by for node in record.nodes if node.suppressed_by
    }
    assert traced == cuts


def test_a_store_can_be_cleared_and_reused() -> None:
    store = TraceStore(TraceConfig(capacity=4))
    assert len(store) == 0
    store.clear()
    assert store.latest() is None
    assert store.records() == ()


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        TraceConfig(capacity=0)


def test_text_chars_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="text_chars"):
        TraceConfig(text_chars=-1)


_READER_PROBE = """
import sys

BANNED = (
    "numpy", "faiss", "torch", "spacy", "sentence_transformers",
    "transformers", "streamlit", "fastapi", "uvicorn", "pydantic",
)


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BANNED:
            raise ImportError(name + " is blocked by the zero-dependency guard")
        return None


sys.meta_path.insert(0, _Blocker())
for name in [n for n in sys.modules if n.split(".")[0] in BANNED]:
    del sys.modules[name]

from spiyweb.trace import load_traces

records = load_traces(sys.argv[1])
assert len(records) == 2, len(records)
assert records[0].nodes and records[0].edges
assert any(node.text for node in records[0].nodes)
print("ok")
"""


def test_a_trace_file_is_readable_without_the_index_or_a_single_dependency(
    open_tiny: Callable[..., SpiywebIndex], tmp_path: Path
) -> None:
    """D38's whole point: the viewer opens a file, not an index.

    Blocking numpy and FAISS is how "no second copy of the index" stops being
    a claim and starts being a check - a reader that quietly needed the store
    back would fail here and nowhere else.
    """
    index = open_tiny(TraceConfig(directory=tmp_path))
    index.retrieve("who raised the tower")
    index.retrieve("what happened afterwards")
    assert index.traces.path is not None

    done = subprocess.run(
        [sys.executable, "-c", _READER_PROBE, str(index.traces.path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_a_half_written_last_line_is_tolerated(
    open_tiny: Callable[..., SpiywebIndex], tmp_path: Path
) -> None:
    """The writer is usually still running - that is the normal case.

    A record mid-append is a truncated final line, and it will be complete a
    moment later. A reader that raised on it would fail on every busy
    application it was pointed at.
    """
    index = open_tiny(TraceConfig(directory=tmp_path))
    index.retrieve("who raised the tower")
    index.retrieve("what happened afterwards")
    path = index.traces.path
    assert path is not None

    whole = path.read_text(encoding="utf-8")
    path.write_text(whole + '{"trace_id": "half', encoding="utf-8")
    assert len(load_traces(path)) == 2


def test_a_damaged_line_in_the_middle_is_loud(
    open_tiny: Callable[..., SpiywebIndex], tmp_path: Path
) -> None:
    """That is data loss; a skipped record leaves a hole nobody can see."""
    index = open_tiny(TraceConfig(directory=tmp_path))
    index.retrieve("who raised the tower")
    index.retrieve("what happened afterwards")
    path = index.traces.path
    assert path is not None

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(["{ broken", *lines]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is not a readable trace record"):
        load_traces(path)


def test_a_trace_written_by_a_newer_build_is_refused(tmp_path: Path) -> None:
    """A reader that guesses at an unknown schema shows confident nonsense."""
    import json as _json

    path = tmp_path / "traces.jsonl"
    path.write_text(
        _json.dumps({"schema_version": SCHEMA_VERSION + 1})
        + "\n"
        + _json.dumps({"schema_version": SCHEMA_VERSION + 1})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be read by this build"):
        load_traces(path)
