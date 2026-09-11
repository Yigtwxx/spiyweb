"""Frames of one query: pure lines, checked without a terminal."""

from __future__ import annotations

import re

import pytest

from conftest import canonical_record
from spiyweb.animate import (
    Glyphs,
    Style,
    block_height,
    ghosts,
    ledger_line,
    query_block,
    ranking,
    ring_map,
    visible,
)
from spiyweb.config import WatchConfig
from spiyweb.terminal import printed_width

SGR = re.compile(r"\x1b\[[0-9;]*m")


def style(*, color: bool = False, unicode: bool = True, **knobs: object) -> Style:
    return Style(
        color=color,
        glyphs=Glyphs.unicode() if unicode else Glyphs.ascii(),
        config=WatchConfig(**knobs),  # type: ignore[arg-type]
    )


def test_hop_zero_shows_only_the_seeds() -> None:
    assert [n.id for n in visible(canonical_record(), 0)] == ["A", "C"]


def test_each_hop_reveals_its_ring_and_the_last_shows_everything() -> None:
    record = canonical_record()
    assert [n.id for n in visible(record, 1)] == ["A", "C", "D", "B"]
    assert [n.id for n in visible(record, 2)] == ["A", "C", "D", "B", "F"]


def test_a_suppressed_atom_reads_as_a_vote_not_a_loss() -> None:
    record = canonical_record()
    assert [g.id for g in ghosts(record, 0)] == ["A_dup"]
    rows = ranking(record, 2, 1.0, style())
    ghost = next(row for row in rows if "A_dup" in row)
    assert "duplicate of A" in ghost and "vote +1" in ghost


def test_converging_evidence_is_tagged_on_d_once_its_hop_has_settled() -> None:
    record = canonical_record()
    early = "\n".join(ranking(record, 1, 0.2, style()))
    late = "\n".join(ranking(record, 1, 0.8, style()))
    assert "converging" not in early
    assert "converging" in late
    assert late.index("D") < late.index("B"), "D outranks B - the §2.6 promotion"


def test_bars_grow_in_during_the_hop_they_arrive() -> None:
    record = canonical_record()
    start = next(r for r in ranking(record, 1, 0.0, style()) if " D " in r)
    end = next(r for r in ranking(record, 1, 1.0, style()) if " D " in r)
    assert "0.00" in start
    assert "2.88" in end


def test_plain_mode_emits_no_escape_codes_and_colour_does() -> None:
    record = canonical_record()
    plain = query_block(record, 2, 1.0, style(), width=120, with_map=True, final=True)
    assert not any(SGR.search(line) for line in plain)
    painted = query_block(
        record, 2, 1.0, style(color=True), width=120, with_map=True, final=True
    )
    assert any(SGR.search(line) for line in painted)


@pytest.mark.parametrize("width", [60, 80, 100, 140])
@pytest.mark.parametrize("unicode", [True, False])
def test_no_line_exceeds_the_width(width: int, unicode: bool) -> None:
    record = canonical_record()
    for hop in range(record.hops_used + 1):
        block = query_block(
            record, hop, 0.5, style(unicode=unicode), width=width, with_map=True,
            final=False,
        )  # fmt: skip
        assert all(printed_width(line) <= width for line in block)


def test_a_narrow_terminal_drops_the_map_and_keeps_the_ranking() -> None:
    record = canonical_record()
    narrow = query_block(record, 2, 1.0, style(), width=70, with_map=True, final=True)
    wide = query_block(record, 2, 1.0, style(), width=140, with_map=True, final=True)
    assert not any(Glyphs.unicode().seed in line for line in narrow)
    assert any(Glyphs.unicode().seed in line for line in wide)
    assert any(" A " in line for line in narrow)


def test_the_map_does_not_move_between_hops() -> None:
    record = canonical_record()

    def where(hop: int) -> tuple[int, int]:
        lines = ring_map(record, hop, 1.0, style(), cols=60, rows=13)
        for row, line in enumerate(lines):
            col = line.find(" A")
            if col >= 0:
                return row, col
        raise AssertionError("A not drawn")

    assert where(0) == where(1) == where(2)


def test_the_final_ledger_line_names_the_stop_reason() -> None:
    record = canonical_record()
    line = ledger_line(record, 2, 1.0, style(), final=True)
    assert "stopped: threshold" in line and "hop 2/2" in line
    live = ledger_line(record, 1, 0.3, style(), final=False)
    assert "spreading" in live


def test_block_height_matches_what_is_drawn() -> None:
    record = canonical_record()
    for with_map in (True, False):
        drawn = query_block(
            record, 2, 1.0, style(), width=140, with_map=with_map, final=True
        )
        assert block_height(record, style(), with_map=with_map) == len(drawn)


def test_a_record_with_nothing_activated_says_so() -> None:
    record = canonical_record(nodes=(), hops_used=0, node_count=0)
    rows = ranking(record, 0, 1.0, style())
    assert rows == ["  nothing activated"]
    assert ring_map(record, 0, 1.0, style(), cols=60, rows=13) == []
