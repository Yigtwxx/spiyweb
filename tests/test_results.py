"""Reading a played query from its record: labels, detail, honesty, Markdown."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from conftest import canonical_record
from spiyweb.animate import Glyphs, Style
from spiyweb.config import WatchConfig
from spiyweb.results import (
    honesty_lines,
    markdown,
    passage_detail,
    preview,
    ranked_passages,
    short_id,
)
from spiyweb.trace import TraceCluster, TraceEvent, TraceNode

SGR = re.compile(r"\x1b\[[0-9;]*m")


def plain(*, unicode: bool = True) -> Style:
    return Style(
        color=False,
        glyphs=Glyphs.unicode() if unicode else Glyphs.ascii(),
        config=WatchConfig(),
    )


@pytest.mark.parametrize(
    ("node_id", "chars", "shown"),
    [
        ("notes/2024/meeting.md:3", 12, "meeting:3"),
        ("shoreham.md:1", 12, "shoreham:1"),
        ("a-very-long-file-name.txt:12", 12, "a-very-l…:12"),
        ("A", 12, "A"),
        ("Wardenclyffe Tower", 12, "Wardenclyff…"),
        ("docs\\deep\\notes.md:0", 12, "notes:0"),
        (".md:4", 12, ".md:4"),
    ],
)
def test_short_id_keeps_the_position_that_tells_passages_apart(
    node_id: str, chars: int, shown: str
) -> None:
    assert short_id(node_id, chars, "…") == shown
    assert len(short_id(node_id, chars, "…")) <= chars


def test_preview_folds_whitespace_and_cuts_with_the_glyph_set() -> None:
    assert preview("one\n two   three", 40, "…") == "one two three"
    assert preview("abcdefghij", 8, "...") == "abcde..."


def test_ranked_passages_are_the_bars_order_without_the_ghosts() -> None:
    ids = [node.id for node in ranked_passages(canonical_record())]
    assert ids == ["A", "C", "D", "B", "F"], "A_dup is a vote, not a passage"


def _record_with_text() -> object:
    record = canonical_record()
    nodes = tuple(
        replace(node, text=f"Passage {node.id} says something about the tower.")
        for node in record.nodes
    )
    return replace(record, nodes=nodes)


def test_passage_detail_names_the_text_the_path_and_the_votes() -> None:
    record = _record_with_text()
    by_id = {node.id: node for node in record.nodes}  # type: ignore[attr-defined]
    shown = "\n".join(passage_detail(record, by_id["D"], plain(), width=80))  # type: ignore[arg-type]
    assert "Passage D says something" in shown
    assert "question → A → D" in shown
    seed = "\n".join(passage_detail(record, by_id["A"], plain(), width=80))  # type: ignore[arg-type]
    assert "first contact - similarity 0.90" in seed and "2 votes" in seed


def test_passage_detail_says_where_a_copy_came_from() -> None:
    record = canonical_record(
        nodes=(
            TraceNode(
                id="a:0", source_id="a", layer="chunk", energy=5.0, hop=0, votes=2
            ),
            TraceNode(
                id="b:0",
                source_id="b",
                layer="chunk",
                energy=0.0,
                hop=-1,
                votes=1,
                suppressed_by="a:0",
            ),
        ),
        paths=(),
    )
    shown = "\n".join(passage_detail(record, record.nodes[0], plain(), width=80))
    assert "also in" in shown and "b" in shown


def test_a_disputed_passage_says_both_sides_are_kept() -> None:
    record = canonical_record()
    node = replace(record.nodes[0], disputed=True)
    shown = "\n".join(passage_detail(record, node, plain(), width=80))
    assert "both are kept" in shown


def test_passage_detail_is_pure_ascii_with_the_ascii_glyphs() -> None:
    record = _record_with_text()
    node = record.nodes[2]  # type: ignore[attr-defined]
    lines = passage_detail(record, node, plain(unicode=False), width=60)  # type: ignore[arg-type]
    assert all(ord(char) < 128 for line in lines for char in line)


def test_a_plain_run_has_nothing_to_confess() -> None:
    assert honesty_lines(canonical_record()) == []


def test_a_contradiction_is_reported_once_with_both_sides_kept() -> None:
    record = canonical_record(
        events=(
            TraceEvent(kind="conflict", node="A", other="C", amount=0.4, hop=1),
            TraceEvent(kind="conflict", node="C", other="A", amount=0.4, hop=1),
        )
    )
    found = honesty_lines(record)
    assert len(found) == 1
    tone, text = found[0]
    assert tone == "warn" and "A and C contradict" in text and "both are kept" in text


def test_a_disputed_claim_uses_the_librarys_own_template() -> None:
    record = canonical_record(
        events=(TraceEvent(kind="polarity", node="F", amount=1.25, hop=2),)
    )
    (_tone, text), *_ = honesty_lines(record)
    assert text.startswith("The corpus disputes this claim: atom F")
    assert "1.25" in text


def test_two_dense_unbridged_themes_are_a_gap_and_two_themes() -> None:
    record = canonical_record(
        clusters=(
            TraceCluster(
                nodes=("A", "B", "D"), energy=9.0, energy_share=0.6, top_node="A"
            ),
            TraceCluster(
                nodes=("C", "F", "E"), energy=6.0, energy_share=0.4, top_node="C"
            ),
        )
    )
    texts = [text for _, text in honesty_lines(record)]
    assert any("A" in t and "C" in t and "theme" not in t for t in texts), texts
    assert any("2 separate themes, led by A, C" in t for t in texts)


def test_markdown_holds_the_question_every_passage_and_the_findings() -> None:
    record = replace(
        _record_with_text(),  # type: ignore[type-var]
        events=(TraceEvent(kind="conflict", node="A", other="C", amount=0.1, hop=0),),
    )
    page = markdown(record)  # type: ignore[arg-type]
    assert page.startswith("# which tower did tesla build")
    assert "1. **A**" in page and "Passage F says" in page
    assert "question -> A -> D" in page
    assert "## Also found" in page and "contradict" in page


def test_novelty_counts_what_only_one_side_found_at_the_same_k() -> None:
    from spiyweb.results import novelty

    found = novelty(["A", "C", "D", "B", "F"], ["A", "C", "X", "Y", "Z"], 5)
    assert found.only_web == ("D", "B", "F")
    assert found.only_topk == ("X", "Y", "Z")
    assert found.both == 2


def test_novelty_folds_propositions_into_their_passage() -> None:
    from spiyweb.results import novelty

    found = novelty(["d:0#p1", "d:0#p2", "e:0"], ["d:0", "f:0"], 2)
    assert found.only_web == ("e:0",) and found.both == 1
