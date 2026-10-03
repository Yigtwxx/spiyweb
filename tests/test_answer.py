"""Answer quality: SQuAD metrics, reply parsing, the reader loop, the summary."""

from __future__ import annotations

import pytest

from spiyweb.config import ReaderConfig
from spiyweb.evaluation.answer import (
    answer_records,
    exact_match,
    format_passages,
    normalize_answer,
    parse_reader_answer,
    summarize_answers,
    token_f1,
)
from spiyweb.evaluation.datasets import MusiqueDataset, MusiqueQuestion


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("  an   Apple, a day ", "apple day"),
        ("Miquette Giraudy.", "miquette giraudy"),
    ],
)
def test_normalize_answer_follows_squad(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_exact_match_accepts_any_gold_after_normalisation() -> None:
    assert exact_match("the Orion Pictures", ["Orion Pictures"]) == 1.0
    assert exact_match("Orion", ["Orion Pictures", "Orion Corp"]) == 0.0
    assert exact_match("NYC", ["New York City", "NYC"]) == 1.0, "aliases count"


def test_token_f1_is_partial_credit_and_takes_the_best_gold() -> None:
    assert token_f1("Steve Hillage", ["Steve Hillage"]) == 1.0
    assert token_f1("Hillage", ["Steve Hillage"]) == pytest.approx(2 / 3)
    assert token_f1("Paris", ["London", "Paris France"]) == pytest.approx(2 / 3)
    assert token_f1("", ["Paris"]) == 0.0, "an empty reply is a miss"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Answer: Mike Medavoy.\nBecause the passage says so.", "Mike Medavoy"),
        ('"Orion Pictures"', "Orion Pictures"),
        ("\n\nyes\n", "yes"),
        ("", ""),
    ],
)
def test_parse_reader_answer_keeps_only_the_span(reply: str, expected: str) -> None:
    assert parse_reader_answer(reply, max_words=12) == expected


def test_parse_reader_answer_caps_the_word_count() -> None:
    assert parse_reader_answer("one two three four", max_words=2) == "one two"


def test_format_passages_numbers_and_titles_each_passage() -> None:
    block = format_passages(["A", "B"], ["alpha text", "beta text"])
    assert block == "Passage 1: A\nalpha text\n\nPassage 2: B\nbeta text"


class ScriptedLLM:
    """Answers by looking for a marker passage title in the prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Answer: Mike Medavoy" if "Mike Medavoy" in prompt else "Orion"


def _dataset() -> MusiqueDataset:
    question = MusiqueQuestion(
        id="2hop__q1",
        question="Who founded the company that distributed UHF?",
        answer="Mike Medavoy",
        hops=2,
        gold_ids=("d1:0", "d2:0"),
        bridge_gold_ids=("d1:0",),
        answer_aliases=("Morris Mike Medavoy",),
    )
    return MusiqueDataset(
        documents=(),
        titles={"d1:0": "UHF (film)", "d2:0": "Mike Medavoy", "d3:0": "Other"},
        texts={"d1:0": "Orion Pictures", "d2:0": "co-founder of Orion", "d3:0": "x"},
        questions=(question,),
    )


def test_answer_records_reads_each_system_over_its_own_passages() -> None:
    record = {
        "id": "2hop__q1",
        "topk": ["d1:0", "d3:0"],
        "web": ["d1:0", "d2:0#p0", "d2:0"],
        "iterative": None,
    }
    llm = ScriptedLLM()
    rows = answer_records([record], _dataset(), llm, ReaderConfig(k=2), log=print)

    row = rows[0]
    assert row["gold"] == ["Mike Medavoy", "Morris Mike Medavoy"]
    assert row["web"] == {"answer": "Mike Medavoy", "em": 1.0, "f1": 1.0}, (
        "a proposition folds into its parent passage, as in every metric"
    )
    assert row["topk"]["em"] == 0.0
    assert "iterative" not in row, "a missing ranking is skipped, not read empty"
    assert len(llm.prompts) == 2


def test_summarize_answers_reports_means_and_paired_differences() -> None:
    rows = [
        {"id": "a", "web": {"em": 1.0, "f1": 1.0}, "topk": {"em": 0.0, "f1": 0.5}},
        {"id": "b", "web": {"em": 0.0, "f1": 0.5}, "topk": {"em": 0.0, "f1": 0.0}},
    ]
    summary = summarize_answers(rows, ReaderConfig(systems=("topk", "web")))

    assert summary["web"] == {"em": 0.5, "f1": 0.75}
    assert summary["topk"] == {"em": 0.0, "f1": 0.25}
    assert summary["web_vs_topk_f1"]["mean"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("field_name", "value"), [("k", 0), ("max_answer_words", 0), ("systems", ())]
)
def test_reader_config_rejects_impossible_values(
    field_name: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ReaderConfig(**{field_name: value})
