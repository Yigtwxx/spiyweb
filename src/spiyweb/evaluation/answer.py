"""Answer quality: one fixed reader over each system's retrieved passages.

Retrieval recall says whether the right passages came back; the user of a
RAG system sees the answer. This module puts every system's top-k passages
in front of the same LLM with the same prompt and scores the reply with the
two standard extractive-QA metrics - exact match and token F1, under the
SQuAD normalisation (lower case, punctuation and articles stripped,
whitespace collapsed) - against the gold answer and its aliases.

Because the reader, prompt and k are fixed, the only thing that differs
between two systems' EM/F1 is the context they retrieved. That is the
comparison this module exists for; it is not a QA leaderboard number.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.config import ReaderConfig
from spiyweb.evaluation.metrics import passages_at_k
from spiyweb.prompts import READER_PROMPT

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from spiyweb.evaluation.datasets import MusiqueDataset
    from spiyweb.llm import LLMClient

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = frozenset(string.punctuation)
_ANSWER_LABEL = re.compile(r"^\s*(?:final\s+)?answer\s*[:\-]\s*", re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lower, no punctuation, no articles, one space."""
    lowered = text.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    no_articles = _ARTICLES.sub(" ", no_punctuation)
    return " ".join(no_articles.split())


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    """1.0 when the normalised prediction equals any normalised gold."""
    target = normalize_answer(prediction)
    return float(any(target == normalize_answer(gold) for gold in golds))


def _token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        # Both empty is agreement; one empty is a miss - the SQuAD rule.
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, golds: Sequence[str]) -> float:
    """Best token-level F1 of the prediction against any gold."""
    return max((_token_f1(prediction, gold) for gold in golds), default=0.0)


def parse_reader_answer(reply: str, max_words: int) -> str:
    """The answer span from a reader reply.

    Only the first non-empty line counts; an "Answer:" label, wrapping quotes
    and a trailing full stop are dropped, and the span is capped at
    `max_words` words. Unlike the chained intermediate answer there is no
    NONE escape: the prompt asks for a best guess, and an empty string here
    simply scores as a miss.
    """
    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    line = _ANSWER_LABEL.sub("", lines[0]).strip().strip("\"'`").strip()
    line = line.rstrip(".").strip()
    return " ".join(line.split()[:max_words])


def format_passages(titles: Sequence[str], texts: Sequence[str]) -> str:
    """The reader's context block: numbered, titled passages."""
    return "\n\n".join(
        f"Passage {number}: {title}\n{text}"
        for number, (title, text) in enumerate(zip(titles, texts, strict=True), 1)
    )


@dataclass(frozen=True)
class ReadAnswer:
    """One system's answer to one question, scored."""

    system: str
    passages: tuple[str, ...]
    answer: str
    em: float
    f1: float


def read_answer(
    llm: LLMClient,
    question: str,
    ranking: Sequence[str],
    dataset: MusiqueDataset,
    golds: Sequence[str],
    system: str,
    config: ReaderConfig,
) -> ReadAnswer:
    """Ask the reader `question` over `ranking`'s top-k passages and score it."""
    passages = passages_at_k(ranking, config.k)
    prompt = READER_PROMPT.format(
        passages=format_passages(
            [dataset.titles[passage] for passage in passages],
            [dataset.texts[passage] for passage in passages],
        ),
        question=question,
    )
    answer = parse_reader_answer(llm.complete(prompt), config.max_answer_words)
    return ReadAnswer(
        system=system,
        passages=tuple(passages),
        answer=answer,
        em=exact_match(answer, golds),
        f1=token_f1(answer, golds),
    )


def answer_records(
    records: Sequence[Mapping[str, object]],
    dataset: MusiqueDataset,
    llm: LLMClient,
    config: ReaderConfig | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    """Read every record's systems; one output row per question.

    A system whose ranking is absent from a record (the iterative column of
    a run that skipped the baseline) is skipped for that record, never read
    over an empty context.
    """
    cfg = config if config is not None else ReaderConfig()
    questions = {question.id: question for question in dataset.questions}
    rows: list[dict[str, object]] = []
    for number, record in enumerate(records, start=1):
        question = questions[str(record["id"])]
        golds = (question.answer, *question.answer_aliases)
        row: dict[str, object] = {"id": question.id, "gold": list(golds)}
        for system in cfg.systems:
            ranking = record.get(system)
            if not ranking:
                continue
            read = read_answer(
                llm,
                question.question,
                ranking,  # type: ignore[arg-type]
                dataset,
                golds,
                system,
                cfg,
            )
            row[system] = {"answer": read.answer, "em": read.em, "f1": read.f1}
        rows.append(row)
        if number % 100 == 0:
            log(f"read {number}/{len(records)} questions")
    return rows


def summarize_answers(
    rows: Sequence[Mapping[str, object]], config: ReaderConfig | None = None
) -> dict[str, object]:
    """Mean EM/F1 per system and the web's paired differences against each
    rival, over the questions where both systems were read.

    The interval is the protocol's paired bootstrap (`evaluation/stats.py`),
    the same one every retrieval number carries.
    """
    from spiyweb.evaluation.stats import paired_ci

    cfg = config if config is not None else ReaderConfig()
    present = [system for system in cfg.systems if any(system in row for row in rows)]
    summary: dict[str, object] = {"questions": len(rows), "k": cfg.k}
    for system in present:
        read = [row[system] for row in rows if system in row]
        summary[system] = {
            metric: sum(float(item[metric]) for item in read) / len(read)  # type: ignore[index]
            for metric in ("em", "f1")
        }
    if "web" in present:
        for rival in (system for system in present if system != "web"):
            both = [row for row in rows if "web" in row and rival in row]
            for metric in ("em", "f1"):
                mean, low, high, p = paired_ci(
                    [float(row["web"][metric]) for row in both],  # type: ignore[index]
                    [float(row[rival][metric]) for row in both],  # type: ignore[index]
                )
                summary[f"web_vs_{rival}_{metric}"] = {
                    "mean": mean,
                    "ci": [low, high],
                    "p": p,
                }
    return summary
