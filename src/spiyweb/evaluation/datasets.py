"""MuSiQue-Ans loading: download, deterministic sampling, corpus pooling.

The dev split ships 20 candidate paragraphs per question with gold
`is_supporting` labels and a per-hop `question_decomposition` - exactly the
annotations support recall and bridge recall need, no extra judging.

The corpus follows the HippoRAG-comparable regime: sample N questions
deterministically, pool their candidate paragraphs, and deduplicate on
`(title, paragraph_text)` - the same Wikipedia paragraph recurs across
questions and must become ONE node, otherwise the graph is full of exact
duplicates the dedup mechanism does not exist yet to suppress. Pooled
distractors of other questions keep retrieval honest.

Everything here is stdlib: the download is one HTTPS GET of a public JSONL
file (CC BY 4.0), so neither `datasets` nor `huggingface_hub` enters the
dependency tree.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.config import EvaluationConfig
from spiyweb.nodes import DocumentInput, TextUnit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

_HOP_PREFIX = re.compile(r"^(\d+)hop")

_DOWNLOAD_TIMEOUT_SECONDS = 120.0

TWOWIKI_DEV_URL = (
    "https://huggingface.co/datasets/kamelliao/2wikimultihopqa/"
    "resolve/main/data/dev.json"
)
"""Original-format 2WikiMultihopQA dev split (JSON array), mirrored on HF."""

HOTPOTQA_DEV_URL = (
    "https://huggingface.co/datasets/RAGLAB/data/resolve/main/"
    "eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json"
)
"""Original-format HotpotQA dev distractor split, mirrored on HF (the
canonical curtis.ml.cmu.edu host is flaky and plain-HTTP)."""

_YES_NO_ANSWERS = frozenset({"yes", "no"})


@dataclass(frozen=True)
class MusiqueQuestion:
    """One evaluation question with its gold labels mapped to corpus ids.

    Attributes:
        id: Original MuSiQue id; the hop count is encoded in its prefix.
        question: The natural-language question.
        answer: Gold answer string (kept for the per-query record; retrieval
            metrics never read it).
        hops: Hop count parsed from the id prefix; equals the number of
            supporting paragraphs.
        gold_ids: Corpus chunk ids of the supporting paragraphs.
        bridge_gold_ids: Corpus chunk ids supporting every decomposition step
            EXCEPT the last - the intermediate documents the multi-hop claim
            is actually about.
        answer_aliases: Other accepted spellings of the answer (MuSiQue ships
            them; HotpotQA and 2Wiki do not). Read only by the answer-quality
            reader, which scores a reply against the answer and every alias.
    """

    id: str
    question: str
    answer: str
    hops: int
    gold_ids: tuple[str, ...]
    bridge_gold_ids: tuple[str, ...]
    answer_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class MusiqueDataset:
    """The pooled corpus and the sampled questions, ready for indexing.

    Attributes:
        documents: One `DocumentInput` per deduplicated paragraph (a single
            unit each, so chunk ids come out as `d00042:0`).
        titles: Corpus chunk id -> paragraph title.
        texts: Corpus chunk id -> paragraph text.
        questions: The sampled questions with corpus-mapped gold labels.
    """

    documents: tuple[DocumentInput, ...]
    titles: Mapping[str, str]
    texts: Mapping[str, str]
    questions: tuple[MusiqueQuestion, ...]


def _fetch_https(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"dataset url must use https, got {url!r}")
    # Scheme is pinned to https above, so urlopen never touches file:// etc.
    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        return response.read()


def download_musique(
    target: Path,
    url: str,
    fetch: Callable[[str], bytes] | None = None,
) -> Path:
    """Download the dataset file to `target` unless it already exists.

    The existence check makes the download stage naturally resumable; pass a
    `fetch` callable to test the flow without the network.
    """
    if target.exists():
        return target
    fetcher = fetch if fetch is not None else _fetch_https
    payload = fetcher(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def _parse_hops(question_id: str) -> int:
    match = _HOP_PREFIX.match(question_id)
    if match is None:
        raise ValueError(
            f"question id {question_id!r} carries no hop prefix; "
            "the file does not look like MuSiQue"
        )
    return int(match.group(1))


def _sample_ids(ids: list[str], config: EvaluationConfig) -> list[str]:
    """Deterministic sample over the LEXICOGRAPHICALLY SORTED id list.

    Sorting first is what makes the draw identical on every platform - set or
    file order must never leak into the experiment identity.
    """
    ordered = sorted(ids)
    if config.sample_size == 0 or config.sample_size >= len(ordered):
        return ordered
    rng = random.Random(config.sample_seed)
    return sorted(rng.sample(ordered, config.sample_size))


def load_dataset(path: Path, config: EvaluationConfig | None = None) -> MusiqueDataset:
    """Parse, sample, pool and map a MuSiQue-Ans JSONL file.

    Raises:
        ValueError: On records missing required fields or gold annotations -
            silence here would surface later as inexplicably broken metrics.
    """
    cfg = config if config is not None else EvaluationConfig()

    records: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = record.get("id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"line {line_number}: record has no usable 'id'")
            records[question_id] = record

    if not records:
        raise ValueError(f"{str(path)!r} contains no records")

    sampled_ids = _sample_ids(list(records), cfg)

    # Pool and deduplicate the candidate paragraphs of the sampled questions.
    keys: set[tuple[str, str]] = set()
    for question_id in sampled_ids:
        for paragraph in _paragraphs_of(records[question_id], question_id):
            keys.add((str(paragraph["title"]), str(paragraph["paragraph_text"])))

    chunk_id_of: dict[tuple[str, str], str] = {}
    documents: list[DocumentInput] = []
    titles: dict[str, str] = {}
    texts: dict[str, str] = {}
    for position, key in enumerate(sorted(keys)):
        title, text = key
        doc_id = f"d{position:05d}"
        chunk_id = f"{doc_id}:0"  # chunk_documents' id scheme, single unit
        chunk_id_of[key] = chunk_id
        documents.append(DocumentInput(source_id=doc_id, units=(TextUnit(text=text),)))
        titles[chunk_id] = title
        texts[chunk_id] = text

    questions: list[MusiqueQuestion] = []
    for question_id in sampled_ids:
        record = records[question_id]
        paragraphs = _paragraphs_of(record, question_id)
        by_idx = {int(paragraph["idx"]): paragraph for paragraph in paragraphs}

        gold_ids = tuple(
            chunk_id_of[(str(p["title"]), str(p["paragraph_text"]))]
            for p in paragraphs
            if bool(p["is_supporting"])
        )
        if not gold_ids:
            raise ValueError(
                f"question {question_id!r} has no supporting paragraph; "
                "the Ans split always has them"
            )

        decomposition = record.get("question_decomposition")
        if not isinstance(decomposition, list) or not decomposition:
            raise ValueError(f"question {question_id!r} has no question_decomposition")
        bridge_ids: list[str] = []
        for step in decomposition[:-1]:
            support_idx = step.get("paragraph_support_idx")
            if not isinstance(support_idx, int) or support_idx not in by_idx:
                raise ValueError(
                    f"question {question_id!r}: decomposition step has no "
                    f"usable paragraph_support_idx ({support_idx!r})"
                )
            paragraph = by_idx[support_idx]
            bridge_ids.append(
                chunk_id_of[(str(paragraph["title"]), str(paragraph["paragraph_text"]))]
            )

        questions.append(
            MusiqueQuestion(
                id=question_id,
                question=str(record["question"]),
                answer=str(record["answer"]),
                answer_aliases=tuple(
                    str(alias) for alias in record.get("answer_aliases") or ()
                ),
                hops=_parse_hops(question_id),
                gold_ids=gold_ids,
                bridge_gold_ids=tuple(bridge_ids),
            )
        )

    return MusiqueDataset(
        documents=tuple(documents),
        titles=titles,
        texts=texts,
        questions=tuple(questions),
    )


def _paragraphs_of(
    record: Mapping[str, object], question_id: str
) -> list[dict[str, object]]:
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError(f"question {question_id!r} has no paragraphs")
    return paragraphs


def load_2wiki(path: Path, config: EvaluationConfig | None = None) -> MusiqueDataset:
    """Parse, sample, pool and map a 2WikiMultihopQA dev JSON file.

    The cross-dataset generalisation check (measurement protocol): the winner
    configuration runs AS IS - nothing here may be tuned on this data. The
    return type is deliberately `MusiqueDataset`: the whole pipeline consumes
    it structurally, so one loader swap is the only difference between runs.

    Mapping notes:
    - Question ids are prefixed with the 2Wiki question type
      (`comparison__<id>`, `compositional__<id>`, ...), so per-type analysis
      falls out of `per_query.jsonl` for free. Sampling draws over these
      prefixed ids, sorted - the draw is deterministic, as everywhere.
    - `hops` = number of distinct supporting titles (2 for most types, 4 for
      bridge_comparison); 2Wiki ids carry no hop prefix to parse.
    - `bridge_gold_ids` is an APPROXIMATION, documented rather than hidden:
      2Wiki has no per-step paragraph annotation like MuSiQue's
      decomposition. Comparison-type and yes/no questions have no
      intermediate document at all, so bridge = all gold (bridge recall
      degenerates to support recall there). For the rest, bridge = the
      supporting paragraphs whose text does NOT contain the answer string
      (the intermediates), falling back to all gold if that empties.

    Raises:
        ValueError: On records missing required fields, or on a supporting
            fact naming a title absent from the question's own context.
    """
    return _load_hotpot_family(path, config, family="2Wiki")


def load_hotpotqa(path: Path, config: EvaluationConfig | None = None) -> MusiqueDataset:
    """Parse, sample, pool and map a HotpotQA dev distractor JSON file.

    The UNTOUCHED third dataset: after tour 13 was motivated by the 2Wiki
    diagnosis, 2Wiki no longer counts as a blind cross-dataset check - this
    loader exists so the generalisation claim can be sealed on data nothing
    was ever tuned against. The winner configuration runs AS IS here, same
    protocol as 2Wiki.

    HotpotQA invented the format 2Wiki copied (`_id`/`question`/`answer`/
    `type`/`context`/`supporting_facts`), so the mapping is shared:
    type-prefixed ids (`bridge__<id>`, `comparison__<id>`), `hops` = distinct
    supporting titles (always 2 here), and the same documented bridge
    approximation - comparison and yes/no questions have no intermediate
    document (bridge = all gold), the rest take the supporting paragraphs
    not containing the answer string.

    Raises:
        ValueError: On records missing required fields, or on a supporting
            fact naming a title absent from the question's own context.
    """
    return _load_hotpot_family(path, config, family="HotpotQA")


def _load_hotpot_family(
    path: Path, config: EvaluationConfig | None, family: str
) -> MusiqueDataset:
    """The shared loader body of the HotpotQA-format family (2Wiki, Hotpot)."""
    cfg = config if config is not None else EvaluationConfig()

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{str(path)!r} is not a non-empty {family} JSON array")

    records: dict[str, dict[str, object]] = {}
    for position, record in enumerate(payload):
        raw_id = record.get("_id")
        qtype = record.get("type")
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError(f"record {position}: no usable '_id'")
        if not isinstance(qtype, str) or not qtype:
            raise ValueError(f"record {position}: no usable 'type'")
        records[f"{qtype}__{raw_id}"] = record

    sampled_ids = _sample_ids(list(records), cfg)

    keys: set[tuple[str, str]] = set()
    context_of: dict[str, dict[str, str]] = {}
    for question_id in sampled_ids:
        by_title: dict[str, str] = {}
        for entry in _context_of(records[question_id], question_id):
            title, sentences = entry[0], entry[1]
            text = " ".join(str(sentence) for sentence in sentences)
            by_title.setdefault(str(title), text)
        context_of[question_id] = by_title
        keys.update((title, text) for title, text in by_title.items())

    chunk_id_of: dict[tuple[str, str], str] = {}
    documents: list[DocumentInput] = []
    titles: dict[str, str] = {}
    texts: dict[str, str] = {}
    for position, key in enumerate(sorted(keys)):
        title, text = key
        doc_id = f"d{position:05d}"
        chunk_id = f"{doc_id}:0"
        chunk_id_of[key] = chunk_id
        documents.append(DocumentInput(source_id=doc_id, units=(TextUnit(text=text),)))
        titles[chunk_id] = title
        texts[chunk_id] = text

    questions: list[MusiqueQuestion] = []
    for question_id in sampled_ids:
        record = records[question_id]
        by_title = context_of[question_id]
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"question {question_id!r} has no answer")

        supporting = record.get("supporting_facts")
        if not isinstance(supporting, list) or not supporting:
            raise ValueError(f"question {question_id!r} has no supporting_facts")
        gold_titles: list[str] = []
        for fact in supporting:
            title = str(fact[0])
            if title not in by_title:
                raise ValueError(
                    f"question {question_id!r}: supporting title {title!r} "
                    "is missing from its own context"
                )
            if title not in gold_titles:
                gold_titles.append(title)
        gold_ids = tuple(chunk_id_of[(title, by_title[title])] for title in gold_titles)

        qtype = question_id.split("__", 1)[0]
        if "comparison" in qtype or answer.strip().lower() in _YES_NO_ANSWERS:
            bridge_ids = gold_ids
        else:
            needle = answer.strip().lower()
            intermediates = tuple(
                chunk_id
                for title, chunk_id in zip(gold_titles, gold_ids, strict=True)
                if needle not in by_title[title].lower()
            )
            bridge_ids = intermediates if intermediates else gold_ids

        questions.append(
            MusiqueQuestion(
                id=question_id,
                question=str(record["question"]),
                answer=answer,
                hops=len(gold_ids),
                gold_ids=gold_ids,
                bridge_gold_ids=bridge_ids,
            )
        )

    return MusiqueDataset(
        documents=tuple(documents),
        titles=titles,
        texts=texts,
        questions=tuple(questions),
    )


def _context_of(record: Mapping[str, object], question_id: str) -> list[list[object]]:
    context = record.get("context")
    if not isinstance(context, list) or not context:
        raise ValueError(f"question {question_id!r} has no context")
    for entry in context:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(
                f"question {question_id!r}: context entry is not a "
                "[title, sentences] pair"
            )
    return context
