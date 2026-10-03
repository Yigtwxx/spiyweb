"""Reading a played query: its passages and what the run had to say.

Everything here takes a `TraceRecord` - the self-contained picture of one
call - never an `Answer`. So a query the monitor ran, one the attached
application made, a `/replay` and the `/demo` recordings all read the same
way, and none of it needs numpy or an open index.

`honesty_lines` is the part `top-k` cannot produce: contradictions kept on
both sides, claims the corpus disputes, themes with no bridge between them.
Each appears only when the run actually produced it - an empty section is
noise, and noise teaches people to stop reading.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.output import DISPUTE_TEMPLATE, ThemeCluster, gap_warnings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spiyweb.animate import Style
    from spiyweb.trace import TraceNode, TraceRecord

__all__ = [
    "Novelty",
    "honesty_lines",
    "markdown",
    "novelty",
    "passage_detail",
    "passage_of",
    "preview",
    "ranked_passages",
    "short_id",
]

TEXT_EXTENSIONS = (".markdown", ".md", ".txt", ".rst")
"""Dropped from a label: on a corpus of text files they say nothing."""


def ranked_passages(record: TraceRecord) -> list[TraceNode]:
    """Activated atoms strongest first - the order the bars are drawn in."""
    alive = [node for node in record.nodes if node.energy > 0.0]
    return sorted(alive, key=lambda node: (-node.energy, node.id))


def preview(text: str, chars: int, ellipsis: str) -> str:
    """One line of a passage: whitespace folded, cut at `chars`."""
    flat = " ".join(text.split())
    if len(flat) <= chars:
        return flat
    return flat[: max(0, chars - len(ellipsis))] + ellipsis


def short_id(node_id: str, chars: int, ellipsis: str) -> str:
    """A node id that fits `chars`, keeping what tells passages apart.

    `notes/2024/meeting.md:3` becomes `meeting:3`: the folder is context
    the ranking does not need, the extension is the same on every row, and
    the position after the colon is the one part two passages of one file
    never share - so when the name still does not fit, it is the name that
    gets cut, never the position.
    """
    name = node_id.replace("\\", "/").rsplit("/", 1)[-1]
    stem, colon, position = name.rpartition(":")
    if not colon:
        stem, position = name, ""
    for extension in TEXT_EXTENSIONS:
        if stem.lower().endswith(extension) and len(stem) > len(extension):
            stem = stem[: -len(extension)]
            break
    tail = f":{position}" if colon else ""
    whole = stem + tail
    if len(whole) <= chars:
        return whole
    room = chars - len(ellipsis) - len(tail)
    if room < 1:
        return whole[: max(0, chars - len(ellipsis))] + ellipsis
    return stem[:room] + ellipsis + tail


def _path_to(record: TraceRecord, node_id: str) -> tuple[str, ...]:
    for path in record.paths:
        if path.node == node_id:
            return path.steps
    return ()


def _converging(record: TraceRecord, node_id: str) -> int:
    for path in record.paths:
        if path.node == node_id:
            return path.converging
    return 0


def passage_detail(
    record: TraceRecord, node: TraceNode, style: Style, *, width: int
) -> list[str]:
    """Everything the record knows about one passage, for `/show`."""
    g = style.glyphs
    dot = f" {g.dot} "
    facts = [node.source_id, node.layer, f"hop {node.hop}", f"energy {node.energy:.2f}"]
    if node.votes > 1:
        facts.append(f"{node.votes} votes")
    lines = [
        style.paint(node.id, "accent", "bold")
        + "  "
        + style.paint(dot.join(facts), "muted")
    ]
    text = " ".join(node.text.split())
    if text:
        lines += ["  " + row for row in textwrap.wrap(text, max(20, width - 4))]
    else:
        lines.append(
            style.paint("  (the record holds no text for this passage)", "dim")
        )
    steps = _path_to(record, node.id)
    if node.hop == 0 and node.seed_similarity is not None:
        how = f"first contact - similarity {node.seed_similarity:.2f} to the question"
    elif steps:
        how = f" {g.to} ".join(["question", *steps])
    else:
        how = ""
    if how:
        lines.append(style.paint("  reached  ", "muted") + how)
    converging = _converging(record, node.id)
    if converging > 1:
        lines.append(
            style.paint("  fed by   ", "muted")
            + f"{converging} atoms at once - converging evidence"
        )
    copies = [n for n in record.nodes if n.suppressed_by == node.id and n.energy <= 0]
    others = sorted({n.source_id for n in copies if n.source_id != node.source_id})
    if others:
        lines.append(
            style.paint("  also in  ", "muted")
            + ", ".join(others)
            + style.paint("  (copies folded into this one as votes)", "dim")
        )
    if node.disputed:
        lines.append(
            style.paint("  disputed ", "warn")
            + "another passage says the opposite - both are kept"
        )
    return lines


def honesty_lines(record: TraceRecord) -> list[tuple[str, str]]:
    """`(tone, text)` for what this run found that a ranking alone hides -
    and nothing at all when it found none of it."""
    out: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for event in record.events:
        if event.kind == "conflict" and event.other:
            pair = frozenset((event.node, event.other))
            if pair in seen:
                continue
            seen.add(pair)
            out.append(
                (
                    "warn",
                    f"{event.node} and {event.other} contradict each other - "
                    "both are kept, neither side is picked",
                )
            )
        elif event.kind == "polarity":
            out.append(
                (
                    "warn",
                    DISPUTE_TEMPLATE.format(
                        node=event.node, absorbed=event.amount, hop=event.hop
                    ),
                )
            )
    clusters = tuple(
        ThemeCluster(
            nodes=cluster.nodes,
            energy=cluster.energy,
            energy_share=cluster.energy_share,
            top_node=cluster.top_node,
            colors=cluster.colors,
        )
        for cluster in record.clusters
    )
    for gap in gap_warnings(clusters):
        out.append(("muted", gap.message))
    if len(clusters) >= 2:
        tops = ", ".join(cluster.top_node for cluster in clusters)
        out.append(("muted", f"{len(clusters)} separate themes, led by {tops}"))
    return out


def passage_of(node_id: str) -> str:
    """A proposition (`doc:0#p3`) stands for its parent passage (`doc:0`) -
    the folding rule the metrics and the baselines use."""
    return node_id.split("#", 1)[0]


def _first_passages(ids: Sequence[str], k: int) -> list[str]:
    out: list[str] = []
    for node_id in ids:
        passage = passage_of(node_id)
        if passage not in out:
            out.append(passage)
        if len(out) == k:
            break
    return out


@dataclass(frozen=True)
class Novelty:
    """The web's top `k` passages against plain top-`k` on the same question:
    Novelty@k, the serendipity half of the Phase 1 objective, on screen."""

    k: int
    only_web: tuple[str, ...]
    only_topk: tuple[str, ...]
    both: int


def novelty(web: Sequence[str], topk: Sequence[str], k: int) -> Novelty:
    """`web` and `topk` are rankings of node ids, best first."""
    web_k = _first_passages(web, k)
    topk_k = _first_passages(topk, k)
    return Novelty(
        k=k,
        only_web=tuple(p for p in web_k if p not in topk_k),
        only_topk=tuple(p for p in topk_k if p not in web_k),
        both=len(set(web_k) & set(topk_k)),
    )


def markdown(record: TraceRecord) -> str:
    """The record as a Markdown page someone can keep, read or paste."""
    lines = [
        f"# {record.query}",
        "",
        f"- profile: {record.profile or 'explore'}",
        f"- hops: {record.hops_used}, stopped: {record.stop_reason}",
        f"- recorded: {record.recorded_at}",
    ]
    if record.index:
        lines.append(f"- index: {record.index}")
    lines += ["", "## Passages", ""]
    for number, node in enumerate(ranked_passages(record), 1):
        steps = _path_to(record, node.id)
        reached = " -> ".join(["question", *steps]) if steps else "first contact"
        votes = f", {node.votes} votes" if node.votes > 1 else ""
        lines += [
            f"{number}. **{node.id}** - energy {node.energy:.2f}, "
            f"hop {node.hop}{votes} ({reached})",
            "",
            f"   {' '.join(node.text.split())}"
            if node.text
            else "   (no text recorded)",
            "",
        ]
    found = honesty_lines(record)
    if found:
        lines += ["## Also found", ""]
        lines += [f"- {text}" for _, text in found]
        lines.append("")
    return "\n".join(lines)
