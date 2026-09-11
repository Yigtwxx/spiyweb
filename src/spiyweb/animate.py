"""The picture of one query, hop by hop: pure text, no terminal.

Everything here takes a `TraceRecord` and returns lines. Nothing sleeps,
nothing reads a key, nothing knows where the lines go - that is what makes
the frames testable and what lets the monitor redraw them however it likes.

A record carries each atom's hop of arrival and its FINAL energy, not an
energy per hop. So a frame at hop `k` shows the atoms that had arrived by
then, with bars that grow in during the hop they arrive (`t` in `0..1`) and
sit still afterwards; converging evidence comes from `TracePath.converging`,
duplicates from `TraceEvent(kind="suppressed")`. Honest to the data - the
picture never invents a number the propagation did not produce.

Two panels side by side when the terminal is wide enough - the ranking on
the left, the ring map on the right - and the ledger line under both. The
map places atoms with `rings.hop_ring_layout`, the same rule the scene
module uses, so one query produces one picture whatever draws it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.rings import hop_ring_layout, ring_radii
from spiyweb.terminal import bar, clip, pad, paint, paint_hop, printed_width

if TYPE_CHECKING:
    from spiyweb.config import WatchConfig
    from spiyweb.trace import TraceNode, TraceRecord

__all__ = [
    "Glyphs",
    "Style",
    "ease",
    "ledger_line",
    "query_block",
    "ranking",
    "ring_map",
]


@dataclass(frozen=True)
class Glyphs:
    """One glyph set, unicode or ASCII, so no drawing code branches on it."""

    seed: str
    on: str
    new: str
    dead: str
    dup: str
    ring: str
    arrow: str
    to: str
    times: str
    prompt: str
    dot: str
    caret: str
    tl: str
    tr: str
    bl: str
    br: str
    h: str
    v: str
    bullet: str
    tip: str
    spin: str
    live: str
    off: str
    pick: str
    unpick: str

    @classmethod
    def unicode(cls) -> Glyphs:
        return cls(
            seed="◉", on="●", new="○", dead="·", dup="⊘", ring="·",
            arrow="←", to="→", times="×", prompt="›", dot="·", caret="▏",  # noqa: RUF001
            tl="╭", tr="╮", bl="╰", br="╯", h="─", v="│",
            bullet="⏺", tip="※", spin="✻✽✶✳✢·", live="●", off="○",
            pick="●", unpick="○",
        )  # fmt: skip

    @classmethod
    def ascii(cls) -> Glyphs:
        return cls(
            seed="@", on="O", new="o", dead=".", dup="x", ring=".",
            arrow="<-", to="->", times="x", prompt=">", dot="-", caret="_",
            tl="+", tr="+", bl="+", br="+", h="-", v="|",
            bullet="*", tip="*", spin="-\\|/", live="*", off="o",
            pick="*", unpick="o",
        )  # fmt: skip


@dataclass(frozen=True)
class Style:
    """How to paint: colour on or off, which glyphs, the knobs."""

    color: bool
    glyphs: Glyphs
    config: WatchConfig

    def paint(self, text: str, *styles: str) -> str:
        return paint(text, *styles, enabled=self.color)

    def hop(self, text: str, hop: int) -> str:
        return paint_hop(text, hop, enabled=self.color)

    @property
    def blocks(self) -> str:
        return " -=#" if self.glyphs.times == "x" else " ▏▎▍▌▋▊▉█"

    @property
    def faint_blocks(self) -> str:
        return " -=#" if self.glyphs.times == "x" else " ░▒▓█"


def ease(t: float) -> float:
    """Ease-out cubic: a bar that shoots up and settles, not a ramp."""
    return 1.0 - (1.0 - max(0.0, min(1.0, t))) ** 3


def visible(record: TraceRecord, hop: int) -> list[TraceNode]:
    """Activated atoms that had arrived by `hop`, strongest first."""
    shown = [n for n in record.nodes if 0 <= n.hop <= hop and n.energy > 0]
    return sorted(shown, key=lambda n: (-n.energy, n.id))


def ghosts(record: TraceRecord, hop: int) -> list[TraceNode]:
    """Suppressed duplicates whose survivor is on screen by `hop`."""
    present = {n.id for n in visible(record, hop)}
    out = [
        n
        for n in record.nodes
        if n.suppressed_by and n.suppressed_by in present and n.energy <= 0
    ]
    return sorted(out, key=lambda n: n.id)


def _converging(record: TraceRecord) -> dict[str, int]:
    return {p.node: p.converging for p in record.paths if p.converging > 0}


def _label(text: str, chars: int) -> str:
    return text if len(text) <= chars else text[: chars - 1] + "…"


def ranking(record: TraceRecord, hop: int, t: float, style: Style) -> list[str]:
    """The bars: one row per visible atom, then the duplicates as ghosts."""
    g, cfg = style.glyphs, style.config
    shown = visible(record, hop)
    if not shown:
        return [style.paint("  nothing activated", "warn")]
    strongest = max(n.energy for n in shown)
    converging = _converging(record)
    rows: list[str] = []
    for node in shown[: cfg.max_rows]:
        fresh = node.hop == hop
        value = node.energy * ease(t) if fresh else node.energy
        weak = node.energy < record.threshold
        label = _label(node.id, cfg.label_chars).ljust(cfg.label_chars)
        if weak:
            meter = style.paint(
                bar(value, strongest, cfg.bar_width, style.faint_blocks).ljust(
                    cfg.bar_width
                ),
                "muted",
            )
            glyph = style.paint(g.dead, "muted")
        else:
            meter = style.hop(
                bar(value, strongest, cfg.bar_width, style.blocks).ljust(cfg.bar_width),
                node.hop,
            )
            glyph = style.hop(g.new if fresh and t < 0.5 else g.on, node.hop)
        cells = [
            f"  {glyph} {style.paint(label, 'bold' if fresh else 'muted')} {meter}",
            style.paint(f"{value:6.2f}", "muted" if weak else ""),
            style.paint(f"h{node.hop}", "muted"),
        ]
        if node.votes > 1 and (node.hop < hop or t > 0.35):
            cells[-1] += style.paint(f" {g.times}{node.votes}", "accent", "bold")
        notes: list[str] = []
        if node.id in converging and (node.hop < hop or t >= 0.5):
            notes.append(style.paint(f"{g.arrow} converging", "good"))
        if node.disputed:
            notes.append(style.paint("disputed", "warn"))
        if weak:
            notes.append(
                style.paint(f"below threshold {record.threshold:.2f}", "muted")
            )
        rows.append("  ".join(cells) + ("  " + " ".join(notes) if notes else ""))
    if len(shown) > cfg.max_rows:
        rows.append(style.paint(f"  +{len(shown) - cfg.max_rows} more", "muted"))
    for ghost in ghosts(record, hop)[:3]:
        label = _label(ghost.id, cfg.label_chars).ljust(cfg.label_chars)
        survivor = _label(ghost.suppressed_by, cfg.label_chars)
        rows.append(
            f"  {style.paint(g.dup, 'muted')} {style.paint(label, 'muted')} "
            + " " * cfg.bar_width
            + "  "
            + style.paint("  -.--", "muted")
            + "  "
            + style.paint(f"duplicate of {survivor} ", "muted")
            + style.paint(f"{g.to} vote +1", "accent")
        )
    return rows


def ring_map(
    record: TraceRecord, hop: int, t: float, style: Style, *, cols: int, rows: int
) -> list[str]:
    """Concentric hops on a character grid, seed mark at the centre."""
    g = style.glyphs
    nodes = [n for n in record.nodes if n.energy > 0 or n.suppressed_by]
    if not nodes or cols < 20 or rows < 5:
        return []
    layout = hop_ring_layout(
        [n.id for n in nodes],
        {n.id: n.hop for n in nodes},
        {n.id: n.energy for n in nodes},
        salt=record.query,
    )
    grid = [[" "] * cols for _ in range(rows)]
    color = [[""] * cols for _ in range(rows)]
    cx, cy = (cols - 1) / 2, (rows - 1) / 2
    # Labels sit beside their atoms, so the rings stop short of the edges by
    # one label's width on each side.
    span = max(10, cols - 1 - 2 * (style.config.label_chars + 2))

    def cell(x: float, y: float) -> tuple[int, int]:
        return round(cy + (y - 0.5) * (rows - 1)), round(cx + (x - 0.5) * span)

    def put(row: int, col: int, ch: str, sty: str) -> None:
        if 0 <= row < rows and 0 <= col < cols:
            grid[row][col], color[row][col] = ch, sty

    radii = ring_radii(n.hop for n in nodes)
    for level, radius in sorted(radii.items()):
        if level < 0 or level > hop:
            continue
        points = max(16, round(2 * math.pi * radius * span))
        for k in range(points):
            angle = 2 * math.pi * k / points
            row, col = cell(
                0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle)
            )
            put(row, col, g.ring, "dim")
    present = {n.id for n in visible(record, hop)}
    labels: list[tuple[int, int, str, str]] = []
    for node in nodes:
        if node.id not in present:
            if not (node.suppressed_by and node.suppressed_by in present):
                continue
            glyph, sty = g.dup, "muted"
        else:
            fresh = node.hop == hop
            if fresh and t < 0.15:
                continue
            glyph = g.new if fresh and t < 0.5 else g.on
            sty = ""
        row, col = cell(*layout[node.id])
        painted = style.hop(glyph, node.hop) if not sty else style.paint(glyph, sty)
        put(row, col, painted, "raw")
        labels.append((row, col, _label(node.id, style.config.label_chars), sty))
    centre = cell(0.5, 0.5)
    if grid[centre[0]][centre[1]] in (" ", g.ring):
        put(centre[0], centre[1], style.paint(g.seed, "accent", "bold"), "raw")
    for row, col, text, sty in labels:
        right = col >= cx
        text = " " + text if right else text + " "
        start = col + 1 if right else col - len(text)
        for i, ch in enumerate(text):
            c = start + i
            if 0 <= c < cols and grid[row][c] in (" ", g.ring):
                grid[row][c], color[row][c] = ch, sty or "hop"
    lines: list[str] = []
    for r in range(rows):
        out = ""
        for c in range(cols):
            ch, sty = grid[r][c], color[r][c]
            if sty == "raw" or sty == "":
                out += ch
            elif sty == "hop":
                out += style.paint(ch, "accent")
            else:
                out += style.paint(ch, sty)
        lines.append(out)
    return lines


def ledger_line(
    record: TraceRecord, hop: int, t: float, style: Style, *, final: bool
) -> str:
    g = style.glyphs
    parts = [
        style.paint(f"hop {hop}/{record.hops_used}", "bold"),
        style.paint("injected", "muted") + f" {record.injected_energy:.2f}",
    ]
    if final and record.ledger is not None:
        book = record.ledger
        parts.append(style.paint("held", "muted") + f" {book.held:.2f}")
        parts.append(style.paint("dissipated", "muted") + f" {book.dissipated:.2f}")
        if book.destroyed > 0:
            parts.append(style.paint("destroyed", "muted") + f" {book.destroyed:.2f}")
    elif final:
        parts.append(style.paint("total", "muted") + f" {record.total_energy:.2f}")
    if final:
        stopped = f"stopped: {record.stop_reason}"
        parts.append(
            style.paint(
                stopped, "good" if record.stop_reason == "threshold" else "warn"
            )
        )
    else:
        parts.append(style.paint("spreading" + g.dot * (1 + int(t * 3)), "accent"))
    sep = "  " + style.paint(g.dot, "muted") + "  "
    return "  " + sep.join(parts)


def query_block(
    record: TraceRecord,
    hop: int,
    t: float,
    style: Style,
    *,
    width: int,
    with_map: bool,
    final: bool,
    map_rows: int | None = None,
) -> list[str]:
    """Head line, the two panels, the ledger line. `map_rows` overrides the
    configured map height when the caller has less room."""
    g, cfg = style.glyphs, style.config
    head = (
        style.paint(g.bullet, "accent")
        + " "
        + style.paint("query", "bold")
        + "  "
        + style.paint('"' + record.query + '"', "muted")
        + "  "
        + style.paint(
            f"{record.profile or 'explore'} {g.dot} {record.node_count} atoms"
            + (f" {g.dot} {record.elapsed_ms:.0f} ms" if record.elapsed_ms else ""),
            "dim",
        )
    )
    left = ranking(record, hop, t, style)
    map_cols = width - cfg.ranking_width - 2
    rows = cfg.map_rows if map_rows is None else map_rows
    if with_map and cfg.map_enabled and map_cols >= 24 and rows >= 5:
        cols = min(map_cols, round(rows * 2.6) + 12)
        right = ring_map(record, hop, t, style, cols=cols, rows=rows)
    else:
        right = []
    if right:
        height = max(len(left), len(right))
        top = (height - len(left)) // 2
        left = [""] * top + left + [""] * (height - len(left) - top)
        right = right + [""] * (height - len(right))
        body = [pad(a, cfg.ranking_width) + b for a, b in zip(left, right, strict=True)]
    else:
        body = left
    lines = [head, "", *body, "", ledger_line(record, hop, t, style, final=final)]
    return [clip(line, width) for line in lines]


def block_height(record: TraceRecord, style: Style, *, with_map: bool) -> int:
    """How many rows `query_block` needs, so a caller can decide the layout
    before drawing (the map is dropped when it does not fit)."""
    cfg = style.config
    rows = min(len(visible(record, record.hops_used)), cfg.max_rows)
    rows += min(len(ghosts(record, record.hops_used)), 3)
    if len(visible(record, record.hops_used)) > cfg.max_rows:
        rows += 1
    body = max(rows, cfg.map_rows) if with_map and cfg.map_enabled else rows
    return 4 + max(1, body)


def width_of(lines: list[str]) -> int:
    return max((printed_width(line) for line in lines), default=0)
