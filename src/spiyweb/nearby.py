"""What the monitor knows about the folders around it - stdlib only.

An index is a folder with a `nodes.json`; its receipt is `meta.json`. Both
are plain JSON, so everything here reads them with `json` and nothing else:
the monitor must be able to list, stamp and describe indexes - and find the
folders worth indexing, and complete a typed path - without the numpy that
opening an index costs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "IndexInfo",
    "TextFolder",
    "complete_path",
    "describe",
    "find_indexes",
    "index_stamp",
    "is_index",
    "read_meta",
    "split_last_token",
    "text_folders",
]

META_FILENAME = "meta.json"
NODES_FILENAME = "nodes.json"

SEARCH_ROOTS = (".", "data", "indexes")
"""Where indexes are looked for, one level down, nearest first. Not a
recursive walk: an index is a folder you made, and it sits near you."""

MAX_LISTED = 12
"""Enough to choose from, few enough to read."""

PRUNED_DIRS = frozenset(
    {"venv", ".venv", "node_modules", "__pycache__", "site-packages", "build", "dist"}
)
"""Never walked into: environments and build output hold no corpus."""


def is_index(path: Path | str) -> bool:
    """The test every verb uses: a folder with a `nodes.json` in it."""
    target = Path(path)
    return target.is_dir() and (target / NODES_FILENAME).is_file()


def read_meta(index: Path | str) -> dict[str, object]:
    """The index's receipt, or `{}` when it has none or it cannot be read -
    the sealed Phase 1 indexes predate several of its fields."""
    try:
        raw = json.loads((Path(index) / META_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def index_stamp(index: Path | str) -> int:
    """Changes whenever the index is rebuilt: `build_index` rewrites the
    receipt last, so its modification time names the build."""
    root = Path(index)
    for name in (META_FILENAME, NODES_FILENAME):
        try:
            return (root / name).stat().st_mtime_ns
        except OSError:
            continue
    return 0


@dataclass(frozen=True)
class IndexInfo:
    """One index, as far as its receipt says - never by opening it."""

    path: Path
    nodes: int | None
    propositions: int | None
    layers: tuple[str, ...]
    """Edge layers that hold at least one edge."""
    model: str | None
    built_ns: int


def describe(index: Path | str) -> IndexInfo:
    meta = read_meta(index)
    edges = meta.get("edges")
    layers = (
        tuple(
            name
            for name, count in edges.items()
            if isinstance(count, int) and count > 0
        )
        if isinstance(edges, dict)
        else ()
    )
    nodes = meta.get("nodes", meta.get("corpus_chunks"))
    propositions = meta.get("propositions")
    model = meta.get("embedding_model")
    return IndexInfo(
        path=Path(index),
        nodes=nodes if isinstance(nodes, int) else None,
        propositions=propositions if isinstance(propositions, int) else None,
        layers=layers,
        model=model if isinstance(model, str) else None,
        built_ns=index_stamp(index),
    )


def find_indexes(
    cwd: Path, first: Iterable[Path | str] = (), *, limit: int = MAX_LISTED
) -> list[IndexInfo]:
    """The indexes worth offering: `first` (the active one, what `/find`
    saw) in that order, then one level under each search root."""
    found: list[IndexInfo] = []
    seen: set[Path] = set()

    def offer(candidate: Path) -> None:
        try:
            key = candidate.resolve()
        except OSError:
            return
        if key not in seen and is_index(candidate):
            seen.add(key)
            found.append(describe(candidate))

    for path in first:
        target = Path(path)
        offer(target if target.is_absolute() else cwd / target)
    for root in SEARCH_ROOTS:
        base = cwd / root
        if not base.is_dir():
            continue
        try:
            entries = sorted(base.iterdir(), key=lambda p: p.name.casefold())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                offer(entry)
    return found[:limit]


@dataclass(frozen=True)
class TextFolder:
    """A folder with text a corpus can be built from, counted recursively -
    the same way `spiyweb index` reads it."""

    path: Path
    files: int


def text_folders(
    cwd: Path,
    suffixes: Sequence[str],
    *,
    max_depth: int,
    max_files: int,
    limit: int = MAX_LISTED,
) -> list[TextFolder]:
    """Folders one or two levels down holding text files, plus `.` itself
    when it holds some directly. Environments, hidden folders and indexes
    are never entered; the walk stops after `max_files` files."""
    counts: dict[Path, int] = {}
    seen = 0
    root = cwd.resolve()
    for current, dirs, files in os.walk(root):
        here = Path(current)
        relative = here.relative_to(root).parts
        dirs[:] = sorted(
            d
            for d in dirs
            if not d.startswith(".")
            and d not in PRUNED_DIRS
            and not (here / d / NODES_FILENAME).is_file()
        )
        if len(relative) >= max_depth:
            dirs[:] = []
        for name in files:
            if Path(name).suffix.lower() not in suffixes:
                continue
            seen += 1
            # A file counts for the folder one level down and the one two
            # levels down that hold it; a file in `.` itself counts for `.`.
            owners = (
                [
                    root / Path(*relative[:depth])
                    for depth in (1, 2)
                    if len(relative) >= depth
                ]
                if relative
                else [root]
            )
            for owner in owners:
                counts[owner] = counts.get(owner, 0) + 1
        if seen >= max_files:
            break
    ordered = sorted(counts.items(), key=lambda item: item[0].as_posix().casefold())
    return [TextFolder(path=path, files=n) for path, n in ordered][:limit]


def split_last_token(text: str) -> tuple[str, str]:
    """`/index my "some do` -> (`/index my `, `"some do`): what a tab would
    complete, with a quoted path counted as one token."""
    start, quoted = 0, False
    for position, char in enumerate(text):
        if char == '"':
            quoted = not quoted
        elif char == " " and not quoted:
            start = position + 1
    return text[:start], text[start:]


def complete_path(token: str, cwd: Path, *, dirs_only: bool = True) -> list[str]:
    """The paths `token` could become, written the way they would be typed:
    `/` separators, a trailing `/` on a folder, quoted when they hold a
    space. Hidden entries only when the typed name starts with a dot."""
    typed = token.strip('"').replace("\\", "/")
    # The folder part keeps its trailing separator: `C:/` and `/` are roots,
    # and cutting the slash off would turn them into `C:` and nothing.
    cut = typed.rfind("/") + 1
    folder, prefix = typed[:cut], typed[cut:]
    base = Path(folder) if folder else Path()
    if not base.is_absolute():
        base = cwd / base
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name.casefold())
    except OSError:
        return []
    out: list[str] = []
    for entry in entries:
        name = entry.name
        if name.startswith(".") and not prefix.startswith("."):
            continue
        if not name.casefold().startswith(prefix.casefold()):
            continue
        is_dir = entry.is_dir()
        if dirs_only and not is_dir:
            continue
        written = folder + name + ("/" if is_dir else "")
        out.append(f'"{written}"' if " " in written else written)
    return out
