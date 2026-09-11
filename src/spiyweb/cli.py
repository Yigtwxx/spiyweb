"""`spiyweb` on the command line: index a corpus, ask it, look at what it did.

Six verbs, and every one of them wraps something the library already does
rather than adding a mechanism of its own. That constraint is the point: a
CLI that grows behaviour the Python API does not have becomes a second
product to keep correct, and this project has a measurement campaign's worth
of evidence about what happens when one mechanism has two implementations.

    spiyweb                               the live monitor (same as `watch`)
    spiyweb watch                         watch this folder's app query, live
    spiyweb version                       what is installed, and what is not
    spiyweb index docs/ my-index          a directory of text files -> an index
    spiyweb query my-index "a question"   the activated web, as text
    spiyweb lint my-index                 what is wrong with the CORPUS
    spiyweb menu                          the old numbered menu

Importing this module costs nothing. `spiyweb version` has to work on a bare
`pip install spiyweb` - that is precisely when somebody needs to be told which
extra is missing - so every heavy import happens inside the subcommand that
needs it, and a missing one is reported as the `pip install` line that fixes
it rather than as a traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from spiyweb import __version__
from spiyweb.config import CorpusLintConfig as CorpusLintDefaults
from spiyweb.profiles import DEFAULT_PROFILE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spiyweb.indexing import DocumentInput
    from spiyweb.session import SpiywebIndex

__all__ = ["main"]

TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".rst")
"""What `spiyweb index` reads when no `--glob` is given. Plain text only:
extracting text from PDF or HTML is a different job with its own failure
modes, and doing it badly inside a retrieval library helps nobody."""

LINT_KINDS = (
    "empty_layer",
    "orphan",
    "isolated",
    "hub",
    "duplicate",
    "duplicate_source",
    "contradiction",
)
"""Order the lint report prints its sections in - configuration first, then
structure, then content. An empty layer explains islands that would otherwise
look like a corpus problem, and an island makes every finding inside it
moot, so those two lead."""

VOTE_MARK = "×"  # noqa: RUF001 - the sign, not the letter
"""Multiplication sign for a vote count. A plain `x3` reads as part of an
identifier; the sign reads as a quantity. Kept as an escape so the source
carries no character a reader could mistake for a Latin letter."""

_SEVERE = frozenset({"empty_layer", "orphan", "contradiction"})
"""Findings that mean something is STRUCTURALLY wrong rather than merely
dense. An empty layer is a weight doing nothing, an orphan is unreachable
knowledge, a contradiction is the corpus arguing with itself - each changes
what retrieval can do at all, so each gets the loud colour."""

LINT_WORST = 10
"""Rows of the "most affected" roll-up. A corpus owner acts on a handful."""

DEFAULT_TOP = 10
"""Passages `query` prints. NOT a `top-k`: the web already stopped itself,
and this only decides how much of its answer fits on a terminal screen."""

EXTRAS = {
    "store": ("numpy", "faiss"),
    "embed": ("sentence_transformers",),
    "entity": ("spacy",),
    "view": ("numpy",),
    "nli": ("torch", "transformers"),
}
"""Extra -> the imports that prove it is installed. Used by `version` only,
which is why a missing one is reported rather than raised."""


class Problem(SystemExit):
    """A failure the user can act on; the message IS the explanation."""

    def __init__(self, message: str) -> None:
        super().__init__(f"spiyweb: {message}")


# --- version ---------------------------------------------------------------


def _installed(modules: Sequence[str]) -> bool:
    from importlib.util import find_spec

    try:
        return all(find_spec(module) is not None for module in modules)
    except (ImportError, ValueError):
        return False


def _version(args: argparse.Namespace) -> int:
    """Say what is installed, because "it does not work" usually means an extra."""
    rows = {name: _installed(modules) for name, modules in sorted(EXTRAS.items())}
    if args.json:
        import json

        print(json.dumps({"version": __version__, "extras": rows}, indent=2))
        return 0
    from spiyweb.terminal import paint, rule, supports_color, terminal_width

    color = supports_color()
    print(rule(f"spiyweb {__version__}", terminal_width(), enabled=color))
    for name, present in rows.items():
        mark = paint(
            "+" if present else "-", "good" if present else "bad", enabled=color
        )
        label = paint(f"[{name}]", "accent" if present else "muted", enabled=color)
        state = (
            paint("installed", "good", enabled=color)
            if present
            else paint("not installed", "muted", enabled=color)
        )
        print(f"  {mark} {label} {'.' * (12 - len(name))} {state}")
    missing = [name for name, present in rows.items() if not present]
    if missing:
        command = f'pip install "spiyweb[{",".join(missing)}]"'
        print()
        print(
            paint("install what you need: ", "muted", enabled=color)
            + paint(command, "accent", "bold", enabled=color)
        )
    return 0


# --- index -----------------------------------------------------------------


def _read_documents(
    root: Path, pattern: str | None, whole: bool
) -> list[DocumentInput]:
    """A directory of text files, as documents the pipeline understands.

    One file is one document, and a blank line separates units - that is the
    plainest possible reading of a plain-text corpus. `--whole-file` turns a
    file into a single unit for corpora whose paragraphs are not passages.
    """
    from spiyweb.indexing import DocumentInput, TextUnit

    if not root.is_dir():
        raise Problem(f"{root} is not a directory")
    # The suffix test is on the lowercased name rather than an `rglob` per
    # suffix: `rglob("*.txt")` matches `README.TXT` on Windows and skips it
    # on macOS and Linux, and a corpus must not depend on where it was
    # indexed. The sort key is the posix relative path - the string that
    # becomes the source id - so document order, and with it every
    # positional artifact, is identical on all three platforms.
    found = (
        root.rglob(pattern)
        if pattern
        else (p for p in root.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES)
    )
    paths = sorted(
        (path for path in found if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    documents: list[DocumentInput] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        units = (
            [text] if whole else [part for part in text.split("\n\n") if part.strip()]
        )
        if not units:
            continue
        documents.append(
            DocumentInput(
                source_id=path.relative_to(root).as_posix(),
                units=tuple(TextUnit(text=unit.strip()) for unit in units),
            )
        )
    if not documents:
        looked = (
            f"--glob {pattern}" if pattern else f"suffixes {', '.join(TEXT_SUFFIXES)}"
        )
        raise Problem(f"no readable text under {root} (looked for {looked})")
    return documents


def _index(args: argparse.Namespace) -> int:
    try:
        from spiyweb.embedding import SentenceTransformerEmbedder
        from spiyweb.entities import load_spacy_pipeline
        from spiyweb.indexing import build_index
    except ImportError as missing:
        raise Problem(
            f"{missing}\nindexing needs the index-time extras: "
            'pip install "spiyweb[index]"'
        ) from missing

    documents = _read_documents(args.docs, args.glob, args.whole_file)
    units = sum(len(document.units) for document in documents)
    print(f"{len(documents)} document(s), {units} unit(s) -> {args.out}")

    embedder = SentenceTransformerEmbedder()
    try:
        pipeline = load_spacy_pipeline()
    except OSError as absent:
        raise Problem(str(absent)) from absent
    manifest = build_index(
        documents,
        args.out,
        embedder=embedder,
        entity_pipeline=pipeline,
        embedding_model=getattr(embedder, "model_name", None),
        force=args.force,
    )
    print(
        f"done: {manifest.chunks} chunk(s), "
        f"{sum(manifest.edges.values())} edge(s) across "
        f"{len([n for n, c in manifest.edges.items() if c])} layer(s)"
    )
    return 0


# --- query -----------------------------------------------------------------


def _query(args: argparse.Namespace) -> int:
    from spiyweb.config import TraceConfig

    index = _open(args.index, trace=TraceConfig(enabled=False))
    profile = args.profile or DEFAULT_PROFILE
    answer = index.retrieve(args.question, profile=profile)
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "query": answer.query,
                    "confidence": {
                        "total_energy": answer.confidence.total_energy,
                        "node_count": answer.confidence.node_count,
                        "hop_depth": answer.confidence.hop_depth,
                    },
                    "stop_reason": answer.result.propagation.stop_reason,
                    "passages": [
                        {
                            "node_id": passage.node_id,
                            "source_id": passage.source_id,
                            "energy": passage.energy,
                            "hop": passage.hop,
                            "votes": passage.votes,
                            "text": passage.text,
                        }
                        for passage in answer.passages[: args.top]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    _render_answer(answer, args.top, profile)
    return 0


def _render_answer(answer: object, top: int, profile: str) -> None:
    """Draw the activated web: energy as length, hop as colour.

    A column of numbers makes a reader divide in their head to see the decay.
    A bar does not, and the decay IS the mechanism - so the bar is the report
    and the number rides alongside it.
    """
    from spiyweb.terminal import (
        bar,
        paint,
        paint_hop,
        rule,
        supports_color,
        terminal_width,
    )

    color = supports_color()
    width = terminal_width()
    passages = answer.passages[:top]  # type: ignore[attr-defined]
    confidence = answer.confidence  # type: ignore[attr-defined]
    stop = answer.result.propagation.stop_reason  # type: ignore[attr-defined]

    print(rule(answer.query[: width - 12], width, enabled=color))  # type: ignore[attr-defined]
    stopped = paint(stop, "good" if stop == "threshold" else "warn", enabled=color)
    print(
        paint(
            f"{confidence.node_count} atoms · {confidence.total_energy:.2f} energy · "
            f"depth {confidence.hop_depth} · profile {profile} · stopped on ",
            "muted",
            enabled=color,
        )
        + stopped
    )
    if not passages:
        print(paint("nothing activated", "warn", enabled=color))
        return

    strongest = max(passage.energy for passage in passages)
    bar_width = max(8, min(24, width // 4))
    print()
    for passage in passages:
        drawn = bar(passage.energy, strongest, bar_width)
        meter = paint_hop(drawn.ljust(bar_width), passage.hop, enabled=color)
        votes = (
            paint(f" {VOTE_MARK}{passage.votes}", "accent", enabled=color)
            if passage.votes > 1
            else ""
        )
        head = (
            f"{meter} {passage.energy:6.2f}  "
            + paint(passage.node_id, "muted", enabled=color)
            + paint(f"  h{passage.hop}", "muted", enabled=color)
            + votes
        )
        print(head)
        text = passage.text or "(this index carries no text)"
        for line in _wrap(text, width - bar_width - 3):
            print(" " * (bar_width + 1) + paint(line, "dim", enabled=color))
        print()

    hidden = len(answer.passages) - len(passages)  # type: ignore[attr-defined]
    if hidden > 0:
        print(
            paint(
                f"{hidden} more below the fold - the web did not stop here, --top did",
                "muted",
                enabled=color,
            )
        )


def _wrap(text: str, width: int) -> list[str]:
    """Wrap on words, because a passage broken mid-word is unreadable."""
    import textwrap

    return textwrap.wrap(" ".join(text.split()), max(width, 20))[:4]


# --- lint ------------------------------------------------------------------


def _lint(args: argparse.Namespace) -> int:
    """Inspect the corpus's shape - no query involved, and none needed."""
    from spiyweb.config import CorpusLintConfig
    from spiyweb.lint import source_summary

    try:
        from spiyweb.indexing import lint_index
    except ImportError as missing:
        raise Problem(
            f'{missing}\nreading an index needs: pip install "spiyweb[store]"'
        ) from missing

    target = Path(args.index)
    if not _is_index(target):
        raise Problem(f"{target} does not look like an index (no nodes.json in it)")
    _progress(f"linting {target} - this reads the whole graph")
    report = lint_index(
        target,
        config=CorpusLintConfig(
            duplicate_weight=args.duplicate_weight,
            hub_share_floor=args.hub_share_floor,
            max_per_kind=args.top,
        ),
    )
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "nodes": report.nodes,
                    "edges": report.edges,
                    "components": report.components,
                    "largest_component": report.largest_component,
                    "isolated": report.isolated,
                    "counts": report.counts,
                    "findings": [
                        {
                            "kind": finding.kind,
                            "subject": finding.subject,
                            "value": finding.value,
                            "nodes": list(finding.nodes),
                            "message": finding.message,
                        }
                        for finding in report.findings
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    from spiyweb.terminal import bar, paint, rule, supports_color, terminal_width

    color = supports_color()
    width = terminal_width()
    print(rule(f"corpus lint - {target}", width, enabled=color))
    headline, summary = report.text.split("\n", 1)
    print(paint(headline, "muted", enabled=color))
    print(paint(summary, "warn" if report.findings else "good", enabled=color))

    for kind in LINT_KINDS:
        found = report.by_kind(kind)
        if not found:
            continue
        tone = "bad" if kind in _SEVERE else "warn"
        print()
        print(
            paint(kind.upper().replace("_", " "), tone, "bold", enabled=color)
            + paint(f"  ({len(found)} shown)", "muted", enabled=color)
        )
        for finding in found:
            print(paint("  - ", tone, enabled=color) + finding.message)

    worst = source_summary(report)
    if worst:
        rows = list(worst.items())[:LINT_WORST]
        top = max(count for _, count in rows)
        print()
        print(paint("MOST AFFECTED", "heading", "bold", enabled=color))
        for subject, count in rows:
            meter = bar(count, top, 12).ljust(12)
            print(
                f"  {count:>3}  " + paint(meter, "warn", enabled=color) + f"  {subject}"
            )
    return 0


def _progress(message: str) -> None:
    """A line about the work, on stderr.

    Progress is diagnostics, not data. `spiyweb lint --json | jq` has to
    receive JSON and nothing else, and "linting ... - this reads the whole
    graph" on stdout makes the first character of the stream a `l`. Found by
    a test that tried to parse the output it had just asked for.
    """
    print(message, file=sys.stderr, flush=True)


def _is_index(path: Path) -> bool:
    return path.is_dir() and (path / "nodes.json").is_file()


def _open(path: Path | str, **options: object) -> SpiywebIndex:
    from spiyweb import open_index

    target = Path(path)
    if not _is_index(target):
        raise Problem(
            f"{target} does not look like an index (no nodes.json in it); "
            "build one with `spiyweb index <docs> <out>`"
        )
    try:
        return open_index(target, **options)  # type: ignore[arg-type]
    except ImportError as missing:
        raise Problem(
            f'{missing}\nopening an index needs: pip install "spiyweb[index]"'
        ) from missing


# --- wiring ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spiyweb",
        description=(
            "Graph-based retrieval: the query is injected as an energy seed "
            "and spreads outward with decay."
        ),
    )
    parser.add_argument("--version", action="version", version=f"spiyweb {__version__}")
    # Not `required=True`: a bare `spiyweb` opens the guided menu instead of
    # printing a usage error at someone who has not learned the verbs yet.
    # `main` handles the empty case, and refuses to prompt when nobody is
    # there to answer.
    subs = parser.add_subparsers(dest="command")

    version = subs.add_parser("version", help="version and which extras are installed")
    version.add_argument("--json", action="store_true", help="machine-readable")
    version.set_defaults(handler=_version)

    index = subs.add_parser("index", help="build an index from a directory of text")
    index.add_argument("docs", type=Path, help="directory of .txt/.md files")
    index.add_argument("out", type=Path, help="where to write the index")
    index.add_argument("--glob", help=f"pattern to read instead of {TEXT_SUFFIXES}")
    index.add_argument(
        "--whole-file",
        action="store_true",
        help="one unit per file instead of one per blank-line-separated block",
    )
    index.add_argument(
        "--force", action="store_true", help="rebuild artifacts that already exist"
    )
    index.set_defaults(handler=_index)

    query = subs.add_parser("query", help="ask an index and print what lit up")
    query.add_argument("index", type=Path)
    query.add_argument("question")
    query.add_argument(
        "--profile",
        choices=("precise", "explore", "compare"),
        help=(
            f"damping, threshold and seed width as one package "
            f"(default {DEFAULT_PROFILE}, the same as the library's)"
        ),
    )
    query.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=(
            f"passages to print (default {DEFAULT_TOP}); the web already stopped itself"
        ),
    )
    query.add_argument("--json", action="store_true", help="machine-readable")
    query.set_defaults(handler=_query)

    lint = subs.add_parser("lint", help="inspect the corpus's shape, without a query")
    lint.add_argument("index", type=Path)
    lint.add_argument(
        "--duplicate-weight",
        type=float,
        default=CorpusLintDefaults.duplicate_weight,
        help="raw semantic cosine at which two atoms count as near-identical",
    )
    lint.add_argument(
        "--hub-share-floor",
        type=float,
        default=CorpusLintDefaults.hub_share_floor,
        help="report a hub whose strongest neighbour gets no more than this share",
    )
    lint.add_argument(
        "--top",
        type=int,
        default=CorpusLintDefaults.max_per_kind,
        help="findings kept per kind, worst first",
    )
    lint.add_argument("--json", action="store_true", help="machine-readable")
    lint.set_defaults(handler=_lint)

    watch = subs.add_parser(
        "watch",
        help="the live monitor: every query your app runs, played as it happens",
    )
    watch.add_argument(
        "--dir",
        default=None,
        help="folder holding the marker and traces.jsonl (default .spiyweb)",
    )
    watch.set_defaults(handler=_watch)

    menu = subs.add_parser("menu", help="the guided menu: questions, no monitor")
    menu.set_defaults(handler=_menu)

    return parser


def _watch(args: argparse.Namespace) -> int:
    from spiyweb.config import WatchConfig
    from spiyweb.watch import interactive

    config = WatchConfig() if args.dir is None else WatchConfig(attach_dir=args.dir)
    return interactive(config)


def _menu(args: argparse.Namespace) -> int:
    from spiyweb.wizard import interactive

    return interactive()


def _writable_stdout() -> None:
    """Make stdout survive a corpus this console cannot spell.

    Python on Windows encodes stdout with the console codepage - cp1254 on a
    Turkish machine - and this project indexes Turkish and English text by
    design. A passage with a macron in it then raises `UnicodeEncodeError`
    out of `print`, and a diagnostic tool that crashes while diagnosing is
    worse than one that prints a question mark.

    `errors="replace"` and not a silent swallow: the character is visibly
    lost, so nobody mistakes the output for the corpus.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the `spiyweb` command.

    With no arguments this opens the live monitor - nothing is asked - but
    only in a terminal. A pipe or a CI job gets the usage text and a
    non-zero exit, because a prompt with nobody at it is a hung build rather
    than a friendly one; a terminal that cannot read single keys or redraw
    a screen gets the numbered menu instead.
    """
    _writable_stdout()
    args = build_parser().parse_args(argv)
    try:
        if getattr(args, "command", None) is None:
            from spiyweb.watch import interactive

            return interactive()
        return int(args.handler(args))
    except Problem:
        raise
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
