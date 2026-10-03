"""Record the queries `/demo` plays - with the real pipeline, on a sample corpus.

    python examples/make_demo.py

Needs the index extras and the spaCy model (`/install index` in the
monitor, or `pip install "spiyweb[index]"`). Builds a thirty-four-passage index
in a temporary folder with the same `spiyweb index` a user runs, asks it
three questions with the default profile, and writes the three trace
records to `src/spiyweb/demo.jsonl`, which ships in the wheel.

`/demo` replays those records with no dependency at all, so someone who
has just installed spiyweb can watch a web spread before downloading a
model. The records are real output - this script is how they were made,
and rerunning it after a change to the pipeline is how they stay honest.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from spiyweb import open_index
from spiyweb.cli import main as spiyweb
from spiyweb.config import TraceConfig

OUT = Path(__file__).resolve().parents[1] / "src" / "spiyweb" / "demo.jsonl"

CORPUS = {
    "tesla.md": (
        "Nikola Tesla built Wardenclyffe Tower in Shoreham to send electric "
        "power without wires.",
        "Tesla hoped the tower would also carry telephone messages across "
        "the Atlantic.",
    ),
    "morgan.md": (
        "The banker J. P. Morgan agreed to finance the Wardenclyffe project in 1901.",
        "Morgan stopped funding Tesla after Guglielmo Marconi sent a radio "
        "signal across the Atlantic.",
    ),
    "marconi.md": (
        "Guglielmo Marconi sent the first transatlantic radio signal in December 1901.",
        "Marconi's transmitter in Cornwall cost far less than Tesla's tower.",
    ),
    "shoreham.md": (
        "Shoreham is a small village on the north shore of Long Island, New York.",
        "The Tesla Science Center now preserves the Wardenclyffe site in Shoreham.",
    ),
    "newspaper.md": (
        # The first passage repeats tesla.md word for word: a second source
        # saying the same thing is what a vote is.
        "Nikola Tesla built Wardenclyffe Tower in Shoreham to send electric "
        "power without wires.",
        "The tower was demolished in 1917 and its scrap paid part of Tesla's "
        "hotel debts.",
    ),
    "waldorf.md": (
        "Tesla lived at the Waldorf-Astoria hotel and left large unpaid bills there.",
        "The hotel took the Wardenclyffe property as security for Tesla's debt.",
    ),
    "edison.md": (
        "Thomas Edison championed direct current for city power grids.",
        "Edison and Tesla quarrelled over alternating current in the 1880s.",
    ),
    # Unrelated passages: a web that lights up everything proves nothing, so
    # most of the corpus has to be something the question should NOT reach.
    "curie.md": (
        "Marie Curie discovered polonium and radium with Pierre Curie.",
        "Curie was the first person to win Nobel Prizes in two sciences.",
    ),
    "danube.md": (
        "The Danube flows through ten countries before reaching the Black Sea.",
        "Vienna, Budapest and Belgrade all stand on the Danube.",
    ),
    "everest.md": (
        "Mount Everest rises 8,849 metres above sea level.",
        "Tenzing Norgay and Edmund Hillary first reached the summit in 1953.",
    ),
    "coffee.md": (
        "Coffee plants first grew wild in the highlands of Ethiopia.",
        "Coffee houses spread through Istanbul in the sixteenth century.",
    ),
    "bees.md": (
        "Honey bees tell each other where flowers are with a waggle dance.",
        "A single hive can hold tens of thousands of worker bees.",
    ),
    "printing.md": (
        "Johannes Gutenberg built a printing press with movable type around 1440.",
        "The Gutenberg Bible was among the first books printed in Europe.",
    ),
    "jazz.md": (
        "Jazz grew out of blues and ragtime in New Orleans.",
        "Louis Armstrong made the trumpet solo central to jazz.",
    ),
    "volcano.md": (
        "Mount Vesuvius buried Pompeii under ash in the year 79.",
        "Pompeii was rediscovered by excavators in the eighteenth century.",
    ),
    "chess.md": (
        "Chess reached Europe through Persia and the Arab world.",
        "Garry Kasparov lost a match to the computer Deep Blue in 1997.",
    ),
}

QUESTIONS = (
    "who built a tower on Long Island",
    "why did the money for Tesla's tower run out",
    "what happened to the tower in the end",
)


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        docs = Path(scratch) / "docs"
        docs.mkdir()
        for name, passages in CORPUS.items():
            (docs / name).write_text("\n\n".join(passages) + "\n", encoding="utf-8")
        index_dir = Path(scratch) / "demo-index"
        code = spiyweb(["index", str(docs), str(index_dir)])
        if code:
            return code
        index = open_index(index_dir, trace=TraceConfig(attach_dir=None))
        lines = []
        for sequence, question in enumerate(QUESTIONS):
            answer = index.retrieve(question)
            if answer.trace is None:
                print("tracing is off; nothing to record", file=sys.stderr)
                return 1
            # The scratch path means nothing on anyone else's machine.
            record = replace(answer.trace, index="demo", sequence=sequence)
            lines.append(json.dumps(record.to_dict(), ensure_ascii=False))
            top = ", ".join(p.node_id for p in answer.passages[:4])
            print(f"{question!r}: {answer.trace.hops_used} hop(s) -> {top}")
        index.close()
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} record(s) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
