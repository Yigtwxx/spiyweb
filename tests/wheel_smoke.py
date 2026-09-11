"""Wheel smoke: what someone who typed `pip install spiyweb` actually gets.

Deliberately NOT a `test_*.py`. This runs under an interpreter that has
neither pytest nor numpy installed, so it must be a plain script with plain
asserts; pytest never collects it, because `python_files` defaults to
`test_*.py`.

The `check` job cannot do this job. It syncs `store`, `entity` and `web`
because FAISS and spaCy are units under test, which puts numpy and faiss on
the path and makes `import spiyweb` succeed there whatever the wheel holds.
This script is the only place the packaging claims are measured on the
ARTIFACT rather than on the checkout.
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import spiyweb
import spiyweb.indexing

# 1. The typing marker really shipped.
assert (Path(spiyweb.__file__).parent / "py.typed").is_file(), "py.typed missing"

# 2. One version source, proved on a freshly built artifact rather than on an
#    editable install whose metadata can go stale.
assert version("spiyweb") == spiyweb.__version__, (
    f"metadata {version('spiyweb')} != __version__ {spiyweb.__version__}"
)

# 3. Zero dependencies - the claim `dependencies = []` makes, checked on the
#    installed package instead of on the import graph.
assert "numpy" not in sys.modules, "numpy reached the import graph"
assert "faiss" not in sys.modules, "faiss reached the import graph"

# 4. Every declared name resolves.
for name in spiyweb.__all__:
    getattr(spiyweb, name)

# 5. The index-time facade opens with nothing installed, and the two
#    FAISS-bound names ask for the extra BY NAME instead of failing obscurely.
try:
    store_class = spiyweb.indexing.VectorStore
except ImportError as error:
    assert "spiyweb[store]" in str(error), str(error)
else:
    raise SystemExit(f"VectorStore resolved without faiss: {store_class!r}")

# 6. The canonical trace of CLAUDE.md section 2.6 - the one result this
#    project is actually about. `D` is never the most similar node to the
#    query, yet two weak converging paths promote it to third place. A wheel
#    where this moves is a wheel shipping a different product.
graph = spiyweb.Graph.from_edges(
    [
        ("A", "A_dup", 0.0),
        ("A", "B", 0.8),
        ("A", "D", 0.4),
        ("C", "D", 0.6),
        ("C", "E", 0.3),
        ("D", "F", 0.5),
    ]
)
result = spiyweb.propagate(graph, {"A": 0.9, "C": 0.7}, spiyweb.PropagationConfig())
ranked = [node for node, _ in result.ranked()]
assert ranked == ["A", "C", "D", "B", "F"], ranked
assert abs(result.energy_of("D") - 2.875) < 1e-9, result.energy_of("D")

# 7. The trace reader works with nothing installed - the D38 claim, measured
#    on the artifact. A file written by somebody's application, opened by a
#    process that has no index, no model and no numpy.
import json  # noqa: E402  - after the zero-dependency assertions, on purpose
import tempfile  # noqa: E402

from spiyweb.trace import TraceNode, TraceRecord, load_traces  # noqa: E402

record = TraceRecord(
    trace_id="smoke",
    sequence=0,
    recorded_at="2026-08-26T00:00:00.000+00:00",
    kind="plain",
    query="does the reader need the index",
    nodes=(
        TraceNode(id="A", source_id="a", layer="chunk", energy=5.6, hop=0, votes=1),
    ),
    edges=(),
    paths=(),
    clusters=(),
    events=(),
    stop_reason="threshold",
    hops_used=0,
    injected_energy=10.0,
    threshold=1.5,
    total_energy=5.6,
    node_count=1,
    dedup_mode="off",
)
with tempfile.TemporaryDirectory() as folder:
    written = Path(folder) / "traces.jsonl"
    written.write_text(json.dumps(record.to_dict()) + chr(10), encoding="utf-8")
    assert load_traces(written) == (record,), "the trace reader did not round-trip"
    assert load_traces(Path(folder)) == (record,), "a directory should resolve"

print(f"wheel smoke ok: spiyweb {spiyweb.__version__}, {len(spiyweb.__all__)} names")
