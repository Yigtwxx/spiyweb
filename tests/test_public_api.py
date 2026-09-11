"""The public surface, made executable.

This file is not here for coverage. `import spiyweb` makes a promise - these
names, that version, no heavy dependency - and until now the promise lived
only in a docstring and a `dependencies = []` line nothing ever checked.

The zero-dependency test is the load-bearing one. CI installs the `store`,
`entity` and `web` extras because FAISS and spaCy are units under test, so a
plain `import spiyweb` there succeeds no matter what the import graph looks
like. The probe therefore runs in a SUBPROCESS with every optional dependency
blocked at `sys.meta_path`. Subprocess and not in-process: purging and
re-importing `spiyweb` here would duplicate every dataclass in the session and
quietly break `isinstance` for whatever ran next.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import subprocess
import sys
from types import ModuleType

import pytest

import spiyweb
import spiyweb.indexing

PUBLIC_API = frozenset(
    {
        "AbsorptionRecord",
        "Activation",
        "ActivationPath",
        "Answer",
        "COMPARE",
        "DEFAULT_PROFILE",
        "ColoredAnswer",
        "ColoredResult",
        "ColoredRetrievalConfig",
        "ColoredRetrievalResult",
        "Confidence",
        "ConflictConfig",
        "ConflictQuestion",
        "ConflictRecord",
        "ConsolidationConfig",
        "ConsolidationReport",
        "CorpusLintConfig",
        "DedupConfig",
        "Destroyed",
        "DisputeRecord",
        "DisputeWarning",
        "EXPLORE",
        "EdgeLayer",
        "EdgeUsage",
        "EmbeddingConfig",
        "EntityEdgeConfig",
        "EntityExtractionConfig",
        "Finding",
        "GapWarning",
        "Graph",
        "LLMConfig",
        "LayerWeights",
        "LearnedLayer",
        "LearnedLayerConfig",
        "Ledger",
        "LintReport",
        "MassConfig",
        "NLICandidateConfig",
        "NLIEdgeConfig",
        "NLIModel",
        "NLIModelConfig",
        "NegativeEdge",
        "NegativeSeedConfig",
        "Node",
        "NodeLayer",
        "OutputConfig",
        "PRECISE",
        "PROFILES",
        "Passage",
        "Polarity",
        "PolarityConfig",
        "Profile",
        "PropagationConfig",
        "PropagationResult",
        "Proposition",
        "PropositionConfig",
        "RefusalReport",
        "RetrievalConfig",
        "RetrievalResult",
        "SeedSource",
        "SemanticEdgeConfig",
        "SimilarityFn",
        "SpiywebIndex",
        "StructuralEdgeConfig",
        "ThemeCluster",
        "ThermalConfig",
        "ThermalSession",
        "TraceConfig",
        "TraceLedger",
        "TraceRecord",
        "TraceStore",
        "TransformersNLIModel",
        "__version__",
        "activation_paths",
        "adaptive_threshold",
        "build_conflict_question",
        "build_derivation_edges",
        "build_ledger",
        "build_nli_edges",
        "build_refusal_report",
        "color_composition",
        "conflict_adjacency",
        "contradiction_label_index",
        "destroyed_per_node",
        "dispute_warnings",
        "entity_edge_labels",
        "extract_propositions",
        "find_survivor",
        "gap_warnings",
        "lint_corpus",
        "load_traces",
        "negative_field",
        "neutralize",
        "node_masses",
        "open_index",
        "propagate",
        "source_summary",
        "propagate_colored",
        "prune_layers",
        "retrieve",
        "retrieve_colored",
        "theme_clusters",
    }
)
"""The query-time contract. A name leaving or joining this set is a release
note, so it fails here first and gets written down on purpose."""

INDEX_TIME_API = frozenset(
    {
        "Chunk",
        "ChunkRef",
        "DocumentInput",
        "Embedder",
        "EncoderLike",
        "EntityPipeline",
        "IndexLayout",
        "IndexManifest",
        "LLMClient",
        "LLMError",
        "NativeOllamaClient",
        "OpenAICompatClient",
        "SCHEMA_VERSION",
        "SentenceTransformerEmbedder",
        "TextUnit",
        "VectorStore",
        "build_entity_edges",
        "build_index",
        "build_semantic_edges",
        "build_semantic_edges_fast",
        "build_structural_edges",
        "chunk_documents",
        "detect_device",
        "extract_entities",
        "lint_index",
        "load_entities",
        "load_graph",
        "load_nli_edges",
        "load_propositions",
        "load_similarity",
        "load_spacy_pipeline",
        "load_store",
        "load_texts",
        "read_manifest",
        "resolve_device",
        "shared_subject_pairs",
    }
)
"""The index-time contract, on the same terms. A literal rather than a
copy of `__all__`, so a name joining or leaving that list fails here
instead of agreeing with itself."""

STORE_BOUND = frozenset({"VectorStore", "build_semantic_edges_fast"})
"""The only two index-time names that need `spiyweb[store]` to resolve."""


def test_the_public_surface_is_exactly_the_declared_one() -> None:
    assert frozenset(spiyweb.__all__) == PUBLIC_API
    assert len(spiyweb.__all__) == len(PUBLIC_API), "no duplicate entries"


def test_every_declared_name_actually_resolves() -> None:
    for name in spiyweb.__all__:
        assert hasattr(spiyweb, name), name
    # STORE_BOUND excluded deliberately: `hasattr` swallows
    # AttributeError and nothing else, so touching those two without
    # faiss installed would ERROR here rather than fail. Their
    # behaviour is asserted in the isolation probe, where the missing
    # dependency is the whole point.
    for name in set(spiyweb.indexing.__all__) - STORE_BOUND:
        assert hasattr(spiyweb.indexing, name), name


def test_nothing_public_leaks_outside_the_declaration() -> None:
    """The reverse direction: an accidental re-export is a contract change too.

    Submodule attributes (`spiyweb.config`, `spiyweb.core`, ...) are set by the
    import machinery rather than by us, so they are excluded.
    """
    leaked = {
        name
        for name, value in vars(spiyweb).items()
        if not name.startswith("_") and not isinstance(value, ModuleType)
    } - set(spiyweb.__all__)
    assert leaked == set(), leaked


def test_the_measurement_harness_config_is_not_library_contract() -> None:
    """`EvaluationConfig` configures the benchmark, not the library.

    It has not moved - `spiyweb.config` is where every caller in this
    repository already imported it from - it simply stopped being re-exported
    as part of the public surface.
    """
    from spiyweb.config import EvaluationConfig, IterativeBaselineConfig

    assert EvaluationConfig is not None and IterativeBaselineConfig is not None
    assert "EvaluationConfig" not in PUBLIC_API
    assert "IterativeBaselineConfig" not in PUBLIC_API


def test_the_version_has_exactly_one_source() -> None:
    """`pyproject.toml` reads `__version__`; two copies drift, and a drifted
    version number lies about which code produced a measurement artifact.
    """
    try:
        installed = importlib.metadata.version("spiyweb")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        pytest.skip("spiyweb is not installed; run `uv sync` first")
    assert installed == spiyweb.__version__, (
        "metadata and __version__ disagree - after bumping __version__ run "
        "`uv sync --reinstall-package spiyweb`"
    )


def test_the_typing_marker_ships_with_the_package() -> None:
    """Without `py.typed`, every annotation in this package is invisible to
    mypy and pyright from the outside."""
    marker = importlib.resources.files("spiyweb").joinpath("py.typed")
    assert marker.is_file()


def test_the_index_time_facade_is_separate_from_the_query_contract() -> None:
    """Building a web and querying one are different jobs with different
    dependency profiles, so they get different front doors."""
    assert frozenset(spiyweb.indexing.__all__) == INDEX_TIME_API
    assert INDEX_TIME_API & PUBLIC_API == frozenset()
    assert STORE_BOUND <= INDEX_TIME_API


def test_the_scene_module_stays_outside_the_zero_dependency_promise() -> None:
    """`spiyweb.scene` costs numpy, so it is an extra and never an eager import.

    It was promoted into the package in Faz 2.2 so that every consumer draws
    one picture rather than reimplementing the layout.
    That promotion is only free while `import spiyweb` stays untouched by it.
    """
    assert "scene" not in PUBLIC_API
    assert not any(name.startswith("scene") for name in spiyweb.__all__)
    assert "spiyweb.scene" not in sys.modules or "numpy" in sys.modules, (
        "scene may only be loaded once numpy is; it must never ride in on "
        "`import spiyweb`"
    )


_ISOLATION_PROBE = """
import sys

# Hand-maintained, and drift is survivable: this probe tests the
# import GRAPH, while the `wheel` CI job tests the installed
# ARTIFACT in an environment where nothing optional exists at all.
# A heavy dependency missing from this tuple still fails there.
BANNED = (
    "numpy", "faiss", "torch", "spacy", "sentence_transformers",
    "transformers", "streamlit", "fastapi", "uvicorn", "pydantic",
)


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BANNED:
            raise ImportError(name + " is blocked by the zero-dependency guard")
        return None


sys.meta_path.insert(0, _Blocker())
for name in [n for n in sys.modules if n.split(".")[0] in BANNED]:
    del sys.modules[name]

import spiyweb
import spiyweb.indexing

for attr in spiyweb.__all__:
    getattr(spiyweb, attr)
for attr in spiyweb.indexing.__all__:
    if attr in ("VectorStore", "build_semantic_edges_fast"):
        continue
    getattr(spiyweb.indexing, attr)

try:
    spiyweb.indexing.VectorStore
except ImportError as error:
    assert "spiyweb[store]" in str(error), str(error)
else:
    raise SystemExit("VectorStore resolved with faiss blocked")

print("ok")
"""


def test_the_whole_surface_imports_with_zero_dependencies() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _ISOLATION_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout
