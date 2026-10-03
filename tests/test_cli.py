"""The terminal command, and the one promise it makes on a bare install.

`spiyweb version` has to work with nothing installed. That is not a nicety:
the moment somebody types `spiyweb query` and gets an import error, the next
thing they need is a straight answer about which extra is missing, and a
`version` command that itself needs numpy to run cannot give it. So the
subprocess probe here blocks every optional dependency and runs the command
anyway - the same guard `tests/test_public_api.py` puts on `import spiyweb`.

The rest is ordinary: each verb wraps something the library already does, so
what is worth testing is the wiring and the failure messages, not the
retrieval underneath - that has its own tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING, ClassVar

import pytest

from spiyweb.cli import EXTRAS, Problem, build_parser, main

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spiyweb.session import SpiywebIndex


def test_every_verb_is_reachable() -> None:
    parser = build_parser()
    for verb in ("version", "index", "query", "lint", "watch", "menu"):
        assert parser.parse_args([verb, *_stub(verb)]).command == verb


def _stub(verb: str) -> list[str]:
    return {
        "version": [],
        "index": ["docs", "out"],
        "query": ["idx", "a question"],
        "lint": ["idx"],
        "watch": [],
        "menu": [],
    }[verb]


def test_no_verb_is_the_guided_menu_rather_than_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare `spiyweb` used to be a usage error at somebody who had not
    learned the verbs yet. It opens the menu now - and in a pipe, where
    there is nobody to answer, it still refuses rather than blocking."""
    assert build_parser().parse_args([]).command is None
    monkeypatch.setattr("spiyweb.wizard.is_interactive", lambda: False)
    assert main([]) == 2


def test_version_reports_every_extra(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["extras"]) == set(EXTRAS)
    assert all(isinstance(value, bool) for value in payload["extras"].values())


def test_version_names_the_pip_line_for_what_is_missing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "It does not work" almost always means an extra is not installed."""
    monkeypatch.setattr("spiyweb.cli._installed", lambda modules: False)
    assert main(["version"]) == 0
    printed = capsys.readouterr().out
    assert 'pip install "spiyweb[' in printed
    for extra in EXTRAS:
        assert extra in printed


_BARE_PROBE = """
import sys

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

from spiyweb.cli import main

assert main(["version"]) == 0
print("ok")
"""


def test_version_works_on_a_bare_install() -> None:
    """The command that explains a missing extra must not need one itself."""
    done = subprocess.run(
        [sys.executable, "-c", _BARE_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


# --- query ----------------------------------------------------------------


def test_query_prints_what_lit_up(
    capsys: pytest.CaptureFixture[str],
    open_tiny: Callable[..., SpiywebIndex],
    tiny_index_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI is a face over `SpiywebIndex`, so it must show the same run."""
    index = open_tiny()
    monkeypatch.setattr("spiyweb.cli._open", lambda path, **options: index)

    assert main(["query", str(tiny_index_root), "who raised the tower"]) == 0
    printed = capsys.readouterr().out
    assert "atoms" in printed
    assert "stopped on threshold" in printed
    assert "The tower was raised on the shore." in printed
    # Energy is drawn, not only printed: the decay is the mechanism.
    assert "█" in printed, "no energy bar was drawn"
    # ...and captured output is not a terminal, so no escape codes leaked.
    assert "[" not in printed


def test_query_json_carries_the_confidence_signal(
    capsys: pytest.CaptureFixture[str],
    open_tiny: Callable[..., SpiywebIndex],
    tiny_index_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D17: the library reports the signal, the caller owns the policy."""
    index = open_tiny()
    monkeypatch.setattr("spiyweb.cli._open", lambda path, **options: index)

    assert main(["query", str(tiny_index_root), "who raised the tower", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["confidence"]["node_count"] > 0
    assert payload["confidence"]["hop_depth"] >= 0
    assert payload["passages"][0]["energy"] > 0
    assert payload["passages"][0]["text"]


def test_query_refuses_a_directory_that_is_not_an_index(tmp_path: Path) -> None:
    with pytest.raises(Problem, match="does not look like an index"):
        main(["query", str(tmp_path), "anything"])


# --- index ----------------------------------------------------------------


def test_a_directory_of_text_becomes_documents(tmp_path: Path) -> None:
    """One file is one document; a blank line separates units."""
    from spiyweb.cli import _read_documents

    (tmp_path / "nested").mkdir()
    (tmp_path / "one.md").write_text("first block\n\nsecond block\n", "utf-8")
    (tmp_path / "nested" / "two.txt").write_text("only block", "utf-8")
    (tmp_path / "ignored.bin").write_text("not text", "utf-8")

    documents = _read_documents(tmp_path, None, whole=False)
    assert [document.source_id for document in documents] == [
        "nested/two.txt",
        "one.md",
    ]
    assert [len(document.units) for document in documents] == [1, 2]
    assert documents[1].units[0].text == "first block"


def test_corpus_discovery_ignores_case_and_orders_by_source_id(
    tmp_path: Path,
) -> None:
    """`README.TXT` was indexed on Windows and skipped on macOS and Linux.

    Order is by the posix relative path - the source id - on every
    platform, so `nodes.json` from two machines is byte-identical.
    """
    from spiyweb.cli import _read_documents

    (tmp_path / "nested").mkdir()
    (tmp_path / "Upper.TXT").write_text("shouted", "utf-8")
    (tmp_path / "lower.md").write_text("whispered", "utf-8")
    (tmp_path / "nested" / "z.Markdown").write_text("deep", "utf-8")

    documents = _read_documents(tmp_path, None, whole=True)
    ids = [document.source_id for document in documents]
    assert ids == ["Upper.TXT", "lower.md", "nested/z.Markdown"]
    assert ids == sorted(ids), "byte order of the posix path, not the OS's"


def test_whole_file_keeps_a_document_in_one_piece(tmp_path: Path) -> None:
    from spiyweb.cli import _read_documents

    (tmp_path / "one.md").write_text("first block\n\nsecond block\n", "utf-8")
    documents = _read_documents(tmp_path, None, whole=True)
    assert len(documents[0].units) == 1


def test_an_empty_corpus_says_what_it_looked_for(tmp_path: Path) -> None:
    from spiyweb.cli import _read_documents

    with pytest.raises(Problem, match="looked for"):
        _read_documents(tmp_path, None, whole=False)


def test_a_missing_corpus_directory_is_named(tmp_path: Path) -> None:
    from spiyweb.cli import _read_documents

    with pytest.raises(Problem, match="is not a directory"):
        _read_documents(tmp_path / "absent", None, whole=False)


# --- lint ------------------------------------------------------------------


def test_lint_reports_a_real_index(
    capsys: pytest.CaptureFixture[str], tiny_index_root: Path
) -> None:
    """The corpus-lint verb reads artifacts; no query and no model involved."""
    assert main(["lint", str(tiny_index_root)]) == 0
    printed = capsys.readouterr().out
    assert "atom(s)" in printed
    assert "connected component(s)" in printed


def test_lint_json_carries_the_findings(
    capsys: pytest.CaptureFixture[str], tiny_index_root: Path
) -> None:
    assert main(["lint", str(tiny_index_root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["nodes"] > 0
    assert payload["components"] >= 1
    assert isinstance(payload["findings"], list)
    for finding in payload["findings"]:
        assert finding["kind"] and finding["message"]


def test_lint_thresholds_reach_the_report(
    capsys: pytest.CaptureFixture[str], tiny_index_root: Path
) -> None:
    """The flags are reporting knobs, so loosening one must change the output."""
    assert (
        main(["lint", str(tiny_index_root), "--duplicate-weight", "0.99", "--json"])
        == 0
    )
    strict = json.loads(capsys.readouterr().out)
    assert (
        main(["lint", str(tiny_index_root), "--duplicate-weight", "0.1", "--json"]) == 0
    )
    loose = json.loads(capsys.readouterr().out)
    assert len(loose["findings"]) > len(strict["findings"])


def test_lint_refuses_a_directory_that_is_not_an_index(tmp_path: Path) -> None:
    with pytest.raises(Problem, match="does not look like an index"):
        main(["lint", str(tmp_path)])


def test_output_survives_a_console_that_cannot_spell_the_corpus() -> None:
    """A diagnostic that crashes while diagnosing is worse than a lost glyph.

    Python encodes stdout with the console codepage; on a Turkish Windows box
    that is cp1254, and this project indexes Turkish and English text by
    design. Printing a passage with a macron in it raised UnicodeEncodeError
    out of `print` - found while hand-verifying a lint finding on the real
    MuSiQue index.
    """
    import io

    from spiyweb.cli import _writable_stdout

    narrow = io.TextIOWrapper(io.BytesIO(), encoding="cp1254", errors="strict")
    original = sys.stdout
    sys.stdout = narrow
    try:
        _writable_stdout()
        print("Kalnciems \u0113")
    finally:
        sys.stdout = original


def test_query_defaults_to_a_profile_that_can_actually_spread(
    capsys: pytest.CaptureFixture[str],
    open_tiny: Callable[..., SpiywebIndex],
    tiny_index_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bare library default returns first contact only, and would have
    shown a first-time reader `top-k` with extra steps."""
    from spiyweb.cli import DEFAULT_PROFILE

    index = open_tiny()
    asked: list[str | None] = []
    original = type(index).retrieve

    def record(self: SpiywebIndex, query: str, **options: object) -> object:
        asked.append(options.get("profile"))  # type: ignore[arg-type]
        return original(self, query, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(type(index), "retrieve", record)
    monkeypatch.setattr("spiyweb.cli._open", lambda path, **options: index)

    assert main(["query", str(tiny_index_root), "who raised the tower"]) == 0
    assert asked == [DEFAULT_PROFILE]
    assert f"profile {DEFAULT_PROFILE}" in capsys.readouterr().out


def test_an_explicit_profile_still_wins(
    open_tiny: Callable[..., SpiywebIndex],
    tiny_index_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = open_tiny()
    asked: list[str | None] = []
    original = type(index).retrieve

    def record(self: SpiywebIndex, query: str, **options: object) -> object:
        asked.append(options.get("profile"))  # type: ignore[arg-type]
        return original(self, query, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(type(index), "retrieve", record)
    monkeypatch.setattr("spiyweb.cli._open", lambda path, **options: index)

    assert (
        main(["query", str(tiny_index_root), "a question", "--profile", "precise"]) == 0
    )
    assert asked == ["precise"]


# --- index --propositions ----------------------------------------------------


class _LengthEmbedder:
    """Any text gets a vector; proposition strings are not known in advance."""

    model_name = "fake-e5"

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text) % 7), float(len(text) % 3)] for text in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)


class _NoEntities:
    def pipe(self, texts: list[str]) -> list[object]:
        class Doc:
            ents: tuple[object, ...] = ()

        return [Doc() for _ in texts]


class _ScriptedClient:
    """Stands in for `OpenAICompatClient`: one fact back per call, all logged."""

    calls: ClassVar[list[str]] = []
    models: ClassVar[list[str]] = []

    def __init__(self, config: object) -> None:
        type(self).models.append(config.model)  # type: ignore[attr-defined]

    def complete(self, prompt: str) -> str:
        type(self).calls.append(prompt)
        return f"Fact number {len(type(self).calls)} holds."


@pytest.fixture
def index_stubs(monkeypatch: pytest.MonkeyPatch) -> type[_ScriptedClient]:
    """The three heavy objects `index` builds, swapped for fakes."""
    _ScriptedClient.calls = []
    _ScriptedClient.models = []
    monkeypatch.setattr(
        "spiyweb.embedding.SentenceTransformerEmbedder", _LengthEmbedder
    )
    monkeypatch.setattr("spiyweb.entities.load_spacy_pipeline", _NoEntities)
    monkeypatch.setattr("spiyweb.llm.OpenAICompatClient", _ScriptedClient)
    return _ScriptedClient


def _corpus(root: Path) -> Path:
    docs = root / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Tesla built a tower.\n\nIt was torn down.\n")
    (docs / "b.md").write_text("Wardenclyffe stood on Long Island.\n")
    return docs


def test_index_propositions_adds_the_layer_through_the_default_llm(
    tmp_path: Path, index_stubs: type[_ScriptedClient]
) -> None:
    from spiyweb.config import LLMConfig
    from spiyweb.indexing import IndexLayout, read_manifest

    out = tmp_path / "idx"
    assert main(["index", str(_corpus(tmp_path)), str(out), "--propositions"]) == 0

    assert len(index_stubs.calls) == 3, "one call per unit, entity fallback off"
    assert index_stubs.models == [LLMConfig().model]
    assert read_manifest(out).propositions == 3
    cache = IndexLayout.at(out).llm_cache_jsonl
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 3


def test_an_interrupted_extraction_resumes_from_the_cache(
    tmp_path: Path, index_stubs: type[_ScriptedClient]
) -> None:
    docs, out = _corpus(tmp_path), tmp_path / "idx"
    assert main(["index", str(docs), str(out), "--propositions"]) == 0
    index_stubs.calls.clear()

    # --force rebuilds every artifact, so only the cache can spare the calls.
    assert main(["index", str(docs), str(out), "--propositions", "--force"]) == 0
    assert index_stubs.calls == []


def test_a_plain_index_never_builds_an_llm(
    tmp_path: Path, index_stubs: type[_ScriptedClient]
) -> None:
    from spiyweb.indexing import read_manifest

    out = tmp_path / "idx"
    assert main(["index", str(_corpus(tmp_path)), str(out)]) == 0
    assert index_stubs.models == []
    assert read_manifest(out).propositions == 0


def test_another_model_gets_its_own_cache(
    tmp_path: Path, index_stubs: type[_ScriptedClient]
) -> None:
    out = tmp_path / "idx"
    argv = ["index", str(_corpus(tmp_path)), str(out), "--propositions"]
    assert main([*argv, "--llm-model", "qwen2.5:7b"]) == 0
    assert index_stubs.models == ["qwen2.5:7b"]
    assert (out / "llm_cache_qwen2.5-7b.jsonl").exists()
    assert not (out / "llm_cache.jsonl").exists()


def test_llm_flags_without_propositions_are_refused(
    tmp_path: Path, index_stubs: type[_ScriptedClient]
) -> None:
    argv = ["index", str(_corpus(tmp_path)), str(tmp_path / "idx")]
    with pytest.raises(Problem, match="--llm-model only applies with --propositions"):
        main([*argv, "--llm-model", "qwen2.5:7b"])
    assert not (tmp_path / "idx").exists(), "refused before any work"


def test_an_unreachable_llm_names_the_fix(
    tmp_path: Path,
    index_stubs: type[_ScriptedClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spiyweb.llm import LLMError

    def refuse(self: _ScriptedClient, prompt: str) -> str:
        raise LLMError("connection refused")

    monkeypatch.setattr(_ScriptedClient, "complete", refuse)
    argv = ["index", str(_corpus(tmp_path)), str(tmp_path / "idx"), "--propositions"]
    with pytest.raises(Problem, match=r"ollama pull llama3\.1:8b"):
        main(argv)
