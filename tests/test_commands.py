"""The `/` commands, run against a monitor that never touches a terminal."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from conftest import canonical_record
from spiyweb.commands import (
    COMMANDS,
    dispatch,
    find_usages,
    parse,
    pip_command,
    problem_hint,
)
from spiyweb.config import WatchConfig
from spiyweb.watch import Monitor

if TYPE_CHECKING:
    from pathlib import Path


def quiet_monitor(tmp_path: Path) -> Monitor:
    return Monitor(
        config=WatchConfig(),
        directory=tmp_path / ".spiyweb",
        write=io.StringIO().write,
        poll=lambda _t: None,
        clock=lambda: 0.0,
        size=lambda: (120, 30),
        color=False,
        unicode=True,
        cwd=tmp_path,
    )


def transcript(monitor: Monitor) -> str:
    return "\n".join(monitor.transcript)


# --- parsing --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "name", "args"),
    [
        ("/query idx what now", "query", ("idx", "what", "now")),
        ("  /HELP  ", "help", ()),
        ("/", "help", ()),
        ("/replay 3", "replay", ("3",)),
    ],
)
def test_parse_splits_name_and_arguments(
    line: str, name: str, args: tuple[str, ...]
) -> None:
    invocation = parse(line)
    assert invocation is not None
    assert (invocation.name, invocation.args) == (name, args)


@pytest.mark.parametrize(
    ("line", "args"),
    [
        ('/index "C:\\My Docs\\notes" out', ("C:\\My Docs\\notes", "out")),
        ("/index 'my docs' \"my index\"", ("my docs", "my index")),
        ('/index "unclosed path', ("unclosed path",)),
        ('/query "" now', ("", "now")),
        (
            "/compare why did Tesla's tower fail",
            ("why", "did", "Tesla's", "tower", "fail"),
        ),
    ],
)
def test_parse_keeps_a_quoted_path_together_and_its_backslashes(
    line: str, args: tuple[str, ...]
) -> None:
    invocation = parse(line)
    assert invocation is not None
    assert invocation.args == args


def test_text_without_a_slash_is_a_question_not_a_command() -> None:
    assert parse("who built the tower") is None


def test_every_command_appears_in_help(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/help")
    shown = transcript(monitor)
    for command in COMMANDS:
        assert command.usage in shown, command.usage


def test_the_old_menu_items_are_all_commands() -> None:
    names = {command.name for command in COMMANDS}
    assert {"query", "lint", "index", "install", "config", "doctor"} <= names
    assert not {"menu", "version"} & names, "the form and /doctor replace them"


# --- /find ------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_find_lists_imports_and_suggests_the_opened_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "app" / "main.py",
        "import os\nfrom spiyweb import open_index\n\n"
        "index = open_index(\"my-index\")\nother = SpiywebIndex.open('elsewhere')\n",
    )
    _write(tmp_path / ".venv" / "lib" / "x.py", "import spiyweb\n")
    _write(tmp_path / "node_modules" / "y.py", "import spiyweb\n")
    _write(tmp_path / "a/b/c/d/e/f/g/deep.py", "import spiyweb\n")
    _write(
        tmp_path / "vendor" / "spiyweb" / "__init__.py", "from spiyweb.core import x\n"
    )
    _write(tmp_path / "vendor" / "spiyweb" / "core" / "propagate.py", "")
    (tmp_path / "my-index").mkdir()
    (tmp_path / "my-index" / "nodes.json").write_text("[]", encoding="utf-8")

    report = find_usages(tmp_path, max_depth=6, max_files=100, max_file_bytes=10_000)
    assert [(u.path.name, u.line) for u in report.usages] == [("main.py", 2)]
    assert report.index_hints == ("my-index", "elsewhere")

    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/find")
    shown = transcript(monitor)
    assert "main.py" in shown and "1 import, first at line 2" in shown
    assert "main.py:2" not in shown, "one row per file, not per line"
    assert monitor.active_index is not None
    assert monitor.active_index.endswith("my-index")
    assert "elsewhere" in shown and "not found" in shown


def test_find_stops_at_the_file_cap_and_says_so(tmp_path: Path) -> None:
    for number in range(5):
        _write(tmp_path / f"f{number}.py", "import spiyweb\n")
    report = find_usages(tmp_path, max_depth=2, max_files=3, max_file_bytes=10_000)
    assert report.truncated and report.scanned == 3


def test_find_with_nothing_to_find_is_a_warning(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/find")
    assert "nothing imports spiyweb" in transcript(monitor)


# --- /install ---------------------------------------------------------------------


def test_pip_command_prefers_pip_then_uv_then_gives_up() -> None:
    assert pip_command("index", python="py", has_pip=True) == [
        "py", "-m", "pip", "install", "spiyweb[index]"
    ]  # fmt: skip
    assert pip_command("view", python="py", has_pip=False, which=lambda _: "/uv") == [
        "/uv", "pip", "install", "--python", "py", "spiyweb[view]"
    ]  # fmt: skip
    assert pip_command("view", python="py", has_pip=False, which=lambda _: None) is None


class Spawned:
    """What `Monitor.spawn` was asked to start, instead of starting it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, monitor: Monitor) -> None:
        self.calls: list[dict[str, object]] = []

        def fake_spawn(argv: list[str], **options: object) -> object:
            self.calls.append({"argv": list(argv), **options})
            return object()

        monkeypatch.setattr(monitor, "spawn", fake_spawn)

    @property
    def argvs(self) -> list[list[str]]:
        return [call["argv"] for call in self.calls]  # type: ignore[misc]


class FakeJob:
    def __init__(self, exit_code: int = 0, *, stopped: bool = False) -> None:
        self.exit_code, self.stopped = exit_code, stopped


@pytest.fixture
def no_slow_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/doctor` without a network call or a torch import."""
    monkeypatch.setattr("spiyweb.commands._answers", lambda url, timeout: False)
    real = __import__("spiyweb.cli", fromlist=["_installed"])._installed
    monkeypatch.setattr(
        "spiyweb.cli._installed",
        lambda modules: False if "torch" in modules else real(modules),
    )


def test_doctor_names_every_missing_piece_with_its_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("spiyweb.cli._installed", lambda modules: False)
    monkeypatch.setattr("spiyweb.commands._answers", lambda url, timeout: False)
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/doctor")
    shown = transcript(monitor)
    assert "[store]" in shown and "/install store" in shown
    assert "spaCy model" in shown and "/install index fetches it" in shown
    assert "not answering" in shown and "proposition layer" in shown
    assert "no torch yet" in shown, "torch absent: nothing imported, said so"


def test_doctor_reports_the_device_when_torch_is_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("spiyweb.cli._installed", lambda modules: True)
    monkeypatch.setattr("spiyweb.commands._answers", lambda url, timeout: True)
    monkeypatch.setattr("spiyweb.embedding.detect_device", lambda: "cuda")
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/doctor")
    shown = transcript(monitor)
    assert "device" in shown and "cuda" in shown
    assert "not installed" not in shown


def test_install_without_an_argument_is_the_doctor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_slow_checks: None
) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/install")
    assert "[store]" in transcript(monitor) and "spaCy model" in transcript(monitor)


def test_install_refuses_an_unknown_extra_without_running_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor = quiet_monitor(tmp_path)
    spawned = Spawned(monkeypatch, monitor)
    dispatch(monitor, "/install evil;rm")
    assert "no extra called" in transcript(monitor)
    assert spawned.calls == []


def test_install_asks_first_and_runs_in_the_background_only_on_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_slow_checks: None
) -> None:
    monkeypatch.setattr("spiyweb.commands.pip_command", lambda extra: ["pip", extra])
    monitor = quiet_monitor(tmp_path)
    spawned = Spawned(monkeypatch, monitor)
    dispatch(monitor, "/install view")
    assert monitor.prompt is not None and "(y/n)" in monitor.prompt.question
    monitor.buffer = "n"
    monitor._submit()
    assert spawned.calls == [] and "not installed" in transcript(monitor)

    dispatch(monitor, "/install view")
    monitor.buffer = "y"
    monitor._submit()
    assert spawned.argvs == [["pip", "view"]]
    on_done = spawned.calls[0]["on_done"]
    on_done(FakeJob(0))  # type: ignore[operator]
    assert "[store]" in transcript(monitor), "the doctor view follows the install"


def test_installing_index_fetches_the_spacy_model_when_it_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_slow_checks: None
) -> None:
    from spiyweb.config import EntityExtractionConfig

    model = EntityExtractionConfig().spacy_model
    monkeypatch.setattr("spiyweb.commands.pip_command", lambda extra: ["pip", extra])
    monkeypatch.setattr("spiyweb.cli._installed", lambda modules: model not in modules)
    monitor = quiet_monitor(tmp_path)
    spawned = Spawned(monkeypatch, monitor)
    dispatch(monitor, "/install index")
    monitor.buffer = "y"
    monitor._submit()
    spawned.calls[0]["on_done"](FakeJob(0))  # type: ignore[operator]
    assert spawned.argvs[1] == [sys.executable, "-m", "spacy", "download", model]


def test_a_failed_install_says_so_and_fetches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("spiyweb.commands.pip_command", lambda extra: ["pip", extra])
    monitor = quiet_monitor(tmp_path)
    spawned = Spawned(monkeypatch, monitor)
    dispatch(monitor, "/install index")
    monitor.buffer = "y"
    monitor._submit()
    spawned.calls[0]["on_done"](FakeJob(1))  # type: ignore[operator]
    assert "installing [index] failed" in transcript(monitor)
    assert len(spawned.calls) == 1


def test_index_runs_in_the_background_and_its_index_becomes_the_active_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spiyweb.progress import IndexProgress

    monitor = quiet_monitor(tmp_path)
    spawned: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        monitor,
        "spawn_spiyweb",
        lambda args, **options: spawned.append((list(args), options)),
    )
    dispatch(monitor, '/index "my docs" out --propositions')
    args, options = spawned[0]
    assert args == ["index", "my docs", "out", "--propositions"]
    progress = options["progress"]
    assert isinstance(progress, IndexProgress) and progress.propositions
    assert monitor.prompt is None, "nothing blocks: the job runs, the screen lives"
    options["on_done"](FakeJob(0))  # type: ignore[operator]
    assert monitor.active_index == str(tmp_path / "out")
    assert "out is ready" in transcript(monitor)


def test_a_stopped_index_job_points_at_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor = quiet_monitor(tmp_path)
    spawned: list[dict[str, object]] = []
    monkeypatch.setattr(
        monitor, "spawn_spiyweb", lambda args, **options: spawned.append(options)
    )
    dispatch(monitor, "/index docs out")
    spawned[0]["on_done"](FakeJob(1, stopped=True))  # type: ignore[operator]
    assert "--force" in transcript(monitor)
    assert monitor.active_index is None


def test_a_cli_problem_about_an_extra_becomes_an_install_hint() -> None:
    message = (
        'No module named faiss\nopening an index needs: pip install "spiyweb[index]"'
    )
    assert problem_hint(message).endswith("/install index")
    assert problem_hint("plain trouble") == "plain trouble"


# --- /query, /lint, /replay -------------------------------------------------------


def test_query_with_no_index_known_offers_a_choice_or_asks(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/query what")
    assert monitor.prompt is not None, "no index nearby: a path is asked for"
    assert "path to an index" in monitor.prompt.question


def test_a_plain_question_needs_an_index_first(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "who built the tower")
    shown = transcript(monitor)
    assert "no index here yet" in shown and "/index" in shown and "/demo" in shown


def test_query_plays_its_own_record_instead_of_writing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePassage:
        node_id, energy, text = "A", 5.6, "Tesla built Wardenclyffe on Long Island."

    class FakeAnswer:
        trace = canonical_record()
        passages = (FakePassage(),)

    class FakeIndex:
        def retrieve(self, question: str, *, profile: str) -> FakeAnswer:
            assert profile == "explore"
            return FakeAnswer()

    opened: list[object] = []

    def fake_open(path: str, **options: object) -> FakeIndex:
        opened.append(options["trace"])
        return FakeIndex()

    monkeypatch.setattr("spiyweb.cli._open", fake_open)
    monkeypatch.setattr("spiyweb.commands.is_index", lambda path: str(path) == "idx")
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/query idx who built the tower")
    assert len(monitor.queue) == 1, "queued to play, not written to the file"
    assert opened[0].attach_dir is None, "never through the marker, or it replays"  # type: ignore[attr-defined]
    record, after = monitor.queue[0]
    assert record.query == "which tower did tesla build"
    assert any("Wardenclyffe" in line for line in after)
    assert monitor.active_index is not None
    assert monitor.active_index.endswith("idx"), "a question's index becomes active"
    assert not (tmp_path / ".spiyweb" / "traces.jsonl").exists()


class _ClosableIndex:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_an_index_is_opened_once_and_reopened_after_a_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from spiyweb.commands import open_cached

    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    meta = index_dir / "meta.json"
    meta.write_text("{}", encoding="utf-8")
    opened: list[_ClosableIndex] = []

    def fake_open(path: str, **options: object) -> _ClosableIndex:
        assert "embedder" not in options, "no default model named: no sharing"
        opened.append(_ClosableIndex())
        return opened[-1]

    monkeypatch.setattr("spiyweb.cli._open", fake_open)
    monitor = quiet_monitor(tmp_path)
    monkeypatch.chdir(tmp_path)
    first = open_cached(monitor, "idx")
    assert open_cached(monitor, str(index_dir)) is first, "one index, two spellings"
    assert len(opened) == 1
    later = meta.stat().st_mtime_ns + 5_000_000_000
    os.utime(meta, ns=(later, later))  # what a rebuild does to the receipt
    second = open_cached(monitor, "idx")
    assert second is not first and first.closed, "rebuilt: reopened, old closed"


def test_an_index_naming_the_default_model_shares_one_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from spiyweb.commands import open_cached
    from spiyweb.config import EmbeddingConfig

    built: list[object] = []

    class FakeEmbedder:
        def __init__(self) -> None:
            built.append(self)

    monkeypatch.setattr("spiyweb.embedding.SentenceTransformerEmbedder", FakeEmbedder)
    given: list[object] = []

    def fake_open(path: str, **options: object) -> _ClosableIndex:
        given.append(options.get("embedder"))
        return _ClosableIndex()

    monkeypatch.setattr("spiyweb.cli._open", fake_open)
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "meta.json").write_text(
            json.dumps({"embedding_model": EmbeddingConfig().model}), encoding="utf-8"
        )
    monitor = quiet_monitor(tmp_path)
    open_cached(monitor, str(tmp_path / "a"))
    open_cached(monitor, str(tmp_path / "b"))
    assert len(built) == 1, "one model for both indexes"
    assert given == [built[0], built[0]]
    assert "loading the embedding model" in transcript(monitor)


def test_a_library_warning_lands_in_the_transcript_not_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import warnings

    class WarningIndex:
        def retrieve(self, question: str, *, profile: str) -> object:
            warnings.warn("these settings cannot spread", UserWarning, stacklevel=1)
            answer_type = type("A", (), {"trace": None, "passages": ()})
            return answer_type()

    monkeypatch.setattr("spiyweb.cli._open", lambda path, **o: WarningIndex())
    monitor = quiet_monitor(tmp_path)
    from spiyweb.commands import run_query

    run_query(monitor, str(tmp_path), "anything")
    assert "these settings cannot spread" in transcript(monitor)
    assert capsys.readouterr().err == ""


def test_replay_plays_the_last_n_recorded(tmp_path: Path) -> None:
    from spiyweb.config import TraceConfig
    from spiyweb.trace import TraceStore

    store = TraceStore(TraceConfig(directory=tmp_path / ".spiyweb"))
    for number in range(3):
        store.append(canonical_record(sequence=number))
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/replay 2")
    assert [r.sequence for r, _ in monitor.queue] == [1, 2]


def test_lint_runs_the_cli_verb_as_a_background_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("spiyweb.commands.is_index", lambda path: str(path) == "idx")
    monitor = quiet_monitor(tmp_path)
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        monitor, "spawn_spiyweb", lambda args, **options: spawned.append(list(args))
    )
    dispatch(monitor, "/lint idx")
    assert spawned == [["lint", "idx"]]


# --- /config ------------------------------------------------------------------------


def test_config_is_an_arrow_key_list_that_cycles_and_persists(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/config")
    picker = monitor.picker
    assert picker is not None and picker.sticky
    assert picker.options[0][0] == "profile"
    monitor.handle_key("enter", 0.0)  # profile: explore -> precise
    assert monitor.settings.profile == "precise"
    monitor.handle_key("down", 0.0)
    monitor.handle_key(" ", 0.0)  # map: on -> off
    assert monitor.settings.map_enabled is False
    assert monitor.config.map_enabled is False, "the live config follows"
    monitor.handle_key("escape", 0.0)
    assert monitor.picker is None
    reloaded = quiet_monitor(tmp_path)
    assert reloaded.settings.profile == "precise"
    assert reloaded.settings.map_enabled is False


def test_a_picker_takes_the_arrows_and_the_prompt_takes_the_text(
    tmp_path: Path,
) -> None:
    monitor = quiet_monitor(tmp_path)
    chosen: list[str] = []
    monitor.pick("which?", [("a", "first"), ("b", "second")], chosen.append)
    for key in ("down", "enter"):
        monitor.handle_key(key, 0.0)
    assert chosen == ["b"] and monitor.picker is None
    answers: list[str] = []
    monitor.ask("name?", answers.append, default="anon")
    monitor.handle_key("enter", 0.0)
    assert answers == ["anon"]


# --- /indexes, the /index form, where a question goes -------------------------


def _make_index(path: Path, **meta: object) -> Path:
    import json

    path.mkdir(parents=True)
    (path / "nodes.json").write_text("[]", encoding="utf-8")
    if meta:
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_indexes_lists_what_is_near_and_enter_makes_one_active(
    tmp_path: Path,
) -> None:
    from spiyweb.config import EmbeddingConfig

    _make_index(tmp_path / "alpha", nodes=12, embedding_model=EmbeddingConfig().model)
    _make_index(tmp_path / "toy", nodes=5, embedding_model="hash-bow-64")
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/indexes")
    picker = monitor.picker
    assert picker is not None
    labels = "\n".join(label for _, label in picker.options)
    assert "alpha" in labels and "12 nodes" in labels
    assert "built with hash-bow-64" in labels, "an index this monitor cannot ask"
    monitor.handle_key("enter", 0.0)
    assert monitor.active_index is not None
    assert monitor.active_index.endswith("alpha")
    assert "questions now go to alpha" in transcript(monitor)


def test_indexes_with_none_near_says_how_to_build_one(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/indexes")
    assert "no index nearby" in transcript(monitor) and monitor.picker is None


def test_a_question_with_one_index_nearby_just_uses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_index(tmp_path / "only")
    asked: list[str] = []
    monkeypatch.setattr(
        "spiyweb.commands.run_query",
        lambda monitor, index, question: asked.append(index),
    )
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "who built the tower")
    assert len(asked) == 1 and asked[0].endswith("only")
    assert "using only" in transcript(monitor)


def test_a_question_with_several_indexes_nearby_asks_which(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_index(tmp_path / "one")
    _make_index(tmp_path / "two")
    asked: list[str] = []
    monkeypatch.setattr(
        "spiyweb.commands.run_query",
        lambda monitor, index, question: asked.append(question),
    )
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "who built the tower")
    assert monitor.picker is not None and asked == []
    monitor.handle_key("enter", 0.0)
    assert asked == ["who built the tower"]


def test_index_without_arguments_offers_the_folders_with_text(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "docs" / "b.txt").write_text("y", encoding="utf-8")
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/index")
    picker = monitor.picker
    assert picker is not None and "which folder" in picker.title
    assert "docs/" in picker.options[0][1] and "2 files" in picker.options[0][1]
    monitor.handle_key("enter", 0.0)
    form = monitor.picker
    assert form is not None and form.sticky, "the options come next"
    assert "docs-index" in form.options[0][1]


def test_the_index_form_toggles_options_and_builds_with_their_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor = quiet_monitor(tmp_path)
    started: list[list[str]] = []
    monkeypatch.setattr(
        monitor,
        "spawn_spiyweb",
        lambda args, **options: started.append(list(args)),
    )
    dispatch(monitor, "/index notes")
    keys = {value: n for n, (value, _) in enumerate(monitor.picker.options)}  # type: ignore[union-attr]

    def choose(value: str) -> None:
        picker = monitor.picker
        assert picker is not None
        picker.cursor = keys[value]
        monitor.handle_key("enter", 0.0)

    choose("propositions")
    choose("force")
    assert "on" in monitor.picker.options[keys["force"]][1]  # type: ignore[union-attr]
    choose("out")
    assert monitor.prompt is not None and monitor.picker is None
    monitor.buffer = "corpus-index"
    monitor._submit()
    assert monitor.picker is not None, "the form comes back after the answer"
    assert "corpus-index" in monitor.picker.options[0][1]
    choose("build")
    assert started == [["index", "notes", "corpus-index", "--propositions", "--force"]]
    assert monitor.picker is None


def test_cancel_in_the_index_form_builds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor = quiet_monitor(tmp_path)
    started: list[object] = []
    monkeypatch.setattr(
        monitor, "spawn_spiyweb", lambda args, **options: started.append(args)
    )
    dispatch(monitor, "/index notes")
    picker = monitor.picker
    assert picker is not None
    picker.cursor = len(picker.options) - 1
    monitor.handle_key("enter", 0.0)
    assert started == [] and "not built" in transcript(monitor)


# --- /demo -----------------------------------------------------------------------


def test_demo_queues_the_shipped_recordings_and_says_what_they_are(
    tmp_path: Path,
) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/demo")
    assert len(monitor.queue) == 3
    shown = transcript(monitor)
    assert "recorded with the real pipeline" in shown
    *_, (_last, after) = monitor.queue
    assert any("that was a recording" in line for line in after)
    assert any("/index <folder>" in line for line in after)
    first_after = monitor.queue[0][1]
    assert " 1 " in first_after[1], "passages are numbered for /show"


def test_every_demo_recording_plays_through_to_the_transcript(tmp_path: Path) -> None:
    from spiyweb.commands import demo_records

    records = demo_records()
    assert all(record.index == "demo" for record in records), "no scratch path"
    monitor = quiet_monitor(tmp_path)
    monitor.config = WatchConfig(hop_delay_ms=0)
    dispatch(monitor, "/demo")
    now = 0.0
    for _ in range(400):
        now += 0.1
        monitor.tick(now)
        if not monitor.queue and monitor.playing is None:
            break
    assert len(monitor.played) == 3
    shown = transcript(monitor)
    assert "Wardenclyffe" in shown and "that was a recording" in shown


# --- /show, /save ------------------------------------------------------------------


def _played(tmp_path: Path) -> Monitor:
    from dataclasses import replace

    record = canonical_record()
    record = replace(
        record,
        nodes=tuple(replace(n, text=f"text of {n.id}") for n in record.nodes),
    )
    monitor = quiet_monitor(tmp_path)
    monitor.played.append(record)
    return monitor


def test_show_with_nothing_played_says_so(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/show 1")
    assert "nothing has been asked yet" in transcript(monitor)


def test_show_n_opens_that_passage(tmp_path: Path) -> None:
    monitor = _played(tmp_path)
    dispatch(monitor, "/show 3")
    shown = transcript(monitor)
    assert "text of D" in shown and "question → A → D" in shown
    dispatch(monitor, "/show 9")
    assert "there are 5 passages" in transcript(monitor)


def test_show_alone_lists_the_passages_and_stays_open(tmp_path: Path) -> None:
    monitor = _played(tmp_path)
    dispatch(monitor, "/show")
    picker = monitor.picker
    assert picker is not None and picker.sticky and len(picker.options) == 5
    monitor.handle_key("down", 0.0)
    monitor.handle_key("enter", 0.0)
    assert "text of C" in transcript(monitor)
    assert monitor.picker is not None, "open another, or esc"


def test_save_writes_the_query_as_markdown_here(tmp_path: Path) -> None:
    monitor = _played(tmp_path)
    dispatch(monitor, "/save notes/answer.md")
    page = (tmp_path / "notes" / "answer.md").read_text(encoding="utf-8")
    assert page.startswith("# which tower did tesla build")
    assert "saved notes/answer.md" in transcript(monitor)
    dispatch(monitor, "/save")
    assert list(tmp_path.glob("spiyweb-*.md")), "a dated name by default"


def test_a_query_that_never_left_the_seeds_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusal:
        text = "Only the seeds activated.\nMissing: a source that connects A with C."

    class StuckAnswer:
        trace = canonical_record(hops_used=0)
        passages = ()

        def refusal(self) -> Refusal:
            return Refusal()

    class StuckIndex:
        def retrieve(self, question: str, *, profile: str) -> StuckAnswer:
            return StuckAnswer()

    monkeypatch.setattr("spiyweb.cli._open", lambda path, **o: StuckIndex())
    monitor = quiet_monitor(tmp_path)
    from spiyweb.commands import run_query

    run_query(monitor, str(tmp_path), "anything")
    _record, after = monitor.queue[0]
    assert any("Missing: a source" in line for line in after)


# --- /compare, /tune, warm follow-ups ------------------------------------------------


def _texted_record() -> object:
    from dataclasses import replace

    record = canonical_record()
    return replace(
        record, nodes=tuple(replace(n, text=f"text of {n.id}") for n in record.nodes)
    )


@dataclass(frozen=True)
class SpyAnswer:
    """Shaped like `Answer` - a frozen dataclass, so `replace` works on it."""

    trace: object = field(default_factory=_texted_record)
    passages: tuple[object, ...] = ()
    result: object = field(
        default_factory=lambda: type("R", (), {"propagation": object()})()
    )


class SpyIndex:
    """Records how it was asked; answers with the canonical web."""

    def __init__(self) -> None:
        self.asked: list[dict[str, object]] = []
        self.store = type(
            "Store",
            (),
            {
                "search": lambda self, vector, k: [
                    (n, 1.0 - i / 10) for i, n in enumerate(["A", "C", "X", "Y", "Z"])
                ][:k]
            },
        )()

    def retrieve(self, question: str, **options: object) -> SpyAnswer:
        self.asked.append(options)
        return SpyAnswer()

    def close(self) -> None:
        pass


@pytest.fixture
def spy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SpyIndex:
    index = SpyIndex()
    monkeypatch.setattr("spiyweb.cli._open", lambda path, **o: index)
    (tmp_path / "idx").mkdir()
    (tmp_path / "idx" / "nodes.json").write_text("[]", encoding="utf-8")
    return index


def test_compare_shows_what_only_the_web_found_at_the_same_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy: SpyIndex
) -> None:
    class Embedder:
        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return [[1.0]]

    monkeypatch.setattr(
        "spiyweb.commands.shared_embedder", lambda monitor, key: Embedder()
    )
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/compare which tower did tesla build")
    _record, after = monitor.queue[0]
    shown = "\n".join(after)
    assert "the web against plain top-k  (k=5" in shown
    assert "both found       2/5" in shown
    assert "only the web     3" in shown and "text of D" in shown
    assert "only top-k       3" in shown and "X, Y, Z" in shown


def test_compare_needs_the_model_the_monitor_holds(
    tmp_path: Path, spy: SpyIndex
) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/compare anything")
    assert "needs an index built with the default model" in transcript(monitor)
    assert spy.asked == [], "nothing was asked half-way"


def test_tune_shows_the_knobs_then_changes_them_and_asks_again(
    tmp_path: Path, spy: SpyIndex
) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/tune")
    assert "profile explore" in transcript(monitor) and "damping 0.75" in transcript(
        monitor
    )
    dispatch(monitor, "who built it")
    assert spy.asked[-1] == {"profile": "explore"}
    dispatch(monitor, "/tune damping 0.5 seed 3")
    config = spy.asked[-1]["config"]
    assert "profile" not in spy.asked[-1], "a supplied config is never overlaid"
    assert config.seed_width == 3 and config.propagation.damping == 0.5  # type: ignore[attr-defined]
    record, _ = monitor.queue[-1]
    assert record.profile == "tuned", "the picture says it ran tuned"
    assert "tuned" in monitor.status_bar(120)
    assert monitor.settings.profile == "explore", "nothing was saved"
    dispatch(monitor, "/tune reset")
    assert spy.asked[-1] == {"profile": "explore"}


@pytest.mark.parametrize(
    ("line", "said"),
    [
        ("/tune damping 1.5", "damping must lie strictly between 0 and 1"),
        ("/tune speed 3", "no knob 'speed'"),
        ("/tune damping", "pairs of name and value"),
        ("/tune seed many", "'many' is not a number"),
    ],
)
def test_tune_refuses_what_a_profile_cannot_be(
    tmp_path: Path, line: str, said: str
) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, line)
    assert said in transcript(monitor) and monitor.tuned is None


def test_warm_follow_ups_carry_residue_until_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy: SpyIndex
) -> None:
    monkeypatch.setattr(
        "spiyweb.thermal.residue_of", lambda previous, thermal: {"A": 1.0}
    )
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "first question")
    assert "residue" not in spy.asked[-1], "warm is off by default"
    monitor.settings.warm = True
    dispatch(monitor, "first question")
    assert "residue" not in spy.asked[-1], "the first warm question starts cold"
    dispatch(monitor, "a follow-up")
    assert spy.asked[-1]["residue"] == {"A": 1.0}
    assert "warm" in monitor.status_bar(120)
    dispatch(monitor, "/reset")
    dispatch(monitor, "after the reset")
    assert "residue" not in spy.asked[-1]


def test_a_warm_follow_up_that_stayed_on_warm_ground_is_not_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WarmAnswer:
        trace = canonical_record(
            hops_used=0, injected_energy=15.5, settings={"seed_energy": 10.0}
        )
        passages = ()

        def refusal(self) -> object:
            raise AssertionError("no refusal for residue that covered the web")

    class WarmIndex:
        def retrieve(self, question: str, **options: object) -> WarmAnswer:
            return WarmAnswer()

    monkeypatch.setattr("spiyweb.cli._open", lambda path, **o: WarmIndex())
    monitor = quiet_monitor(tmp_path)
    from spiyweb.commands import run_query

    run_query(monitor, str(tmp_path), "a follow-up")
    assert len(monitor.queue) == 1


def test_the_index_form_glob_can_be_cleared_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor = quiet_monitor(tmp_path)
    started: list[list[str]] = []
    monkeypatch.setattr(
        monitor, "spawn_spiyweb", lambda args, **o: started.append(list(args))
    )
    dispatch(monitor, "/index notes")
    keys = {v: n for n, (v, _) in enumerate(monitor.picker.options)}  # type: ignore[union-attr]

    def choose(value: str, answer: str | None = None) -> None:
        monitor.picker.cursor = keys[value]  # type: ignore[union-attr]
        monitor.handle_key("enter", 0.0)
        if answer is not None:
            monitor.buffer = answer
            monitor._submit()

    choose("glob", "**/*.md")
    choose("glob", "")
    choose("build")
    assert "--glob" not in started[0], "an empty answer means every file again"


def test_tune_and_compare_ask_cold_even_when_follow_ups_are_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy: SpyIndex
) -> None:
    monkeypatch.setattr("spiyweb.thermal.residue_of", lambda p, t: {"A": 1.0})
    monitor = quiet_monitor(tmp_path)
    monitor.settings.warm = True
    dispatch(monitor, "first")
    dispatch(monitor, "/tune seed 2")
    assert "residue" not in spy.asked[-1], "a knob is judged on cold ground"


def test_a_rebuilt_index_forgets_the_warm_ground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy: SpyIndex
) -> None:
    import os

    monitor = quiet_monitor(tmp_path)
    monitor.settings.warm = True
    dispatch(monitor, "first")
    assert monitor.warm_from is not None
    meta = tmp_path / "idx" / "nodes.json"
    later = meta.stat().st_mtime_ns + 5_000_000_000
    os.utime(meta, ns=(later, later))
    from spiyweb.commands import open_cached

    open_cached(monitor, str(tmp_path / "idx"))
    assert monitor.warm_from is None, "old node ids never warm a new graph"
