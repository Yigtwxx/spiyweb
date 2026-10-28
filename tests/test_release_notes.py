"""scripts/release_notes.py: the CHANGELOG section a GitHub Release takes as
its notes, and the exit codes the release workflow gates on."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import spiyweb

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "release_notes.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_notes", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_notes = _load_script()

CHANGELOG = """# Changelog

## [Unreleased]

### Added
- something coming

## [0.2.4] - 2026-10-03

### Changed
- the propagation

### Fixed
- a bug

## [0.2.3] - 2026-10-03

## [0.2.2] - 2026-10-03

### Fixed
- the last section, at the end of the file
"""


def test_section_is_the_body_up_to_the_next_heading() -> None:
    body = release_notes.section("0.2.4", CHANGELOG)
    assert body == "### Changed\n- the propagation\n\n### Fixed\n- a bug\n"


def test_the_last_section_runs_to_the_end_of_the_file() -> None:
    body = release_notes.section("0.2.2", CHANGELOG)
    assert body == "### Fixed\n- the last section, at the end of the file\n"


def test_a_missing_version_has_no_section() -> None:
    assert release_notes.section("9.9.9", CHANGELOG) is None


def test_a_version_is_not_a_prefix_match() -> None:
    # "0.2" must not pick up the "0.2.4" heading.
    assert release_notes.section("0.2", CHANGELOG) is None


@pytest.mark.parametrize(
    ("version", "code"),
    [("v0.2.4", 0), ("0.2.4", 0), ("0.2.3", 1), ("9.9.9", 1)],
)
def test_main_gates_on_a_non_empty_section(
    version: str,
    code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    monkeypatch.setattr(release_notes, "CHANGELOG", changelog)
    assert release_notes.main(["release_notes.py", version]) == code


def test_main_without_a_version_is_a_usage_error() -> None:
    assert release_notes.main(["release_notes.py"]) == 2


def test_the_current_version_has_release_notes() -> None:
    # The release workflow stops a tag whose version has no section; this
    # catches the missing section on the bump commit instead of at the tag.
    text = release_notes.CHANGELOG.read_text(encoding="utf-8")
    assert release_notes.section(spiyweb.__version__, text)
