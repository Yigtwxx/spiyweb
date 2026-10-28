# Releasing

The publishing decision is the owner's: a release is a `v*` tag the owner
pushes. `.github/workflows/release.yml` builds that tag and uploads it to PyPI
through Trusted Publishing, so no token exists anywhere. This file is the
checklist that makes the tag boring when it happens.

## The version has one source

`src/spiyweb/__init__.py`. `pyproject.toml` reads it through
`dynamic = ["version"]`; `tests/wheel_smoke.py` asserts on a freshly built
artifact that the installed metadata and `spiyweb.__version__` agree. A
drifted version number lies about which code produced a measurement, which is
why there is a test rather than a convention.

Bumping is one edit:

```python
__version__ = "0.1.0"
```

While below 1.0, SemVer's promise here is the one written at the top of
`CHANGELOG.md`: removing or renaming a declared name is a MINOR bump, and a
change that moves a measured retrieval number is never silent.

## Before the build

```bash
ruff check . && ruff format --check .
pytest
```

If the release changes what a query produces - the propagation, the
profiles, dedup, the trace format - re-record the `/demo` queries so the
monitor's first impression is the current pipeline's real output:

```bash
python examples/make_demo.py      # needs the index extras and the spaCy model
```

It rewrites `src/spiyweb/demo.jsonl`; commit that file with the change.
The wheel smoke checks it shipped and plays with nothing installed.

## Build and verify the artifact

Build **both** targets with a bare `uv build`, never `uv build --wheel`.

That is not a style preference. `uv build` makes the sdist first and then
builds the wheel **from that sdist**, which is the path a release takes. On
2026-08-26 the two commands produced different wheels, because a gitignored
file (the browser bundle of the time) was named only under the wheel target
and the sdist-built wheel lost it. Anything a future release needs to pack
that git does not track goes under `[tool.hatch.build]`, which covers both
targets, and CI builds the release way so the two can never disagree.

```bash
rm -rf dist && uv build
uvx twine check dist/*

uv venv .relenv
uv pip install --python .relenv/bin/python dist/*.whl
.relenv/bin/python tests/wheel_smoke.py
.relenv/bin/spiyweb version
```

Four things have to be true, and the smoke script asserts them:
`py.typed` shipped, the metadata version matches `__version__`, no optional
dependency reached the import graph, and the canonical trace of CLAUDE.md §2.6
still ranks `D` third.

`spiyweb version` is the check by hand: it proves the console script
was registered, and on an extras-free install every row should read
`not installed` — which is the correct answer, not a failure.

## Upload to PyPI

Bump `__version__`, commit, then tag that commit with the same number:

```bash
git tag v0.2.5 && git push origin v0.2.5
```

The `release` workflow refuses a tag whose version has no `CHANGELOG.md`
section or does not match the built wheel, publishes from the `pypi`
environment - which is also what lists the upload under the repository's
Deployments - and then creates the GitHub Release, with that CHANGELOG
section as its notes (`scripts/release_notes.py`) and the wheel and sdist
attached.

Trusted Publishing has to be registered once on the PyPI side: project
`spiyweb` → *Publishing* → *Add a new publisher* → GitHub, owner `Yigtwxx`,
repository `spiyweb`, workflow `release.yml`, environment `pypi`.

A manual run (*Actions* → *release* → *Run workflow*) rebuilds the current
`main` and skips files PyPI already has, so it is safe to start by hand.

**A version number burns on upload.** PyPI will not let `0.1.0` be replaced -
only yanked - so the rehearsal below is worth the twenty minutes it costs.

## Upload to TestPyPI

Needs an account on <https://test.pypi.org> and an API token. The token is
never committed and never pasted into a file in this repository.

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-...
```

Then verify from the other side, on a machine or container that has never seen
this checkout:

```bash
uv venv .testpypi
uv pip install --python .testpypi/bin/python \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  spiyweb

.testpypi/bin/spiyweb version
.testpypi/bin/python -c "import spiyweb; print(spiyweb.__version__, len(spiyweb.__all__))"
```

`--extra-index-url` is required and is not a shortcut: TestPyPI does not
mirror the real index, so the extras' dependencies (numpy, FAISS, FastAPI)
resolve from PyPI while `spiyweb` itself comes from TestPyPI.

A version number burns on upload. TestPyPI will refuse to accept `0.1.0`
twice, so a second rehearsal needs `0.1.1` — that is the cost of the rehearsal
and it is worth paying before the real index.
