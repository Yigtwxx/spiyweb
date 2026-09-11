# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Public API policy

`import spiyweb` is the **query-time** contract: everything in
`spiyweb.__all__`, and nothing else. `spiyweb.indexing` is the **index-time**
contract, on the same terms. Anything reachable only through a submodule path
— `spiyweb.core.*`, `spiyweb.evaluation.*`, `spiyweb.config.*`, `server.*`,
`ui/` — is internal and may change in any release without a note here.

Two rules hold from the first tagged release onwards:

1. Removing or renaming a declared name is a MINOR bump while the project is
   below 1.0, and it lands here under **Removed** with the replacement path.
2. A change that moves a measured retrieval number is never a silent one. It
   lands under **Changed** with the measurement that justifies it, because a
   benchmark comparison across two versions is only honest when the diff
   between them is written down.

`spiyweb.evaluation` is the measurement harness, not the library. It is the
regression suite, and its CLI flags are not covered by the policy above.

## [Unreleased]

### Changed

- `/find` prints one row per file (import count, first line) instead of
  one per import, and skips spiyweb's own package tree when run inside a
  checkout of it.
- Mouse capture is **off by default** so the terminal's own selection
  and copy keep working; `/config` turns it on (wheel scrolls, click
  picks, copy with shift+drag). Page Up / Page Down scroll the transcript
  either way.

## [0.2.0] - 2026-09-11

The terminal is the interface. `pip install spiyweb` (or `uv tool install
spiyweb`), open a terminal in the project folder, type `spiyweb`: the
window becomes the monitor, and every query the application in the next
terminal - or one started with `! python app.py` - makes is played hop by
hop. MINOR bump: bare `spiyweb` changed behaviour and the browser face,
`spiyweb.viewer`, `inspect_url()`, `spiyweb view` and the `[web]` extra
are gone (see Removed).

### Changed

- **Bare `spiyweb` is a screen, not a menu.** Typing `spiyweb` in a
  terminal used to ask four questions; it now takes the window over the way
  `claude` does - screen and scrollback wiped, a welcome box with the
  spider, a transcript, a status line, a boxed `/` prompt and a status bar -
  and asks nothing. Every question the menu asked is a command (`/query`,
  `/lint`, `/index`, `/install`). The menu is still there behind `/menu`
  and `spiyweb menu`, and it is what runs where a terminal cannot read
  single keys or redraw a screen; a pipe still gets the usage text and
  exit 2, so nothing scripted changes.
- **Records reach the disk on a monitor's request too.** `TraceStore`
  appended to a file only when `TraceConfig.directory` was set. It now also
  appends to `<attach_dir>/traces.jsonl` while a `spiyweb` monitor's marker
  in that folder is fresh (`TraceConfig.attach_dir`, default `.spiyweb`;
  `None` switches it off - the production setting). Passage text lands
  there, in a folder that ignores itself from git. Each record is one
  binary append, on both paths.
- `hop_ring_layout`, `ring_radii` and `layout_seed` moved out of
  `spiyweb.scene` into `spiyweb.rings`, numpy-free; `scene` re-exports them
  and the layout is unchanged in every measured property (the random start
  angle now comes from `random.Random` instead of numpy's generator).
- `spiyweb.terminal.bar` takes an optional block ramp, for ASCII consoles.

- **The library default can spread.** 0.1.2 moved the terminal and the
  viewer onto the `explore` profile and left `SpiywebIndex.retrieve()` at a
  warning, so a first-time caller of the LIBRARY still got first contact
  only - top-k with extra steps, silently unless they read stderr. Now
  `SpiywebIndex.retrieve()` and `ThermalSession` apply `DEFAULT_PROFILE`
  (`explore`) whenever the caller expresses no preference: no `profile`,
  and no `config` at the call or at `open()`. A config the caller built is
  never overlaid - it runs as given, and `retrieve()` still warns if it
  cannot spread. `retrieve_colored()` is unchanged: `ColoredRetrievalConfig()`
  is the measured operating point and clears the bar on every colour.

  **No measured number moves.** `RetrievalConfig()` and `PropagationConfig()`
  keep the canonical trace of CLAUDE.md §2.6; the evaluation harness calls
  the `retrieve()` primitive with its own configs and never passes through
  `SpiywebIndex`. On the tiny four-passage test corpus the default now runs
  at damping .75 / threshold .01 / width 8 instead of .60 / .15 / 5, which is
  why the two ledger tests rebuild their expectation from the config the
  answer reports rather than from `RetrievalConfig()`.

### Added

- **The live monitor** (`spiyweb`, `spiyweb watch [--dir]`): open a terminal
  in the project folder while the application runs in another, and every
  query the application makes is played here as it happens - the ranking
  bars on the left growing in hop by hop, the ring map on the right, the
  ledger line under both. The two processes meet through a marker file and
  a tailed JSONL; no socket, no server, no code in the application. Only
  closing the terminal ends it (ctrl-c twice inside a second is the
  emergency exit). New stdlib-only modules: `keys` (raw keys with a
  timeout), `rings`, `pet`, `animate` (pure frames from a `TraceRecord`),
  `watch` (marker, tail, loop, screen) and `commands`.
- `/` commands: `/help`, `/find` (files that `import spiyweb` and the index
  they `open_index(...)`, which becomes the default for the next two),
  `/query [index] <question>` (played from the answer's own record, never
  through the file), `/lint`, `/index`, `/install [extra]` (the extras
  table, or `pip`/`uv pip` into this interpreter after a y/n), `/replay [n]`,
  `/config` (an arrow-key list: profile, ring map, spider, hop delay,
  glyphs, colour - saved to `.spiyweb/monitor.json`), `/clear`, `/menu`,
  `/version`. Text without a slash is a question for the current index.
  A CLI error that says `pip install "spiyweb[x]"` says `/install x` here.
- `! <command>` runs a shell command from the monitor, in the background,
  its output streaming into the transcript - the way to start the project
  from the same window and watch it; `/jobs` lists them, `/kill [n]` stops
  one, and every job ends with the monitor. The child gets no stdin.
- The input line has a cursor: left/right move inside it, home/end and
  ctrl-a/ctrl-e jump, delete removes under it, typing inserts there; the
  caret sits on the character it covers.
- Mouse: the wheel scrolls the transcript, a left click chooses a list
  item (`/config`, an index to pick) or a command suggestion under the
  prompt. SGR mouse reporting, VT input on the Windows console, both
  switched off again on exit.
- `TraceConfig.attach_dir` / `attach_stale_s`, `spiyweb.config.WatchConfig`
  (the monitor's knobs; not in `spiyweb.__all__`), `spiyweb.trace.WATCH_MARKER`,
  `attached_trace_path`, `ensure_private_dir`.
- `spiyweb.terminal`: `HIDE_CURSOR`/`SHOW_CURSOR`, `CLEAR_SCREEN`, `HOME`,
  `ERASE_LINE`, `cursor_to`, `clip`, `pad`, `printed_width`,
  `supports_screen`, `supports_unicode`, `terminal_size`.
- `spiyweb.wizard.discover(cwd=...)`: where "nearby" starts.

- `spiyweb.DEFAULT_PROFILE`: the one source of the name the terminal, the
  viewer and the library all fall back to. The two `"explore"` literals in
  `cli.py` and `viewer/app.py` are gone.
- `Answer.profile` and `Answer.config`: which profile ran and the resolved
  configuration it produced, so a caller never has to infer either. The
  trace record's `profile` field now carries the resolved name instead of
  `""` when the default applied.
- `ThermalSession(profile=...)`, with the same overlay rule and the same
  default as `SpiywebIndex.retrieve()`; `.profile` and `.config` report what
  the session runs with.

### Fixed

- **The wizard froze on a bare ESC on macOS and Linux.** The POSIX key reader
  did a blocking `read(2)` after every ESC byte to see whether an arrow
  followed, and on a bare ESC - the key the menu documents as "leave" -
  nothing follows, so the menu waited for two keystrokes that never came.
  The reader now polls with a 50 ms `select` before reading, and the decoder
  is a pure function tested on every CI leg, including `ESC O A`
  application-mode arrows. The Windows path never had the problem.
- **The measurement rig's child was shielded from Ctrl-C on Windows only.**
  `server/runner.py` gave the run its own process group with
  `CREATE_NEW_PROCESS_GROUP` and nothing at all on POSIX, where a Ctrl-C at
  the server's terminal reached the whole foreground group and killed the
  run the comment said it was protecting. One helper now emits
  `start_new_session=True` on POSIX and the flag on Windows; both launch
  sites use it.
- **`spiyweb index` skipped `README.TXT` on macOS and Linux.** The directory
  walk did one `rglob` per suffix, and `rglob("*.txt")` is case-insensitive
  on Windows only. The suffix is now tested on the lowercased name, and the
  documents are ordered by their posix relative path - the string that
  becomes the source id - instead of by `Path` comparison, which is
  case-insensitive on Windows and byte-wise elsewhere. Two machines indexing
  the same directory now produce the same `nodes.json` in the same order.
  The wizard's index menu and the rig's index list sort case-folded for the
  same reason.
### Removed

- **The browser face, entirely.** `spiyweb.viewer` (the FastAPI page,
  `inspect_url()`, `serve_index` / `serve_file`, the loopback-token
  serving), the `spiyweb view` verb and the wizard's "Open the viewer"
  entry, the `[web]` extra, the React/Vite front end under `web/`, the
  `server/` measurement rig (routes, SSE, the run supervisor) and the CI jobs
  that built and packed the bundle. The owner has a different interface in
  mind, and a page nobody will keep is a page that rots in the wheel.

  A MINOR-bump removal by the policy at the top of this file, so the next
  version is **0.2.0**. `SpiywebIndex.inspect_url()` is gone with no
  replacement path; every other public name is untouched, and the
  zero-dependency wheel is now checked against five heavy modules instead
  of seven, because pydantic and FastAPI no longer have a way in.

  What stayed is what any interface needs and none of this depended on: the
  trace layer (`TraceRecord`, `load_traces`, JSONL on disk), the energy
  ledger and the render-agnostic `spiyweb.scene` (`[view]`, numpy only) -
  including the 2026-09-03 canvas fixes that moved the ring-spacing rule
  into `scene.ring_radii()`, since the rule belongs to the scene and not to
  whatever draws it.
- The `dev` dependency group dropped `httpx`, which only the viewer's test
  client used.

### Internal

- `.gitattributes` pins LF for every text file. The tree has always been
  LF; this stops a Windows clone without `core.autocrlf` from committing
  CRLF and failing `ruff format --check` on the Linux and macOS legs. Ruff's
  `line-ending` stays `auto`, because a checkout under `autocrlf=true` is
  CRLF on disk and `lf` would fail that same check locally.

## [0.1.2] - 2026-08-26

### Changed

- **`spiyweb query` and the viewer's ASK box run the `explore` profile by
  default.** The bare `RetrievalConfig()` splits 10.0 energy among 5 seeds
  and stops at 15% of it, so the strongest seed forwards at most 1.23 against
  a threshold of 1.50 - **nothing can ever clear it**, and the web returns
  first contact only. On a real index that is 5 atoms at hop 0: `top-k` with
  extra steps, which is the one thing this project exists to not be. With the
  profile the same query returns 34 atoms at depth 2.

  The root cause is two written decisions that do not compose: CLAUDE.md sets
  the threshold at 15% of the injected energy (§2.1, sized for the TWO-seed
  worked example) and the seed width at 5 (§4). **The library default is
  unchanged** - it carries the canonical trace of §2.6 and no measured number
  may move - so the fix is in the terminal and the page, where a person first
  meets the mechanism. Both print which profile they used.

### Added

- `retrieve()` warns when the settings cannot spread past the seed, naming
  the arithmetic and the fix. A warning and not an error: first contact only
  is a legitimate thing to want. But it must not be silent, for the same
  reason the dedup trap now raises - a mechanism that is off while the caller
  believes it is on is how this project lost a measurement campaign.

### Fixed

- `retrieve_colored()` checked the spread of whichever colour the contact
  loop happened to leave in a local variable - the wrong colour as often as
  not, and stale entirely when a colour was skipped. Every colour is checked
  now, since each injects the full seed energy among its own contacts.

## [0.1.1] - 2026-08-26

### Fixed

- **The viewer offered a search box it could not honour.** `capabilities.live`
  meant "an index is attached", not "a question can be embedded", so on an
  install without the `embed` extra the page showed an ASK box and the button
  returned an opaque `500 Internal Server Error`. Found within minutes of the
  0.1.0 release, by opening the link the CLI printed.

  `live` now means a query can actually run - `SpiywebIndex.can_query` checks
  for an embedder without loading one - so the box is simply absent, and the
  page says why and prints the `pip install "spiyweb[embed]"` that fixes it.
  If the route is reached anyway it answers `503` with that same line instead
  of a 500.

## [0.1.2] - 2026-08-26, Phase 2 close-out

Written before the release-day fixes above and left under an `Unreleased`
heading when the three 0.1.x versions were published from one commit
(`b2d3fbd`); every entry below is in the 0.1.2 wheel.

### Added

- **Corpus lint** (`spiyweb.lint_corpus`, `spiyweb.indexing.lint_index`,
  `spiyweb lint <index>`): the diagnostic that needs no query. Every other
  honesty output here explains why ONE retrieval went badly; this one asks
  whether the corpus is shaped so that retrieval can work at all, and answers
  it from the graph's topology alone.

  Four findings, each tied to something this project already claims:
  **orphans** (components nothing bridges - multi-hop cannot help an island),
  **hubs** (known-risk #2 of CLAUDE.md §8, measured as the share the
  strongest neighbour actually receives under the propagation's own
  `split_alpha`, not as a degree), **duplicates** (near-identical passages in
  the RAW cosine layer, plus the within-source count that D7 says can never
  become a vote), and the **contradiction map** (NLI edges aggregated per
  source, because a document that disagrees twenty times is one problem).

  Pure and dependency-free like `output.py`; the prose is template-built with
  no LLM anywhere near it. Reporting thresholds live in `CorpusLintConfig` and
  change what is worth mentioning, never what retrieval does.
- **A bare `spiyweb` asks instead of printing usage.** One word to type, then
  questions: what to do, which index (found for you - the path is the part
  people get wrong), what to ask, how the web should spread. Each option
  carries a grey parenthetical note saying what it does, and the menu prints
  the command it built, so the second time you type it yourself and the menu
  has made itself unnecessary.

  Arrow keys move a filled dot down the list where the terminal can read a
  single keypress (`msvcrt` on Windows, `termios` on POSIX - both stdlib, so
  the zero-dependency promise is untouched); numbers still work, because
  every other menu has trained that muscle. The marker changes SHAPE and not
  only colour, so it survives a terminal that refused colour - which is
  exactly the terminal the colourless path runs in. The numbered prompt is
  not a lesser fallback that rots: it is the same function, it is what runs
  wherever raw input is unavailable, and every test of the menu goes through
  it.

  It only asks when someone is there to answer: stdin and stdout must both
  be a terminal. A pipe, a script or a CI job gets usage and exit 2 - a
  prompt with nobody at it is a hung build.
- **The terminal command draws instead of listing.** `spiyweb query` renders
  each activated atom's energy as a bar and its hop distance as a colour, so
  the decay the project is about is visible rather than something a reader
  divides in their head; `spiyweb lint` colours structural findings apart
  from dense ones; `spiyweb version` marks what is installed.

  The palette is two families and the split carries meaning rather than
  taste: **blue is the web working** (headings, hop depth, what is
  installed), **red is something wrong** (an empty layer, an island, a
  missing extra). A report can be judged before it is read, and a test
  asserts the families never share a colour.

  Hand-written ANSI in `spiyweb.terminal`, not `rich`: `dependencies = []` is
  measured on the wheel, and a prettier CLI is not worth trading it for.
  Three guarantees are tested rather than trusted - **no colour when stdout
  is not a terminal** (so `--json | jq` and `> file` stay clean),
  **`NO_COLOR` is honoured**, and Windows gets its virtual-terminal mode
  enabled rather than printing escape codes literally.
- A fifth lint finding, **empty layers**: a layer merged at a non-zero weight
  that carries no edges at all. Two of these turned up in one day and both had
  gone unnoticed for months - the structural layer is empty in all four sealed
  Phase 1 indexes, and the entity layer was empty on every corpus under a
  hundred chunks. The merged adjacency cannot tell an empty layer from an
  absent one, so nothing after the merge could ever have said so.

  Where the cause is knowable it is named rather than left as an accusation.
  On the sealed indexes the report now reads: *"the structural layer carries
  no edges at all, yet it is merged at weight 0.3 ... Every document here is a
  single chunk, so the layer's within-document relations cannot exist."* That
  resolves a Phase 1 side finding: the structural layer was never broken, and
  `LayerWeights.structural = 0.3` was simply inert in every measured run.

### Changed

- The entity edge builder floors its document-frequency ceiling at 2:
  `max(2, max_df_ratio * n_chunks)`. An entity has to appear in two chunks to
  pair at all, so at the measured default ratio of `0.02` the ceiling fell
  below 2 for **every corpus under 100 chunks** and the entity layer - the
  main hop fuel of CLAUDE.md §2.2 - came out empty without saying so.

  Found by running the real pipeline over three documents on 2026-08-26. The
  one entity that bridged them (`morgan`, df=2) was dropped against a ceiling
  of 0.16; with the floor the same corpus gains the cross-document edge
  `morgan.md:1 - wardenclyffe.md:2` at weight 0.5, which is precisely the
  mechanism this project exists for.

  **No sealed number can move.** The floor binds only where `0.02 * n` falls
  under 2, i.e. below 100 chunks; the smallest measured index holds 3336,
  where the ceiling is 66.7. A test pins both directions.

- `spiyweb.evaluation.contradiction`: a sensitivity harness for the
  contradiction detector, and the bound it can honestly report. Phase 1
  measured 0% on WikiContradict's 63 same-passage pairs and wrote the reason
  down as "the proposition layer is required". That clause is an inference,
  and this module tests it by splitting passages into sentences - the free
  lower bound on what any proposition extractor can deliver, since extractors
  work from sentences.

  **Measured 2026-08-26: the dataset cannot answer the question.** 21 of the
  63 (33.3%) are a single sentence, where no extractor can separate the two
  claims; 30 (47.6%) answer yes/no, which is a reasoning output rather than a
  span and cannot be traced to a sentence at all. Seven remain traceable, and
  a rate over seven cases is not a number - so the harness reports the
  breakdown and refuses to compute one. "The proposition layer is required"
  stays true as a necessary condition and is **not sufficient**: at least a
  third of that bucket is unreachable by design.

### Fixed

- **The release build shipped no browser bundle.** `uv build` makes the sdist
  first and builds the wheel *from it*, so naming the compiled bundle only
  under the wheel target left it out of both: the two-target build produced a
  229 KB wheel with no page while `uv build --wheel` produced the right 1.4 MB
  one. A release uses the first command, so `inspect_url()` would have been
  published permanently broken.

  The `artifacts` key moved to `[tool.hatch.build]`, which covers every
  target, and CI now builds the release way and asserts the bundle is in the
  **sdist** as well as the wheel. Found on 2026-08-26 while preparing to
  publish, before any upload.
- `spiyweb` writes progress to stderr, not stdout. `spiyweb lint --json | jq`
  received `linting ...` as the first line of what was supposed to be JSON.
- `spiyweb` output survives a console that cannot encode the corpus. Python
  encodes stdout with the console codepage - cp1254 on a Turkish Windows
  machine - and this project indexes Turkish and English by design, so a
  passage with a macron in it raised `UnicodeEncodeError` out of `print`.
  Found while hand-verifying a lint finding on the real MuSiQue index.
- `spiyweb view` flushes the link it prints before it blocks on the server.
  Piped - a script, a notebook, a log - the URL sat in the buffer forever and
  the command looked hung.
- `load_traces` tolerates a truncated LAST line and refuses a damaged one
  anywhere else. The writer is usually still appending, so a half-written
  final record is the normal case; a broken line in the middle is data loss
  and now names its line number instead of vanishing.
- `SpiywebIndex.inspect_url()` is safe to call from several threads. Two
  callers racing used to start two servers and drop one handle - a listening
  socket nobody could close, for the life of the process.

### Internal

- The measurement rig and the shipped viewer now share one serialisation
  (`spiyweb.viewer.payload`) instead of each keeping a copy of the same
  ninety lines. `server/inspect_api.py` lost 76 lines; a test builds the
  rig's pydantic models straight from the package's dicts, so a field
  renamed on one side fails there rather than quietly meaning something else
  on one of the two pages.

## [0.1.0] - 2026-08-26

The first version meant to be installed by somebody else. `0.0.1` said "there
is nothing here yet"; there is now a declared API of 96 names, a version
contract, a terminal command and a browser face that runs from the installed
package.

### Added

- **A terminal command.** `pip install spiyweb` puts `spiyweb` on the path:

  ```
  spiyweb version                       what is installed, and what is not
  spiyweb index docs/ my-index          a directory of text files -> an index
  spiyweb query my-index "a question"   the activated web, as text
  spiyweb view my-index                 the browser face, on a link
  ```

  Every verb wraps something the Python API already does - a CLI that grows
  behaviour the library lacks becomes a second product to keep correct.
  `spiyweb version` works on a bare install with no extras, because that is
  exactly when someone needs to be told which extra is missing; it prints the
  `pip install` line that fixes it.
- `docs/releasing.md`: the checklist that makes an upload boring, including
  the one step easy to get wrong - the browser bundle has to be built BEFORE
  the wheel, or `inspect_url()` ships with no page to serve.

- **`SpiywebIndex.inspect_url()`** and `spiyweb.viewer`: the browser face
  now ships in the wheel. Two lines in an application - open an index, print
  the link - start a loopback-only server on an OS-chosen port, behind an
  unguessable token, on a daemon thread that never takes the calling process
  over. `spiyweb.viewer.serve_file("traces.jsonl")` does the same for a file
  an application wrote earlier, on a machine that holds no index at all.
  The page draws recorded calls through the SAME `spiyweb.scene` builder the
  live inspector uses, so the two cannot show different pictures of one
  mechanism.
  `dependencies = []` is untouched: nothing here is imported eagerly, and a
  missing extra says `pip install "spiyweb[web]"` in a sentence.
- `spiyweb.Ledger` / `spiyweb.build_ledger` / `spiyweb.Destroyed` /
  `spiyweb.destroyed_per_node`: the energy ledger of CLAUDE.md §2.1 -
  injected = held + dissipated + destroyed, with the reconstruction's own
  disagreement reported rather than rounded away. It lived in the
  repository's `server/` and could not be reached by anyone who installed
  the package; it is pure arithmetic over a propagation result, so it is
  library code and now says so.
- `TraceRecord.ledger`: every trace carries its own energy ledger, computed
  at record time. The reconstruction needs the graph and the propagation
  config, and a reader holding those would be holding the index - the thing
  a trace exists to avoid.

- `spiyweb.TraceStore` / `spiyweb.TraceRecord` / `spiyweb.TraceConfig` /
  `spiyweb.load_traces`: every `SpiywebIndex` query now leaves a record of
  what it did, and `index.traces` holds the last 200 of them. A record is
  **self-contained** - the activated subgraph's edges, the passages' text,
  the seeds, the votes, the paths, the theme clusters, the suppression cuts
  and the destroyed-energy events - so whatever reads it never loads the
  graph and the vector store a second time. That is the difference between
  showing what an application actually retrieved and re-running a query it
  once ran.
  Tracing is on in memory and costs no disk: passage text lands in the file,
  so writing one is an explicit `TraceConfig(directory=...)`. The whole
  layer switches off with `TraceConfig(enabled=False)`, and
  `spiyweb.trace` imports with no dependency installed - `load_traces()`
  reads a file back with numpy and FAISS absent, which is the check behind
  the "no second copy of the index" claim rather than the claim itself.

- `spiyweb.indexing.build_index(documents, out_dir, ...)`: the index
  pipeline, corpus-agnostic at last. It used to demand a `MusiqueDataset`
  and an `IndexPaths` full of benchmark file names, so "index my own
  corpus" meant faking a benchmark. `spiyweb.evaluation.index` is now a
  thin adapter over it - the sealed measurement runs and a user's own
  corpus go through one implementation instead of two that drift.
  `IndexLayout` holds the artifact names, `IndexPaths` extends it with the
  benchmark's own, and `build_index` returns an `IndexManifest`.
- `spiyweb.SpiywebIndex` / `spiyweb.open_index`: an opened index you can
  query. It loads the graph, the vector store, the texts and the
  similarity backend together and returns `Passage` objects carrying their
  own text - `result.ranked()` handed back node ids, and the text lived
  somewhere the caller had to keep themselves. It also wires BOTH halves
  of duplicate suppression, so the trap closed under **Changed** cannot be
  walked into from here at all. Deliberately no `k`: the web stops on its
  own energy, and a cutoff parameter would reintroduce the thing this
  project argues against.
- `texts.json`: an index now records the text each chunk was indexed as.
  Absent in indexes built earlier, and read as "no texts" rather than an
  error.
- `VectorStore` records the embedding model and a schema version, and
  `SpiywebIndex` refuses a query embedded by a different model. Two
  unrelated models can share a dimension, and cosine across two spaces
  returns confident nonsense rather than an error.
- `tests/test_index_golden.py`: the artifacts the pipeline writes, pinned
  before the split and unchanged after it. Its fixture uses two-unit
  documents on purpose, so the structural edge layer is non-empty - a
  shape none of the four benchmark corpora ever produced.
- `Stats/phase1_results.png` and the script that regenerates it. One figure
  with six panels replaces nine separate charts that lived outside the
  repository, where a reader could not see which of them disagreed. The
  folder is tracked now.
- `RESULTS.md`: the complete Phase 1 measurement record as a single table -
  gate, metric decomposition, per-hop breakdown, all five failed rescue
  rounds, every mechanism ablation, detection quality, and the limits. It
  replaces a folder of separate charts that lived outside the repository,
  where nobody could see which of them disagreed. No run was executed to
  produce it: groups A-D and H were recomputed from the sealed artifacts,
  the intervals come from the closing report.
- `spiyweb.scene`: the render-agnostic layout, scene and comparison code,
  promoted out of the repository-rooted Streamlit tool. numpy only, behind
  the new `spiyweb[view]` extra, never imported eagerly. `server/_paths.py`
  had written down the condition for this promotion in advance; both halves
  of it came true at once, so the `sys.path` adjustment it existed for is
  gone rather than grandfathered.

- `py.typed`: the package ships its type information, so `mypy` and `pyright`
  see the annotations that were always there instead of `Any`.
- `spiyweb.indexing`: the index-time facade — chunking, the edge builders, the
  embedder, entity extraction, the LLM clients and `VectorStore` in one
  namespace. Importable with nothing installed; only touching `VectorStore` or
  `build_semantic_edges_fast` asks for `spiyweb[store]`.
- `RetrievalResult.dedup_mode`, `ColoredRetrievalResult.dedup_mode` and a
  `dedup_mode` entry in the harness receipt (`results.json`): which
  duplicate rules actually ran — `"off"`, `"sources_only"`, `"cosine_only"` or
  `"full"`. See **Changed** for why this exists.
- `tests/test_public_api.py`, `tests/wheel_smoke.py`, and a CI job that builds
  the wheel and imports it in an environment with no extras installed. Until
  now nothing anywhere checked that `dependencies = []` was true.

### Changed

- `retrieve()` and `retrieve_colored()` refuse a `DedupConfig` no rule can
  run. Two rules suppress a duplicate and they need different things: the
  cosine twin test needs a `similarity` backend, the source test does not.
  With an enabled config and neither of them live, `dedup=` was a no-op the
  caller believed in — which is how the 2026-08 measurement campaign ran tour
  after tour with the mechanism silently off. That combination is now a
  `ValueError`; the source-rule-only combination warns and reports
  `dedup_mode="sources_only"`.

  **No sealed number is affected.** The evaluation harness and the inspector
  both pass `similarity` and `dedup` as a pair or pass neither - the
  `load_similarity` call in `evaluation/run.py` and the `make_similarity`
  call in `server/inspect_api.py` are each guarded by `dedup is not None`
  - so neither reaches the refused path. `distinct_sources=False` is
  constructed nowhere outside the tests.
- The version has a single source: `src/spiyweb/__init__.py`. `pyproject.toml`
  reads it through `dynamic = ["version"]` instead of keeping a second copy.

### Removed

- The Streamlit inspector (`ui/`) and the `spiyweb[ui]` extra. The browser
  face (`server/` + `web/`) does the same job against the same artifacts and
  the same `retrieve()`, and maintaining two front ends bought nothing once
  they already shared the scene code.
- With it, the scene's renderer layer - `SceneRenderer`, `VegaLiteRenderer`,
  `PlotlyRenderer`, `RENDERERS`, `get_renderer`, `RendererUnavailable`. Those
  imported Streamlit, and after the promotion they would have shipped that
  import inside the wheel with no caller left to use it. `build_vega_spec`
  stays: it is a pure scene-to-spec function, and its tests pin scene
  semantics that nothing else covers.

- `EvaluationConfig` and `IterativeBaselineConfig` from `spiyweb.__all__`
  (84 → 82 names). They configure the benchmark, not the library. They have
  not moved: `from spiyweb.config import EvaluationConfig` is unchanged and is
  the path every caller in this repository already used.

### Known gaps

- `propagate()` still accepts an enabled `DedupConfig` without a `similarity`
  backend and runs without dedup. `core/` takes arrays and a config and
  returns numbers; the guard lives one layer up in `retrieve()`, which is
  where the trap actually bit. Stated rather than left to be discovered.
- A `UserWarning` raised through `ThermalSession.retrieve` points at
  `thermal.py` rather than at the caller's line.
