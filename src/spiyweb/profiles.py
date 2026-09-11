"""Query profiles (D13) - precise / explore / compare knob packages.

One global damping cannot serve a fact lookup and an exploratory sweep at
once: a precise question wants a small, fast-dying energy ball, an
exploratory one wants a large, slow-dying ball. A profile is exactly the
"damping + threshold + seed width" package the design names, and this module
is a FACTORY over the existing retrieval configs, not a parallel config
tree: `as_retrieval` / `as_colored` overlay the three knobs onto a base
config and preserve everything else it carries.

The caller picks the profile - never an LLM, and never anything inside
`core/`. The preset values below are MEASURED directions, not hand guesses
any more: see the 2026-08-16 note on the presets themselves (open question
#6, plain path measured, coloured path still to confirm).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from spiyweb.config import ColoredRetrievalConfig, RetrievalConfig


@dataclass(frozen=True)
class Profile:
    """One query profile: the D13 "damping + threshold + seed width" package.

    Applying a profile overrides EXACTLY these three knobs on a base config;
    every other field of the base (`split_alpha`, `max_hop`, `max_nodes`,
    `contact_overfetch`, the coloured chain settings, ...) survives
    untouched. Bounds mirror `PropagationConfig`.

    Attributes:
        name: Identifier of the profile (UI selectors key on it).
        damping: Fraction of its energy a node forwards; low = the ball dies
            near the seed, high = it rolls far.
        threshold_ratio: Relative stop threshold; high = early termination.
        seed_width: First-contact atoms. On the coloured path this is PER
            COLOUR - the measured winner uses 2 there, so a plain-path width
            applied via `as_colored` widens every colour's contact.
    """

    name: str
    damping: float
    threshold_ratio: float
    seed_width: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not 0.0 < self.damping < 1.0:
            raise ValueError("damping must lie strictly between 0 and 1")
        if not 0.0 <= self.threshold_ratio < 1.0:
            raise ValueError("threshold_ratio must lie in [0, 1)")
        if self.seed_width < 1:
            raise ValueError("seed_width must be at least 1")

    def as_retrieval(self, base: RetrievalConfig | None = None) -> RetrievalConfig:
        """The profile overlaid onto a plain retrieval config."""
        base = base if base is not None else RetrievalConfig()
        return replace(
            base,
            seed_width=self.seed_width,
            propagation=replace(
                base.propagation,
                damping=self.damping,
                threshold_ratio=self.threshold_ratio,
            ),
        )

    def as_colored(
        self, base: ColoredRetrievalConfig | None = None
    ) -> ColoredRetrievalConfig:
        """The profile overlaid onto a coloured retrieval config.

        The default base is `ColoredRetrievalConfig()`, which carries the
        measured 2026-08-14 grid winner - its `split_alpha`, chain settings
        and colour cap all survive the overlay. Only the three profile knobs
        change, and `seed_width` is per colour here (winner: 2).
        """
        base = base if base is not None else ColoredRetrievalConfig()
        return replace(
            base,
            seed_width=self.seed_width,
            propagation=replace(
                base.propagation,
                damping=self.damping,
                threshold_ratio=self.threshold_ratio,
            ),
        )


# All three thresholds are .01, and that equality is the measurement's
# verdict, not an oversight. The 2026-08-14 hand values (.25 / .05 / .10)
# were measured on MuSiQue, 1000 questions, plain path, on 2026-08-16 and
# REJECTED: at those ratios the activated set equalled `seed_width` in all
# three profiles - the web never left hop 0, so a "profile" was just top-k
# cut to a different width, and PRECISE additionally cost -.0341
# CI [-.0400, -.0283]. Threshold turned out to be the DECIDING knob and the
# measured winner's value is .01; the hand values sat 25x / 5x / 10x above
# it. Held at .01, the remaining two knobs do carry the profile idea:
# the precise direction (damping .45, width 3) moved .3100 -> .3497,
# +.0396 CI [+.0284, +.0510]; explore was neutral and compare -.0015.
# So the presets ship exactly the cells that were measured, and nothing is
# invented to keep the three thresholds distinct. What is NOT yet measured:
# the same overlay on the COLOURED path, where a profile changes which
# passages get retrieved and therefore misses the intermediate-answer cache
# (~30-50 min per cell). Open question #6 stays open on that half.

PRECISE = Profile(name="precise", damping=0.45, threshold_ratio=0.01, seed_width=3)
"""Small, fast-dying ball: forwards little and touches few atoms.

The measured winner among the three directions on MuSiQue's plain path
(+.0396 over the base regime), which is worth noticing: MuSiQue is 100%
multi-hop and the narrow ball still won - consistent with the coloured
winner's per-colour `seed_width=2`.
"""

EXPLORE = Profile(name="explore", damping=0.75, threshold_ratio=0.01, seed_width=8)
"""Large, slow-dying ball: forwards most of its energy and keeps rolling.

Measured neutral against the base regime on MuSiQue - not a loss, and the
reach it buys is real; it simply did not pay on a benchmark whose questions
all have a definite answer.
"""

COMPARE = Profile(name="compare", damping=0.60, threshold_ratio=0.01, seed_width=6)
"""Two-sided question: wide enough contact to hold both compared regions.

The natural pairing is the coloured multi-seed path (each side its own
colour, the bridge is the answer); applied there via `as_colored`, remember
the width is per colour. Measured -.0015 on the plain path, i.e. flat - and
the plain path is the wrong home for this profile anyway.
"""

PROFILES: dict[str, Profile] = {
    profile.name: profile for profile in (PRECISE, EXPLORE, COMPARE)
}
"""Name -> profile map for UI selectors and config files."""

DEFAULT_PROFILE = EXPLORE.name
"""The profile that runs when a caller names none and supplies no config -
in `SpiywebIndex.retrieve()`, `ThermalSession`, the terminal and the viewer
alike. The reason is arithmetic rather than taste.

`RetrievalConfig()` splits 10.0 energy among 5 seeds and stops at 15% of it,
so the strongest seed forwards at most 1.23 against a threshold of 1.50 -
nothing can ever clear it and the web returns first contact only. That is
`top-k` with extra steps, which is the one thing this project exists to not
be, and it is what a first-time caller would have seen.

`RetrievalConfig()` itself stays where CLAUDE.md §2.1 put it: it carries the
canonical worked example, and no measured number may move. So the default is
applied one level up, where a person first meets the mechanism, and only
when that person has expressed no preference at all: an explicit config is
never overlaid, and the spread warning in `retrieve()` keeps guarding it."""
