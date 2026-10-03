"""Entity edge builder: rarity-weighted 1/df sums, df guard, omission rule."""

from __future__ import annotations

import pytest

from spiyweb import EntityEdgeConfig
from spiyweb.edges import build_entity_edges

# The canonical hand example: df(A)=3, df(B)=df(C)=2 over four chunks.
CORPUS: dict[str, list[str]] = {
    "c1": ["A", "B"],
    "c2": ["A"],
    "c3": ["A", "C"],
    "c4": ["B", "C"],
}


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_config_max_df_ratio_outside_half_open_interval_raises(ratio: float) -> None:
    with pytest.raises(ValueError, match="max_df_ratio"):
        EntityEdgeConfig(max_df_ratio=ratio)


def test_config_default_is_the_measured_grid_winner() -> None:
    # 2026-08-14 MuSiQue grid: the original 0.5 hand value produced 2.5M
    # entity edges (mean degree 438); 0.02 won and is now the default.
    assert EntityEdgeConfig().max_df_ratio == pytest.approx(0.02)


def test_canonical_hand_example_yields_exact_rarity_fractions() -> None:
    edges = build_entity_edges(CORPUS, config=EntityEdgeConfig(max_df_ratio=1.0))
    assert edges == [
        ("c1", "c2", pytest.approx(1 / 3)),
        ("c1", "c3", pytest.approx(1 / 3)),
        ("c1", "c4", pytest.approx(1 / 2)),
        ("c2", "c3", pytest.approx(1 / 3)),
        ("c3", "c4", pytest.approx(1 / 2)),
    ]
    pairs = {(u, v) for u, v, _ in edges}
    assert ("c2", "c4") not in pairs, (
        "a pair sharing no entity is omitted entirely, never emitted at 0.0"
    )


def test_two_shared_entities_sum_their_contributions() -> None:
    edges = build_entity_edges(
        {"c1": ["A", "B"], "c2": ["A", "B"], "c3": ["A"]},
        config=EntityEdgeConfig(max_df_ratio=1.0),
    )
    weights = {(u, v): w for u, v, w in edges}
    assert weights[("c1", "c2")] == pytest.approx(1 / 3 + 1 / 2), (
        "converging evidence: contributions from several shared entities add up"
    )


def test_max_df_ratio_drops_the_stopword_entity() -> None:
    # df(A)=3 over n=4 chunks: ratio 0.5 -> max_df=2, so A (the near-stopword)
    # is dropped and only the rare B/C pairs survive. The ratio is explicit:
    # the measured 0.02 default assumes a 1000-question corpus, not 4 chunks.
    edges = build_entity_edges(CORPUS, config=EntityEdgeConfig(max_df_ratio=0.5))
    weights = {(u, v): w for u, v, w in edges}
    assert weights == {
        ("c1", "c4"): pytest.approx(1 / 2),
        ("c3", "c4"): pytest.approx(1 / 2),
    }, "1/df bounds a stopword's weight but not its clique; the guard does"


def test_max_df_ratio_boundary_is_strict() -> None:
    # df(A)=2 over n=4 chunks equals max_df exactly at ratio 0.5; the guard
    # drops only entities strictly ABOVE the bound, so A survives.
    edges = build_entity_edges(
        {"c1": ["A"], "c2": ["A"], "c3": [], "c4": []},
        config=EntityEdgeConfig(max_df_ratio=0.5),
    )
    assert edges == [("c1", "c2", pytest.approx(1 / 2))]


def test_entity_in_a_single_chunk_emits_nothing() -> None:
    assert build_entity_edges({"c1": ["A"], "c2": ["B"]}) == []


def test_duplicate_entity_within_one_chunk_counts_once() -> None:
    edges = build_entity_edges(
        {"c1": ["A", "A"], "c2": ["A"]},
        config=EntityEdgeConfig(max_df_ratio=1.0),
    )
    assert edges == [("c1", "c2", pytest.approx(1 / 2))], (
        "mentioning an entity twice in one chunk is not stronger evidence"
    )


def test_empty_chunk_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="chunk id must not be empty"):
        build_entity_edges({"": ["A"]})


def test_empty_entity_string_raises_with_the_chunk_id() -> None:
    with pytest.raises(ValueError, match=r"'c1'.*empty entity"):
        build_entity_edges({"c1": [""]})


def test_empty_mapping_and_entityless_chunks_yield_no_edges() -> None:
    assert build_entity_edges({}) == []
    assert build_entity_edges({"c1": [], "c2": []}) == []


def test_output_is_canonically_ordered_and_sorted() -> None:
    edges = build_entity_edges(
        {"z": ["A"], "a": ["A"], "m": ["A"]},
        config=EntityEdgeConfig(max_df_ratio=1.0),
    )
    assert edges == sorted(edges)
    assert all(u < v for u, v, _ in edges), "each pair is emitted as (min, max)"


def test_result_is_independent_of_mapping_insertion_order() -> None:
    reordered = {key: CORPUS[key] for key in ["c4", "c2", "c3", "c1"]}
    cfg = EntityEdgeConfig(max_df_ratio=1.0)
    assert build_entity_edges(CORPUS, cfg) == build_entity_edges(reordered, cfg), (
        "fixed sorted iteration makes the sums bit-identical on every platform"
    )


def test_a_small_corpus_still_gets_an_entity_layer() -> None:
    """The guard must not make the main hop fuel structurally impossible.

    At the measured default ratio of 0.02, `max_df_ratio * n_chunks` falls
    below 2 for every corpus under 100 chunks - and an entity needs two
    chunks to pair at all. The layer therefore came out empty on any small
    corpus, silently. Found on 2026-08-26 by running the real pipeline over
    three documents: `morgan` bridged two of them and was dropped against a
    threshold of 0.16.
    """
    entities = {
        "a:0": ["morgan", "tesla"],
        "a:1": ["morgan"],
        "b:0": ["marconi"],
    }
    edges = build_entity_edges(entities, EntityEdgeConfig())
    assert edges == [("a:0", "a:1", 0.5)], edges


def test_the_floor_never_binds_on_a_corpus_the_ratio_can_serve() -> None:
    """It cannot move a sealed number: 0.02 * 3336 is 66.7, far above 2."""
    shared = {f"c:{i}": ["everywhere"] for i in range(200)}
    # df = 200 against a ceiling of 0.02 * 200 = 4: the guard still fires,
    # exactly as it did before the floor existed.
    assert build_entity_edges(shared, EntityEdgeConfig()) == []


def test_max_df_cap_is_an_absolute_ceiling_on_top_of_the_ratio() -> None:
    # Ratio 1.0 alone keeps every entity; a cap of 2 drops A (df=3) exactly
    # like the ratio-0.5 guard does - the lower ceiling wins.
    capped = build_entity_edges(
        CORPUS, config=EntityEdgeConfig(max_df_ratio=1.0, max_df_cap=2)
    )
    by_ratio = build_entity_edges(CORPUS, config=EntityEdgeConfig(max_df_ratio=0.5))
    assert capped == by_ratio


def test_max_df_cap_does_not_grow_with_the_corpus() -> None:
    # One entity in 50 of 1000 chunks: the ratio ceiling (0.1 * 1000 = 100)
    # keeps its 1225-pair clique, a cap of 40 drops it whatever n is.
    corpus = {f"c{i}": (["hub"] if i < 50 else []) for i in range(1000)}
    assert len(build_entity_edges(corpus, EntityEdgeConfig(max_df_ratio=0.1))) == 1225
    assert (
        build_entity_edges(corpus, EntityEdgeConfig(max_df_ratio=0.1, max_df_cap=40))
        == []
    )


def test_max_df_cap_default_is_the_measured_winner_and_floored_at_two() -> None:
    # 2026-10-03, pre-registered: cap 100 halved the sealed entity layers with
    # S@5 moving by at most +.0003; None remains the way to switch it off.
    assert EntityEdgeConfig().max_df_cap == 100
    with pytest.raises(ValueError, match="max_df_cap"):
        EntityEdgeConfig(max_df_cap=1)
