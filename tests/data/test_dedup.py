"""MinHash + LSH near-duplicate detection (report §处理流水线 step 2)."""

from __future__ import annotations

import numpy as np

from dpt.data.dedup import (
    MinHasher,
    MinHashIndex,
    cross_dedup,
    exact_jaccard,
    near_duplicate_dedup,
    optimal_lsh_params,
)

BASE = ("A P wave crosses the upper mantle at 7.5 km/s. Ignoring ray bending, how many seconds does the wave "
        "take to cover the 150 km between the source and a recording station on the surface?")
NEAR = BASE.replace("take to", "need to").replace(". Ignoring", "; ignoring")
OTHER = "Compute the wavelength of a 15 Hz shear wave travelling through sediments at 3000 m/s."


def test_signatures_are_deterministic_and_seed_dependent():
    h1, h2, h3 = MinHasher(seed=1), MinHasher(seed=1), MinHasher(seed=2)
    hs = h1.shingle_hashes(BASE)
    assert np.array_equal(h1.signature(hs), h2.signature(hs))
    assert not np.array_equal(h1.signature(hs), h3.signature(hs))


def test_multiply_add_shift_is_exact_mod_2_64():
    """Each permutation is h(x) = ((a*x + b) mod 2^64) >> 32, computed with exact uint64 wrap-around."""
    h = MinHasher(num_perm=4)
    x = 2**32 - 1
    sig = h.signature(np.array([x], dtype=np.uint64))
    expected = [((int(a) * x + int(b)) % 2**64) >> 32 for a, b in zip(h.a, h.b)]
    assert [int(v) for v in sig] == expected


def test_estimates_are_unbiased_across_many_pairs():
    """Guards against correlated 'permutations' (e.g. (a*x+b) mod p with small a), which break MinHash."""
    rng = np.random.RandomState(0)
    vocab = [f"w{i}" for i in range(400)]
    errs = []
    h = MinHasher(num_perm=128)
    for _ in range(30):
        words = list(rng.choice(vocab, 60))
        other = [w if rng.rand() > 0.3 else str(rng.choice(vocab)) for w in words]
        a, b = h.shingle_hashes(" ".join(words)), h.shingle_hashes(" ".join(other))
        errs.append(float((h.signature(a) == h.signature(b)).mean()) - exact_jaccard(a, b))
    assert abs(np.mean(errs)) < 0.03 and np.max(np.abs(errs)) < 0.15


def test_signature_estimate_tracks_exact_jaccard():
    h = MinHasher(num_perm=256)
    a, b = h.shingle_hashes(BASE), h.shingle_hashes(NEAR)
    exact = exact_jaccard(a, b)
    est = float((h.signature(a) == h.signature(b)).mean())
    assert 0.8 < exact < 0.9 and abs(est - exact) < 0.1


def test_lsh_params_favor_recall_at_threshold():
    b, r = optimal_lsh_params(0.8, 128)
    assert b * r <= 128
    assert 1 - (1 - 0.8**r) ** b > 0.95  # ≥95% chance an exactly-0.8 pair becomes a candidate


def test_near_duplicates_cluster_and_first_is_kept():
    res = near_duplicate_dedup([BASE, OTHER, NEAR])
    assert res.kept == [0, 1]
    assert res.removed[0]["index"] == 2 and res.removed[0]["duplicate_of"] == 0
    assert res.removed[0]["jaccard"] >= 0.8


def test_priority_chooses_representative():
    res = near_duplicate_dedup([BASE, NEAR], priority=[1, 0])
    assert res.kept == [1]


def test_transitive_clusters():
    a = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma"
    b = a.replace("sigma", "tau")
    c = b.replace("alpha", "omega")
    res = near_duplicate_dedup([a, b, c], threshold=0.8)
    assert len(res.kept) == 1


def test_empty_texts_are_never_duplicates():
    res = near_duplicate_dedup(["", "", "ab"])
    assert res.kept == [0, 1, 2]


def test_cross_dedup_against_other_bank():
    res = cross_dedup([NEAR, OTHER], [("bank/1", BASE)])
    assert res.kept == [1] and res.removed[0]["duplicate_of"] == "bank/1"


def test_index_query_returns_verified_scores():
    idx = MinHashIndex(0.5)
    idx.add("x", BASE)
    hits = idx.query(NEAR)
    assert hits and hits[0][0] == "x" and hits[0][1] >= 0.5
    assert idx.query(OTHER) == []


def test_instance_dedup_keeps_every_instance_of_a_template_but_drops_rewordings():
    from dpt.data.dedup import instance_dedup

    gens = [{"name": "travel_time", "params": {"distance_km": 100, "velocity_km_s": v}} for v in (5.0, 6.0, 7.0)]
    gens.append({"name": "travel_time", "params": {"velocity_km_s": 6, "distance_km": 100.0}})   # reworded twin
    gens.append({"name": "wavelength", "params": {"distance_km": 100, "velocity_km_s": 6.0}})     # other solver
    res = instance_dedup(gens)
    assert res.kept == [0, 1, 2, 4]
    assert res.removed == [{"index": 3, "duplicate_of": 1, "match": "same_instance"}]
