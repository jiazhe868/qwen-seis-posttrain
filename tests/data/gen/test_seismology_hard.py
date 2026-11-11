"""The hard T1 families: each solver checked against an independent computation (forward model, ray tracing by
quadrature, stress-tensor rotation), and the bank kept apart from the 2026-10 families."""

from __future__ import annotations

import math

import pytest

from dpt.data.gen import seismology_hard as H
from dpt.data.gen.common import rng_for
from dpt.data.gen.families import FAMILIES, by_split, heldout_families_are_present

requires_private_heldout_families = pytest.mark.skipif(
    not heldout_families_are_present(), reason="the held-out eval families are not published (README, Data & licensing)")


@requires_private_heldout_families
def test_hard_bank_is_separate_from_v1():
    train, held = by_split("train", "hard"), by_split("heldout", "hard")
    assert len(train) == 12 and len(held) == 8
    assert all(f.steps >= 3 for f in train + held)
    assert not {f.name for f in train + held} & {f.name for f in by_split("train") + by_split("heldout")}
    assert {"crustal structure", "fault mechanics"} <= {f.topic for f in held}       # kinds no family trains on


def test_placeholder_parameter_ranges_are_never_drawn():
    """Families whose shown values must agree (make=...) list their inputs with a placeholder range [0, 1]."""
    for f in by_split("train", "hard") + by_split("heldout", "hard"):
        placeholder = [q.name for q in f.params if (q.lo, q.hi) == (0, 1)]
        assert f.make is not None or not placeholder, (f.name, placeholder)


def test_drawn_values_are_physical():
    rng = rng_for(2, "physical")
    for f in by_split("train", "hard") + by_split("heldout", "hard"):
        for _ in range(20):
            p = f.sample(rng)
            for k, v in p.items():
                if k.endswith("km_s"):
                    assert 1.0 <= v <= 20.0, (f.name, k, v)
                if k.endswith("_deg") and k.startswith("theta"):
                    assert 5 <= v <= 175, (f.name, k, v)
            assert f.solve(p) > 0 or f.abs_tol, (f.name, p)


def test_epicentre_is_recovered_from_consistent_data():
    vp, vs, (x0, y0) = 6.2, 3.6, (31.0, 22.0)
    k = 1 / vs - 1 / vp
    p = {"xb_km": 70, "xc_km": 25, "yc_km": 65, "vp_km_s": vp, "vs_km_s": vs, "xt_km": 120, "yt_km": -30,
         "sp_a_s": math.hypot(x0, y0) * k, "sp_b_s": math.hypot(x0 - 70, y0) * k,
         "sp_c_s": math.hypot(x0 - 25, y0 - 65) * k}
    assert H._epicentre(p) == pytest.approx((x0, y0))
    assert H._town_distance(p) == pytest.approx(math.hypot(x0 - 120, y0 + 30))


@pytest.mark.parametrize("baz", [35.0, 140.0, 255.0, 330.0])
def test_array_recovers_back_azimuth_and_apparent_velocity(baz):
    vapp, prop = 9.0, math.radians(baz + 180)
    sx, sy = math.sin(prop) / vapp, math.cos(prop) / vapp
    p = {"xb_km": 5.0, "yb_km": -2.0, "xc_km": 1.0, "yc_km": 6.0}
    p.update(tb_s=sx * 5.0 + sy * -2.0, tc_s=sx * 1.0 + sy * 6.0)
    assert H._backazimuth(p) == pytest.approx(baz)
    assert H._apparent_velocity(p) == pytest.approx(vapp)


def test_three_layer_head_wave_matches_snell_ray_segments():
    p = {"distance_km": 180.0, "h1_km": 4.0, "h2_km": 18.0, "v1_km_s": 4.5, "v2_km_s": 6.4, "v3_km_s": 8.1}
    t, xs = 0.0, 0.0
    for h, v in ((p["h1_km"], p["v1_km_s"]), (p["h2_km"], p["v2_km_s"])):
        th = math.asin(v / p["v3_km_s"])            # critical refraction at the top of layer 3
        t += 2 * h / (v * math.cos(th))
        xs += 2 * h * math.tan(th)
    t += (p["distance_km"] - xs) / p["v3_km_s"]
    assert H._head3(p) == pytest.approx(t)


def test_intercept_inversion_returns_the_forward_thickness():
    v1, v2, v3, h1, h2 = 3.2, 6.1, 8.0, 2.5, 14.0
    p = {"v1_km_s": v1, "v2_km_s": v2, "v3_km_s": v3, "ti2_s": 2 * h1 * math.sqrt(1 / v1 ** 2 - 1 / v2 ** 2),
         "ti3_s": 2 * h1 * math.sqrt(1 / v1 ** 2 - 1 / v3 ** 2) + 2 * h2 * math.sqrt(1 / v2 ** 2 - 1 / v3 ** 2)}
    assert H._h2_from_intercepts(p) == pytest.approx(h2)


def test_dipping_refractor_inversion_returns_velocity_and_dip():
    v1, v2, dip = 2.0, 3.6, math.radians(8.0)
    ic = math.asin(v1 / v2)
    p = {"v1_km_s": v1, "vd_km_s": v1 / math.sin(ic + dip), "vu_km_s": v1 / math.sin(ic - dip)}
    assert FAMILIES["dipping_refractor_velocity"].solve(p) == pytest.approx(v2)
    assert FAMILIES["dipping_refractor_dip"].solve(p) == pytest.approx(8.0)


def test_gutenberg_richter_count_from_a_sample():
    p = {f"m{i}": m for i, m in enumerate([2.0, 2.1, 2.0, 2.4, 2.2, 3.1, 2.0, 2.6, 2.3, 2.0], 1)}
    p.update(mc=2.0, n_total=1000, m_target=4.5)
    b = math.log10(math.e) / (2.27 - 1.95)
    assert H._gr_b(p) == pytest.approx(b)
    assert H._gr_count(p) == pytest.approx(1000 * 10 ** (-b * 2.5))
    rng = rng_for(1, "gr")
    draws = [FAMILIES["gr_count_from_magnitude_sample"].sample(rng) for _ in range(50)]
    assert all(min(d[f"m{i}"] for i in range(1, 11)) >= d["mc"] for d in draws)      # nothing below Mc
    assert any(d["m1"] == d["mc"] for d in draws)                                      # the Mc bin is populated


def test_wadati_fit_on_exact_data_gives_the_true_ratio():
    r, vp, t0 = 1.75, 6.0, 4.0
    p = {}
    for i, d in enumerate([30.0, 80.0, 150.0, 220.0], 1):
        p[f"tp{i}_s"], p[f"sp{i}_s"] = t0 + d / vp, (r - 1) * d / vp
    assert H._wadati_fit_ratio(p) == pytest.approx(r)
    assert H._poisson_from_wadati(p) == pytest.approx((r * r - 2) / (2 * (r * r - 1)))


def test_two_interface_energy_limits():
    p = {"rho1_kg_m3": 2200, "v1_m_s": 2500, "rho2_kg_m3": 2200, "v2_m_s": 2500, "rho3_kg_m3": 2600, "v3_m_s": 4000}
    z2, z3 = 2200 * 2500, 2600 * 4000
    assert H._two_interface_energy(p) == pytest.approx(((z3 - z2) / (z3 + z2)) ** 2)   # no first contrast


def test_q_and_depth_phase_special_cases():
    q, dt, fa, fb = 250.0, 20.0, 1.0, 6.0
    p = {"dt_s": dt, "fa_hz": fa, "fb_hz": fb, "ra": 0.7 * math.exp(-math.pi * fa * dt / q),
         "rb": 0.7 * math.exp(-math.pi * fb * dt / q)}
    assert H._q_spectral(p) == pytest.approx(q)
    assert H._pp_depth({"dt_s": 10.0, "p_s_deg": 0.0, "v_km_s": 6.0}) == pytest.approx(30.0)   # vertical ray
