"""Long T1 families (dpt.data.gen.seismology_long, 2026-11-07): training only, no held-out kind, and every solver
agrees with an independent computation of the same chain."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from dpt.data.gen.families import FAMILIES, by_split

LONG = {f.name: f for f in by_split("train", "long")}
HELDOUT = {f.name for f in FAMILIES.values() if f.split == "heldout"}


def test_six_training_families_none_held_out():
    assert len(LONG) == 6
    assert not by_split("heldout", "long")
    assert not set(LONG) & HELDOUT
    other = {f.name for f in FAMILIES.values() if f.bank != "long"}
    assert not set(LONG) & other
    for f in LONG.values():
        assert f.steps >= 6 and len(f.templates) == 2


def _independent(name: str, p: dict[str, float]) -> float:
    if name == "locate_then_local_magnitude":                  # 3-D location by least squares
        k = p["vp_km_s"] * p["vs_km_s"] / (p["vp_km_s"] - p["vs_km_s"])
        st = [(0, 0, k * p["sp_a_s"]), (p["xb_km"], 0, k * p["sp_b_s"]), (p["xc_km"], p["yc_km"], k * p["sp_c_s"])]
        a = np.array([[2 * (sx - st[0][0]), 2 * (sy - st[0][1])] for sx, sy, _ in st[1:]], float)
        b = np.array([st[0][2] ** 2 - r ** 2 + sx ** 2 + sy ** 2 - st[0][0] ** 2 - st[0][1] ** 2
                      for sx, sy, r in st[1:]], float)
        x, y = np.linalg.solve(a, b)
        h = math.sqrt(st[0][2] ** 2 - x ** 2 - y ** 2)
        r = math.dist((x, y, h), (p["xd_km"], p["yd_km"], 0))
        return math.log10(p["amp_mm"]) + 2.76 * math.log10(r) - 2.48
    if name == "wadati_origin_then_depth":
        tp = np.array([p[f"tp{i}_s"] for i in range(1, 5)])
        ts = np.array([p[f"ts{i}_s"] for i in range(1, 5)])
        slope, icpt = np.polyfit(tp, ts - tp, 1)
        r = p["vp_km_s"] * (p["tp4_s"] + icpt / slope)
        return math.sqrt(r * r - p["d4_km"] ** 2)
    if name == "refraction_then_reflection_time":
        v1 = np.polyfit([p["t1_s"], p["t2_s"]], [p["x1_km"], p["x2_km"]], 1)[0]
        slope, icpt = np.polyfit([p["x3_km"], p["x4_km"]], [p["t3_s"], p["t4_s"]], 1)
        v2 = 1 / slope
        h = icpt / (2 * math.sqrt(1 / v1 ** 2 - 1 / v2 ** 2))
        return 2 * math.hypot(p["xr_km"] / 2, h) / v1
    if name == "fault_slip_stress_energy":
        m0 = 10 ** (1.5 * p["mw"] + 9.1)
        slip = m0 / (p["mu_gpa"] * 1e9 * p["l_km"] * p["w_km"] * 1e6)
        es = (2 / math.pi) * slip / (p["w_km"] * 1e3) / 2 * m0      # Δσ/(2μ)·M0 with μ cancelled
        return (2 / 3) * math.log10(es) - 2.9
    if name == "attenuation_then_amplitude":                     # the same chain as one power law
        g1, g2 = p["a1"] * p["r1_km"], p["a2"] * p["r2_km"]
        frac = (p["f2_hz"] * p["r3_km"] - p["f_hz"] * p["r1_km"]) / (p["f_hz"] * (p["r2_km"] - p["r1_km"]))
        return g1 * (g2 / g1) ** frac / p["r3_km"]
    if name == "great_circle_origin_time":                       # haversine form
        la, lb = math.radians(p["lat_e"]), math.radians(p["lat_s"])
        dl = math.radians(p["lon_s"] - p["lon_e"])
        hav = math.sin((lb - la) / 2) ** 2 + math.cos(la) * math.cos(lb) * math.sin(dl / 2) ** 2
        delta = math.degrees(2 * math.asin(math.sqrt(hav)))
        return p["tarr_s"] - p["a_s"] - p["b_s_per_deg"] * delta
    raise KeyError(name)


@pytest.mark.parametrize("name", sorted(LONG))
def test_solver_matches_an_independent_computation(name):
    f, rng = LONG[name], random.Random(7)
    for _ in range(40):
        p = f.sample(rng)
        got, ref = f.solve(p), _independent(name, p)
        assert math.isclose(got, ref, rel_tol=1e-6, abs_tol=1e-9), (name, p, got, ref)
        for t in range(2):
            q = f.render(p, t)
            assert "{" not in q and "}" not in q


def test_attenuation_answers_stay_readable():
    f, rng = LONG["attenuation_then_amplitude"], random.Random(3)
    assert min(f.solve(f.sample(rng)) for _ in range(200)) >= 0.05
