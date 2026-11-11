"""Long T1 families (bank "long", 2026-11-07): six to eight steps that chain physics from different topics.

The v2 RL bank needs problems the SFT model still fails sometimes. In RL steps 1–20, 88% of T1 groups were solved
8/8. The hard bank (``seismology_hard``) chains three to five steps. These chain more, and each joins two or three
topics, so a slip in any one step shows in the answer:

- locate_then_local_magnitude: S−P distances → epicentre → focal depth → hypocentral distance to a fourth station →
  local magnitude there;
- wadati_origin_then_depth: least-squares Wadati line → origin time → hypocentral distance → focal depth;
- refraction_then_reflection_time: direct and head-wave slopes → intercept → layer thickness → reflection time;
- fault_slip_stress_energy: Mw → moment → average slip → stress drop → radiated energy → energy magnitude;
- attenuation_then_amplitude: geometric spreading → Q from two stations → source level → amplitude at a
  third station and a higher frequency;
- great_circle_origin_time: great-circle distance from coordinates → travel time → origin time.

Every family states its conventions and formulas, so each problem has one answer. Draws are generated from a
consistent truth, rounded for display, and rejected when the rounded values would move the answer by more than a
fraction of its tolerance (the conditioning checks in each ``make``). The solver works on the shown values, as in
the other banks.

None of these is a held-out kind (D-Num: tsunami, Omori, Fresnel, Dix, Brune, recurrence, strain, Young's modulus,
exceedance, Ms; D-Num-hard: receiver functions, gradient rays, spectral plateau, GMPE, directivity, Coulomb).
``tests/data/gen/test_seismology_long.py`` checks the names and the solvers.
"""

from __future__ import annotations

import dataclasses
import math
import random

from dpt.data.gen.families import Family, Param, register

P = Param
LONG = "long"


def _long(f: Family) -> Family:
    return register(dataclasses.replace(f, bank=LONG))


def _rounded(params: tuple[Param, ...], raw: dict[str, float]) -> dict[str, float]:
    return {q.name: q.round(raw[q.name]) for q in params}


def _stable(solver, p: dict[str, float], keys: tuple[str, ...], step: float, tol: float) -> bool:
    """The answer moves by less than ``tol`` (relative) when any shown value in ``keys`` moves by half its rounding
    step ``step``: the problem is well conditioned at the precision it is shown with."""
    base = solver(p)
    if not math.isfinite(base) or base == 0:
        return False
    for k in keys:
        for s in (-0.5 * step, 0.5 * step):
            q = dict(p)
            q[k] = p[k] + s
            try:
                v = solver(q)
            except (ValueError, ZeroDivisionError):
                return False
            if not math.isfinite(v) or abs(v - base) > tol * abs(base):
                return False
    return True


# =============================================================================================
# 1. Locate, depth, then local magnitude at a fourth station (8 steps)
# =============================================================================================
_LM = (P("xb_km", 0, 1, integer=True), P("xc_km", 0, 1, integer=True), P("yc_km", 0, 1, integer=True),
       P("sp_a_s", 0, 1, decimals=2), P("sp_b_s", 0, 1, decimals=2), P("sp_c_s", 0, 1, decimals=2),
       P("vp_km_s", 0, 1, decimals=2), P("vs_km_s", 0, 1, decimals=2),
       P("xd_km", 0, 1, integer=True), P("yd_km", 0, 1, integer=True), P("amp_mm", 0, 1, digits=3))


def _lm_hypocentre(p: dict[str, float]) -> tuple[float, float, float]:
    k = p["vp_km_s"] * p["vs_km_s"] / (p["vp_km_s"] - p["vs_km_s"])
    ra, rb, rc = (k * p[s] for s in ("sp_a_s", "sp_b_s", "sp_c_s"))
    x = (ra * ra - rb * rb + p["xb_km"] ** 2) / (2 * p["xb_km"])
    y = ((ra * ra - rc * rc + p["xc_km"] ** 2 + p["yc_km"] ** 2) / 2 - x * p["xc_km"]) / p["yc_km"]
    h = math.sqrt(ra * ra - x * x - y * y)                 # depth: A's sphere at the epicentre
    return x, y, h


def _lm(p: dict[str, float]) -> float:
    x, y, h = _lm_hypocentre(p)
    r = math.sqrt((x - p["xd_km"]) ** 2 + (y - p["yd_km"]) ** 2 + h * h)
    return math.log10(p["amp_mm"]) + 2.76 * math.log10(r) - 2.48


def _lm_make(rng: random.Random) -> dict[str, float]:
    while True:
        vp = rng.uniform(5.8, 6.6)
        vs = vp / rng.uniform(1.70, 1.80)
        xb, xc, yc = rng.uniform(40, 90), rng.uniform(10, 60), rng.uniform(40, 90)
        x0, y0, h0 = rng.uniform(5, 70), rng.uniform(5, 70), rng.uniform(12, 30)
        k = 1 / vs - 1 / vp
        sp = lambda xs, ys: math.sqrt((x0 - xs) ** 2 + (y0 - ys) ** 2 + h0 * h0) * k  # noqa: E731
        raw = {"xb_km": xb, "xc_km": xc, "yc_km": yc, "vp_km_s": vp, "vs_km_s": vs,
               "sp_a_s": sp(0, 0), "sp_b_s": sp(xb, 0), "sp_c_s": sp(xc, yc),
               "xd_km": rng.uniform(-60, 160), "yd_km": rng.uniform(-60, 160), "amp_mm": 10 ** rng.uniform(-0.5, 2)}
        p = _rounded(_LM, raw)
        try:
            x, y, h = _lm_hypocentre(p)
        except ValueError:
            continue
        rd = math.hypot(x - p["xd_km"], y - p["yd_km"])
        if (math.hypot(x - x0, y - y0) < 2 and abs(h - h0) < 2.5 and 30 <= rd <= 250
                and _stable(_lm, p, ("sp_a_s", "sp_b_s", "sp_c_s"), 0.01, 0.01)):
            return p


_long(Family(
    "locate_then_local_magnitude", "earthquake location and magnitude", _lm, _LM, "",
    (
        "Three stations record a local earthquake. In a map frame (x east, y north, km) station A is at the origin, "
        "B at ({xb_km}, 0) and C at ({xc_km}, {yc_km}); all three are at sea level. The S−P times are {sp_a_s} s at "
        "A, {sp_b_s} s at B and {sp_c_s} s at C. Take vp = {vp_km_s} km/s, vs = {vs_km_s} km/s and straight rays, "
        "so each S−P time gives the hypocentral distance to its station. Find the epicentre by subtracting the "
        "sphere equations of B and of C from that of A, then the focal depth from station A's distance. A fourth "
        "station D at ({xd_km}, {yd_km}) records a peak Wood–Anderson amplitude of {amp_mm} mm. Using "
        "ML = log10(A) + 2.76·log10(R) − 2.48, with A in mm and R the hypocentral distance to D in km, what is "
        "the local magnitude?",
        "S−P intervals of {sp_a_s} s, {sp_b_s} s and {sp_c_s} s are read at stations A (0, 0), B ({xb_km}, 0) and "
        "C ({xc_km}, {yc_km}) (km, x east, y north, all at sea level). With vp = {vp_km_s} km/s and vs = {vs_km_s} "
        "km/s, each interval converts to a hypocentral distance along a straight ray. Subtract B's and C's sphere "
        "equations from A's to get the epicentre, use A's distance for the depth, and compute the hypocentral "
        "distance R to station D at ({xd_km}, {yd_km}), where the peak Wood–Anderson amplitude is {amp_mm} mm. "
        "Report the local magnitude ML = log10(A [mm]) + 2.76·log10(R [km]) − 2.48.",
    ),
    steps=8, rel_tol=0.0, abs_tol=0.06, make=_lm_make,
))


# =============================================================================================
# 2. Wadati line by least squares, origin time, then focal depth (7 steps)
# =============================================================================================
_WO = (P("tp1_s", 0, 1, decimals=2), P("tp2_s", 0, 1, decimals=2), P("tp3_s", 0, 1, decimals=2),
       P("tp4_s", 0, 1, decimals=2), P("ts1_s", 0, 1, decimals=2), P("ts2_s", 0, 1, decimals=2),
       P("ts3_s", 0, 1, decimals=2), P("ts4_s", 0, 1, decimals=2), P("vp_km_s", 0, 1, decimals=2),
       P("d4_km", 0, 1, decimals=1))


def _wo_origin(p: dict[str, float]) -> float:
    tp = [p[f"tp{i}_s"] for i in range(1, 5)]
    sp = [p[f"ts{i}_s"] - p[f"tp{i}_s"] for i in range(1, 5)]
    mx, my = sum(tp) / 4, sum(sp) / 4
    sxx = sum((a - mx) ** 2 for a in tp)
    sxy = sum((a - mx) * (b - my) for a, b in zip(tp, sp))
    slope = sxy / sxx
    intercept = my - slope * mx
    return -intercept / slope                               # S−P = 0 at the origin time


def _wo(p: dict[str, float]) -> float:
    r = p["vp_km_s"] * (p["tp4_s"] - _wo_origin(p))
    return math.sqrt(r * r - p["d4_km"] ** 2)


def _wo_make(rng: random.Random) -> dict[str, float]:
    while True:
        t0, vp = rng.uniform(3, 20), rng.uniform(5.8, 6.6)
        vs, h = vp / rng.uniform(1.70, 1.80), rng.uniform(12, 30)
        dist = sorted([rng.uniform(40, 140) for _ in range(3)]) + [rng.uniform(4, 15)]
        raw = {"vp_km_s": vp, "d4_km": dist[3]}
        for i, d in enumerate(dist, start=1):
            r = math.hypot(d, h)
            raw[f"tp{i}_s"], raw[f"ts{i}_s"] = t0 + r / vp, t0 + r / vs
        p = _rounded(_WO, raw)
        try:
            ok = abs(_wo(p) - h) < 1.5
        except ValueError:
            continue
        if ok and _stable(_wo, p, tuple(f"{a}{i}_s" for a in ("tp", "ts") for i in range(1, 5)), 0.01, 0.015):
            return p


_long(Family(
    "wadati_origin_then_depth", "earthquake location", _wo, _WO, "km",
    (
        "Four stations record P and S arrivals from one earthquake (seconds after 12:00:00 UTC): station 1 P "
        "{tp1_s}, S {ts1_s}; station 2 P {tp2_s}, S {ts2_s}; station 3 P {tp3_s}, S {ts3_s}; station 4 P {tp4_s}, "
        "S {ts4_s}. Fit a Wadati diagram by ordinary least squares (S−P time against P arrival time) and take the "
        "origin time where the line reaches S−P = 0. Station 4 is {d4_km} km from the epicentre and vp = {vp_km_s} "
        "km/s along a straight ray. What is the focal depth in km?",
        "P and S arrival times, in seconds after 12:00:00 UTC, are: station 1 ({tp1_s}, {ts1_s}), station 2 "
        "({tp2_s}, {ts2_s}), station 3 ({tp3_s}, {ts3_s}), station 4 ({tp4_s}, {ts4_s}). Regress the S−P times on "
        "the P times with an ordinary least-squares line; its zero gives the origin time. With vp = {vp_km_s} km/s "
        "on straight rays and an epicentral distance of {d4_km} km to station 4, find the focal depth (km).",
    ),
    steps=7, rel_tol=0.03, make=_wo_make,
))


# =============================================================================================
# 3. Two-layer refraction, then the reflection time at a new offset (6 steps)
# =============================================================================================
_RR = (P("x1_km", 0, 1, decimals=2), P("t1_s", 0, 1, decimals=3), P("x2_km", 0, 1, decimals=2),
       P("t2_s", 0, 1, decimals=3), P("x3_km", 0, 1, decimals=2), P("t3_s", 0, 1, decimals=3),
       P("x4_km", 0, 1, decimals=2), P("t4_s", 0, 1, decimals=3), P("xr_km", 0, 1, decimals=2))


def _rr(p: dict[str, float]) -> float:
    v1 = (p["x2_km"] - p["x1_km"]) / (p["t2_s"] - p["t1_s"])
    v2 = (p["x4_km"] - p["x3_km"]) / (p["t4_s"] - p["t3_s"])
    ti = p["t3_s"] - p["x3_km"] / v2
    h = ti * v1 * v2 / (2 * math.sqrt(v2 * v2 - v1 * v1))
    return math.sqrt(p["xr_km"] ** 2 + 4 * h * h) / v1


def _rr_make(rng: random.Random) -> dict[str, float]:
    while True:
        v1 = rng.uniform(1.6, 3.0)
        v2, h = v1 * rng.uniform(1.6, 2.6), rng.uniform(0.3, 2.0)
        ti = 2 * h * math.sqrt(1 / v1 ** 2 - 1 / v2 ** 2)
        xc = ti / (1 / v1 - 1 / v2)                          # crossover distance
        x1, x2 = rng.uniform(0.05, 0.3) * xc, rng.uniform(0.5, 0.8) * xc
        x3 = xc * rng.uniform(1.3, 1.8)
        x4 = x3 + xc * rng.uniform(0.8, 1.5)
        raw = {"x1_km": x1, "t1_s": x1 / v1, "x2_km": x2, "t2_s": x2 / v1, "x3_km": x3, "t3_s": x3 / v2 + ti,
               "x4_km": x4, "t4_s": x4 / v2 + ti, "xr_km": rng.uniform(0.2, 2.5) * h}
        p = _rounded(_RR, raw)
        try:
            ok = abs(_rr(p) - math.sqrt(p["xr_km"] ** 2 + 4 * h * h) / v1) < 0.01 * _rr(p)
        except ValueError:
            continue
        if ok and _stable(_rr, p, ("t1_s", "t2_s", "t3_s", "t4_s"), 0.001, 0.01):
            return p


_long(Family(
    "refraction_then_reflection_time", "refraction and reflection", _rr, _RR, "s",
    (
        "A refraction survey over one flat layer above a faster half-space records first arrivals at a geophone "
        "line from a surface source. The direct wave arrives at {t1_s} s at {x1_km} km and {t2_s} s at {x2_km} km; "
        "the head wave arrives at {t3_s} s at {x3_km} km and {t4_s} s at {x4_km} km. From the two slopes and the "
        "head wave's intercept time, find the layer thickness. What is the two-way time, in s, of the reflection "
        "from the base of the layer at a source–receiver offset of {xr_km} km?",
        "First breaks over a single horizontal layer on a faster half-space: direct arrivals at ({x1_km} km, "
        "{t1_s} s) and ({x2_km} km, {t2_s} s); refracted (head-wave) arrivals at ({x3_km} km, {t3_s} s) and "
        "({x4_km} km, {t4_s} s). Get both velocities from the slopes, the intercept time of the head-wave line, "
        "and from it the thickness of the layer. Then give the travel time (s) of the wave reflected at the "
        "interface for a source–receiver offset of {xr_km} km.",
    ),
    steps=6, rel_tol=0.02, make=_rr_make,
))


# =============================================================================================
# 4. Mw → moment → slip → stress drop → radiated energy → energy magnitude (6 steps)
# =============================================================================================
_FE = (P("mw", 0, 1, decimals=1), P("l_km", 0, 1, integer=True), P("w_km", 0, 1, integer=True),
       P("mu_gpa", 0, 1, integer=True))


def _fe_parts(p: dict[str, float]) -> tuple[float, float, float, float]:
    m0 = 10 ** (1.5 * p["mw"] + 9.1)                        # N·m
    mu = p["mu_gpa"] * 1e9
    d = m0 / (mu * p["l_km"] * 1e3 * p["w_km"] * 1e3)        # m
    dsig = (2 / math.pi) * mu * d / (p["w_km"] * 1e3)        # Pa
    es = dsig / (2 * mu) * m0                                # J
    return m0, d, dsig, es


def _fe(p: dict[str, float]) -> float:
    return (2 / 3) * math.log10(_fe_parts(p)[3]) - 2.9


def _fe_make(rng: random.Random) -> dict[str, float]:
    while True:
        mw = rng.uniform(5.5, 7.6)
        length = 10 ** (-2.44 + 0.59 * mw) * rng.uniform(0.7, 1.4)
        width = min(length * rng.uniform(0.3, 0.6), rng.uniform(12, 22))
        p = _rounded(_FE, {"mw": mw, "l_km": length, "w_km": width, "mu_gpa": rng.choice([30, 32, 33, 35])})
        if p["l_km"] < 5 or p["w_km"] < 3:
            continue
        dsig = _fe_parts(p)[2] / 1e6
        if 0.5 <= dsig <= 30:
            return p


_long(Family(
    "fault_slip_stress_energy", "source physics", _fe, _FE, "",
    (
        "A strike-slip earthquake of moment magnitude Mw {mw} ruptures a rectangular fault {l_km} km long and "
        "{w_km} km wide in rock of rigidity {mu_gpa} GPa. Use log10(M0) = 1.5·Mw + 9.1 (M0 in N·m), the average "
        "slip from M0 = μ·L·W·D, the stress drop Δσ = (2/π)·μ·D/W, the radiated energy Es = (Δσ / 2μ)·M0 and the "
        "energy magnitude Me = (2/3)·log10(Es) − 2.9 with Es in J. What is Me?",
        "For an Mw {mw} strike-slip event on a {l_km} km by {w_km} km rectangular fault (rigidity {mu_gpa} GPa): "
        "convert Mw to seismic moment with log10(M0 [N·m]) = 1.5·Mw + 9.1, find the average slip D from "
        "M0 = μLWD, the static stress drop Δσ = (2/π)·μ·D/W, then the radiated energy with Orowan's estimate "
        "Es = Δσ·M0/(2μ). Report the energy magnitude Me = (2/3)·log10(Es [J]) − 2.9.",
    ),
    steps=6, rel_tol=0.0, abs_tol=0.05, make=_fe_make,
))


# =============================================================================================
# 5. Q from two stations, then the amplitude at a third station and a higher frequency (7 steps)
# =============================================================================================
_AT = (P("f_hz", 0, 1, decimals=1), P("f2_hz", 0, 1, decimals=1), P("v_km_s", 0, 1, decimals=2),
       P("r1_km", 0, 1, integer=True), P("r2_km", 0, 1, integer=True), P("r3_km", 0, 1, integer=True),
       P("a1", 0, 1, digits=3), P("a2", 0, 1, digits=3))


def _at_q(p: dict[str, float]) -> float:
    return math.pi * p["f_hz"] * (p["r2_km"] - p["r1_km"]) / (
        p["v_km_s"] * math.log((p["a1"] * p["r1_km"]) / (p["a2"] * p["r2_km"])))


def _at(p: dict[str, float]) -> float:
    """A(f2, r3) = S·(1/r3)·exp(−π f2 r3/(Q v)), with the flat source level S from station 1 at f."""
    qv = _at_q(p) * p["v_km_s"]
    source = p["a1"] * p["r1_km"] * math.exp(math.pi * p["f_hz"] * p["r1_km"] / qv)
    return source / p["r3_km"] * math.exp(-math.pi * p["f2_hz"] * p["r3_km"] / qv)


def _at_make(rng: random.Random) -> dict[str, float]:
    while True:
        f, v, q = rng.uniform(1.0, 6.0), rng.uniform(3.2, 3.9), rng.uniform(150, 900)
        f2 = f * rng.uniform(1.5, 3.0)
        r1 = rng.uniform(20, 60)
        r2 = r1 + rng.uniform(50, 130)
        r3 = r2 + rng.uniform(30, 160)
        a1 = 10 ** rng.uniform(0.5, 2.5)
        a2 = a1 * (r1 / r2) * math.exp(-math.pi * f * (r2 - r1) / (q * v))
        p = _rounded(_AT, {"f_hz": f, "f2_hz": f2, "v_km_s": v, "r1_km": r1, "r2_km": r2, "r3_km": r3, "a1": a1,
                           "a2": a2})
        try:
            qe = _at_q(p)
        except (ValueError, ZeroDivisionError):
            continue
        if not 0 < qe:
            continue
        decay = math.pi * (p["f2_hz"] * p["r3_km"] - p["f_hz"] * p["r1_km"]) / (qe * p["v_km_s"])
        if (abs(qe - q) / q < 0.08 and decay <= 3.5 and _at(p) >= 0.05
                and _stable(_at, p, ("a2",), 10 ** (math.floor(math.log10(p["a2"])) - 2), 0.012)):
            return p


_long(Family(
    "attenuation_then_amplitude", "attenuation", _at, _AT, "",
    (
        "The {f_hz} Hz spectral amplitude of S waves from one earthquake is {a1} (arbitrary units) at a station "
        "{r1_km} km away and {a2} at a station {r2_km} km away, on the same azimuth. Assume body-wave geometric "
        "spreading 1/r, a frequency-independent Q, an S speed of {v_km_s} km/s and a flat source spectrum (the "
        "same source level at every frequency), so that A(f, r) = S·(1/r)·exp(−π·f·r / (Q·v)). Estimate Q from "
        "the two stations, then predict the {f2_hz} Hz amplitude (same units) at a third station {r3_km} km away "
        "on that azimuth.",
        "Two stations on one azimuth measure S-wave spectral amplitudes at {f_hz} Hz of {a1} at {r1_km} km and "
        "{a2} at {r2_km} km. Model the amplitude as A(f, r) = S·(1/r)·exp(−π f r/(Q v)) with v = {v_km_s} km/s, "
        "a constant Q and a source level S that does not depend on frequency. Correct both amplitudes for "
        "spreading, solve for Q, recover S, and give the expected amplitude at {f2_hz} Hz and {r3_km} km in the "
        "same units.",
    ),
    steps=7, rel_tol=0.03, invariant=("v_km_s",), make=_at_make,
))


# =============================================================================================
# 6. Great-circle distance from coordinates, travel time, origin time (6 steps)
# =============================================================================================
_GC = (P("lat_e", 0, 1, decimals=2), P("lon_e", 0, 1, decimals=2), P("lat_s", 0, 1, decimals=2),
       P("lon_s", 0, 1, decimals=2), P("a_s", 0, 1, decimals=1), P("b_s_per_deg", 0, 1, decimals=2),
       P("tarr_s", 0, 1, decimals=1))


def _gc_delta(p: dict[str, float]) -> float:
    la, lb = math.radians(p["lat_e"]), math.radians(p["lat_s"])
    dlon = math.radians(p["lon_s"] - p["lon_e"])
    c = math.sin(la) * math.sin(lb) + math.cos(la) * math.cos(lb) * math.cos(dlon)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _gc(p: dict[str, float]) -> float:
    return p["tarr_s"] - (p["a_s"] + p["b_s_per_deg"] * _gc_delta(p))


def _gc_make(rng: random.Random) -> dict[str, float]:
    while True:
        lat_e, lon_e = rng.uniform(-60, 60), rng.uniform(-180, 180)
        az, dist = math.radians(rng.uniform(0, 360)), math.radians(rng.uniform(25, 85))
        la = math.radians(lat_e)
        lb = math.asin(math.sin(la) * math.cos(dist) + math.cos(la) * math.sin(dist) * math.cos(az))
        lon_s = lon_e + math.degrees(math.atan2(math.sin(az) * math.sin(dist) * math.cos(la),
                                                math.cos(dist) - math.sin(la) * math.sin(lb)))
        lon_s = (lon_s + 180) % 360 - 180
        a, b = rng.uniform(150, 260), rng.uniform(6.5, 9.0)          # a straight-line fit valid for 25–85°
        t0 = rng.uniform(0, 59)
        p = _rounded(_GC, {"lat_e": lat_e, "lon_e": lon_e, "lat_s": math.degrees(lb), "lon_s": lon_s,
                           "a_s": a, "b_s_per_deg": b, "tarr_s": 0.0})
        p["tarr_s"] = round(t0 + p["a_s"] + p["b_s_per_deg"] * _gc_delta(p) + 600.0, 1)   # seconds after 10:00
        if 25 <= _gc_delta(p) <= 85 and _stable(_gc, p, ("lat_e", "lon_e", "lat_s", "lon_s"), 0.01, 0.002):
            return p


_long(Family(
    "great_circle_origin_time", "teleseismic timing", _gc, _GC, "s",
    (
        "An earthquake at {lat_e}° latitude, {lon_e}° longitude is recorded by a station at {lat_s}°, {lon_s}° "
        "(degrees, north and east positive, on a spherical Earth). Over 25°–85° the P travel time is approximated "
        "by t = {a_s} s + {b_s_per_deg} s per degree of epicentral distance. The P wave arrives {tarr_s} s after "
        "10:00:00 UTC. Compute the great-circle distance with the spherical law of cosines, then give the origin "
        "time as seconds after 10:00:00 UTC.",
        "Epicentre ({lat_e}° N, {lon_e}° E), station ({lat_s}° N, {lon_s}° E); negative values are south or west, "
        "and the Earth is a sphere. For epicentral distances of 25°–85° use t_P = {a_s} + {b_s_per_deg}·Δ "
        "(seconds, Δ in degrees). The P onset is at {tarr_s} s after 10:00:00 UTC. Find Δ from the spherical law of "
        "cosines and report the origin time in seconds after 10:00:00 UTC.",
    ),
    steps=6, rel_tol=0.0, abs_tol=1.5, make=_gc_make,
))
