"""Harder T1 families (bank "hard", decided 2026-10-27): the base 9B solves 91.6% of D-Num, whose families are
one- to three-step textbook calculations. These chain more steps, invert a measurement for a model parameter,
trace rays through layered media, or reduce redundant data (a least-squares line, a sample of magnitudes).

Conventions are stated in the question, as in ``seismology``, so every problem has one answer. Where rounding
the shown values makes the data slightly inconsistent (three S−P circles that do not meet in one point), the
question names the method, and the solver applies it to the shown values.

``split="heldout"`` families build the harder eval slice (D-Num-hard), never training data. They are other
kinds of problem than every training family here and in ``seismology``: receiver-function crustal thickness
and vp/vs (Ps with PpPs), rays in a linear velocity gradient, moment from a spectral plateau, distance from a
ground-motion model, rupture length from directivity and Coulomb stress change. GPT-6 Luna's wording check
(2026-10-27) found the Coulomb shear sense unstated and one H-κ template unclear about what to report; both
were reworded and checked again. Their definitions are not published: they live in
``seismology_hard_heldout.py``, kept out of git (README, "Data & licensing").
"""

from __future__ import annotations

import dataclasses
import math
import random

from dpt.data.gen.families import Family, Param, register

P = Param
DEG_KM = 111.19                          # km per degree of arc, stated where used
HARD = "hard"


def _hard(f: Family) -> Family:
    return register(dataclasses.replace(f, bank=HARD))


def _rounded(params: tuple[Param, ...], raw: dict[str, float]) -> dict[str, float]:
    return {q.name: q.round(raw[q.name]) for q in params}


def _vertical_slowness(v: float, p: float) -> float:
    """sqrt(1/v² − p²): vertical slowness of a ray with horizontal slowness p in a layer of velocity v."""
    return math.sqrt(1.0 / (v * v) - p * p)


# =============================================================================================
# Location from three stations (training)
# =============================================================================================
_LOC = (P("xb_km", 0, 1, integer=True), P("xc_km", 0, 1, integer=True), P("yc_km", 0, 1, integer=True),
        P("sp_a_s", 0, 1, decimals=2), P("sp_b_s", 0, 1, decimals=2), P("sp_c_s", 0, 1, decimals=2),
        P("vp_km_s", 0, 1, digits=3), P("vs_km_s", 0, 1, digits=3),
        P("xt_km", 0, 1, integer=True), P("yt_km", 0, 1, integer=True))


def _epicentre(p: dict[str, float]) -> tuple[float, float]:
    """Subtract the circle equations of B and of C from that of A (A at the origin, B on the x axis)."""
    k = p["vp_km_s"] * p["vs_km_s"] / (p["vp_km_s"] - p["vs_km_s"])
    ra, rb, rc = (k * p[s] for s in ("sp_a_s", "sp_b_s", "sp_c_s"))
    x = (ra * ra - rb * rb + p["xb_km"] ** 2) / (2 * p["xb_km"])
    y = ((ra * ra - rc * rc + p["xc_km"] ** 2 + p["yc_km"] ** 2) / 2 - x * p["xc_km"]) / p["yc_km"]
    return x, y


def _town_distance(p: dict[str, float]) -> float:
    x, y = _epicentre(p)
    return math.hypot(x - p["xt_km"], y - p["yt_km"])


def _loc_make(rng: random.Random) -> dict[str, float]:
    while True:
        vp = rng.uniform(5.8, 6.6)
        vs = vp / rng.uniform(1.68, 1.80)
        xb, xc, yc = rng.uniform(40, 90), rng.uniform(10, 60), rng.uniform(40, 90)
        x0, y0 = rng.uniform(-10, 80), rng.uniform(-10, 80)
        k = 1 / vs - 1 / vp
        raw = {"xb_km": xb, "xc_km": xc, "yc_km": yc, "vp_km_s": vp, "vs_km_s": vs,
               "sp_a_s": math.hypot(x0, y0) * k, "sp_b_s": math.hypot(x0 - xb, y0) * k,
               "sp_c_s": math.hypot(x0 - xc, y0 - yc) * k,
               "xt_km": rng.uniform(-50, 150), "yt_km": rng.uniform(-50, 150)}
        p = _rounded(_LOC, raw)
        x, y = _epicentre(p)                     # rounding must not move the epicentre much
        if math.hypot(x - x0, y - y0) < 1.5 and math.hypot(x - p["xt_km"], y - p["yt_km"]) >= 40:
            return p


_hard(Family(
    "epicentre_from_three_sp", "earthquake location", _town_distance, _LOC, "km",
    (
        "Three stations record a shallow earthquake. In a local map frame (x east, y north, in km) station A is "
        "at the origin, station B at ({xb_km}, 0) and station C at ({xc_km}, {yc_km}). The S−P times are "
        "{sp_a_s} s at A, {sp_b_s} s at B and {sp_c_s} s at C. Take vp = {vp_km_s} km/s, vs = {vs_km_s} km/s, "
        "straight rays and a source at the surface. Locate the epicentre by subtracting the circle equations of "
        "B and of C from that of A, then give its distance in km from a town at ({xt_km}, {yt_km}).",
        "A local network has station A at (0, 0), B at ({xb_km}, 0) and C at ({xc_km}, {yc_km}), coordinates in "
        "km with x east and y north. An earthquake at negligible depth produces S−P intervals of {sp_a_s} s, "
        "{sp_b_s} s and {sp_c_s} s at A, B and C. With P and S speeds of {vp_km_s} and {vs_km_s} km/s, convert "
        "each interval to a distance, find the epicentre from the two linear equations obtained by subtracting "
        "the B and C circle equations from A's, and report how far (km) it lies from the point ({xt_km}, "
        "{yt_km}).",
    ),
    steps=5, rel_tol=0.02, make=_loc_make,
))


# =============================================================================================
# Plane-wave slowness from a three-station array (training)
# =============================================================================================
_ARR = (P("xb_km", 0, 1, decimals=1), P("yb_km", 0, 1, decimals=1), P("xc_km", 0, 1, decimals=1),
        P("yc_km", 0, 1, decimals=1), P("tb_s", 0, 1, decimals=3), P("tc_s", 0, 1, decimals=3))


def _slowness(p: dict[str, float]) -> tuple[float, float]:
    """(east, north) slowness in s/km from the arrival times at B and C relative to A at the origin."""
    det = p["xb_km"] * p["yc_km"] - p["xc_km"] * p["yb_km"]
    sx = (p["tb_s"] * p["yc_km"] - p["tc_s"] * p["yb_km"]) / det
    sy = (p["xb_km"] * p["tc_s"] - p["xc_km"] * p["tb_s"]) / det
    return sx, sy


def _backazimuth(p: dict[str, float]) -> float:
    sx, sy = _slowness(p)
    return math.degrees(math.atan2(-sx, -sy)) % 360.0     # towards the source: against the propagation


def _apparent_velocity(p: dict[str, float]) -> float:
    return 1.0 / math.hypot(*_slowness(p))


def _arr_make(rng: random.Random) -> dict[str, float]:
    while True:
        baz, vapp = rng.uniform(10, 350), rng.uniform(6.0, 18.0)
        prop = math.radians(baz + 180)
        sx, sy = math.sin(prop) / vapp, math.cos(prop) / vapp
        pts = [(rng.uniform(-8, 8), rng.uniform(-8, 8)) for _ in range(2)]
        raw = {"xb_km": pts[0][0], "yb_km": pts[0][1], "xc_km": pts[1][0], "yc_km": pts[1][1],
               "tb_s": sx * pts[0][0] + sy * pts[0][1], "tc_s": sx * pts[1][0] + sy * pts[1][1]}
        p = _rounded(_ARR, raw)
        det = p["xb_km"] * p["yc_km"] - p["xc_km"] * p["yb_km"]
        if min(math.hypot(*pt) for pt in pts) < 3 or abs(det) < 20:
            continue                             # stations too close to A or nearly collinear
        b = _backazimuth(p)
        if abs((b - baz + 180) % 360 - 180) < 1.0 and 3 < b < 357 and abs(_apparent_velocity(p) / vapp - 1) < 0.01:
            return p


_ARRAY_SETUP = (
    "A small array has three stations; in a local frame (x east, y north, in km) station A is at the origin, B "
    "at ({xb_km}, {yb_km}) and C at ({xc_km}, {yc_km}). A teleseismic P wave, treated as a plane wave, reaches B "
    "{tb_s} s and C {tc_s} s after A (a negative delay means it arrives earlier). ",
    "Three array stations sit at A = (0, 0), B = ({xb_km}, {yb_km}) and C = ({xc_km}, {yc_km}), in km east and "
    "north. A plane P wave crossing the array arrives at B with a delay of {tb_s} s and at C with a delay of "
    "{tc_s} s relative to A (negative = earlier). ",
)

_hard(Family(
    "array_backazimuth", "seismic arrays", _backazimuth, _ARR, "°",
    (
        _ARRAY_SETUP[0] + "From the horizontal slowness vector, what is the back-azimuth to the source, in degrees "
        "clockwise from north?",
        _ARRAY_SETUP[1] + "Solve for the horizontal slowness vector and give the back-azimuth (the direction from "
        "the array towards the source) in degrees, measured clockwise from north.",
    ),
    steps=4, rel_tol=0.0, abs_tol=1.0, make=_arr_make,
))

_hard(Family(
    "array_apparent_velocity", "seismic arrays", _apparent_velocity, _ARR, "km/s",
    (
        _ARRAY_SETUP[0] + "What is the apparent horizontal velocity of the wave across the array, in km/s?",
        _ARRAY_SETUP[1] + "Solve for the horizontal slowness vector; what apparent velocity (km/s) does it imply?",
    ),
    steps=4, rel_tol=0.01, make=_arr_make,
))


# =============================================================================================
# Refraction through three flat layers, forward and inverse (training)
# =============================================================================================
def _head3(p: dict[str, float]) -> float:
    v1, v2, v3 = p["v1_km_s"], p["v2_km_s"], p["v3_km_s"]
    return (p["distance_km"] / v3 + 2 * p["h1_km"] * math.sqrt(1 / v1 ** 2 - 1 / v3 ** 2)
            + 2 * p["h2_km"] * math.sqrt(1 / v2 ** 2 - 1 / v3 ** 2))


def _head3_valid(p: dict[str, float]) -> bool:
    v1, v2, v3 = p["v1_km_s"], p["v2_km_s"], p["v3_km_s"]
    if not v1 < v2 < v3 - 0.3:
        return False
    xcrit = 2 * p["h1_km"] * math.tan(math.asin(v1 / v3)) + 2 * p["h2_km"] * math.tan(math.asin(v2 / v3))
    return p["distance_km"] > xcrit + 10


_V1_3L, _V2_3L, _V3_3L = (P("v1_km_s", 2.5, 5.5, digits=3), P("v2_km_s", 5.6, 7.0, digits=3),
                          P("v3_km_s", 7.6, 8.4, digits=3))

_hard(Family(
    "three_layer_head_wave_time", "refraction seismology", _head3,
    (P("distance_km", 80, 300), P("h1_km", 1.0, 8.0), P("h2_km", 8.0, 30.0), _V1_3L, _V2_3L, _V3_3L), "s",
    (
        "A crust of flat layers has a top layer {h1_km} km thick with P velocity {v1_km_s} km/s, a second layer "
        "{h2_km} km thick at {v2_km_s} km/s, and a half-space at {v3_km_s} km/s below. For a surface source, what "
        "is the travel time (s) of the P head wave refracted along the top of the half-space, recorded at a "
        "distance of {distance_km} km?",
        "Model: layer 1, thickness {h1_km} km, vp = {v1_km_s} km/s; layer 2, thickness {h2_km} km, vp = "
        "{v2_km_s} km/s; half-space vp = {v3_km_s} km/s; all interfaces horizontal. At what time (s) after a "
        "surface shot does the head wave from the half-space arrive {distance_km} km away?",
    ),
    steps=4, valid=_head3_valid,
))

_INT = (P("v1_km_s", 0, 1, digits=3), P("v2_km_s", 0, 1, digits=3), P("v3_km_s", 0, 1, digits=3),
        P("ti2_s", 0, 1, digits=3), P("ti3_s", 0, 1, digits=3))


def _h2_from_intercepts(p: dict[str, float]) -> float:
    v1, v2, v3 = p["v1_km_s"], p["v2_km_s"], p["v3_km_s"]
    h1 = p["ti2_s"] / (2 * math.sqrt(1 / v1 ** 2 - 1 / v2 ** 2))
    return (p["ti3_s"] - 2 * h1 * math.sqrt(1 / v1 ** 2 - 1 / v3 ** 2)) / (2 * math.sqrt(1 / v2 ** 2 - 1 / v3 ** 2))


def _int_make(rng: random.Random) -> dict[str, float]:
    while True:
        v1, v2, v3 = rng.uniform(2.5, 5.0), rng.uniform(5.6, 7.0), rng.uniform(7.6, 8.4)
        h1, h2 = rng.uniform(1.0, 6.0), rng.uniform(6.0, 25.0)
        raw = {"v1_km_s": v1, "v2_km_s": v2, "v3_km_s": v3,
               "ti2_s": 2 * h1 * math.sqrt(1 / v1 ** 2 - 1 / v2 ** 2),
               "ti3_s": 2 * h1 * math.sqrt(1 / v1 ** 2 - 1 / v3 ** 2) + 2 * h2 * math.sqrt(1 / v2 ** 2 - 1 / v3 ** 2)}
        p = _rounded(_INT, raw)
        if p["v1_km_s"] < p["v2_km_s"] < p["v3_km_s"] and abs(_h2_from_intercepts(p) / h2 - 1) < 0.03:
            return p


_hard(Family(
    "second_layer_thickness_from_intercepts", "refraction seismology", _h2_from_intercepts, _INT, "km",
    (
        "A refraction survey over three flat layers measures P velocities of {v1_km_s}, {v2_km_s} and {v3_km_s} "
        "km/s from the top down. The head-wave branch of the second layer extrapolates to an intercept time of "
        "{ti2_s} s at zero offset, and the head-wave branch of the third layer to {ti3_s} s. How thick, in km, "
        "is the second layer?",
        "Travel-time curves from a surface shot show three segments with slopes corresponding to {v1_km_s}, "
        "{v2_km_s} and {v3_km_s} km/s. The second segment's intercept time is {ti2_s} s and the third's is "
        "{ti3_s} s. Assuming horizontal layers, find the thickness (km) of the middle layer.",
    ),
    steps=4, rel_tol=0.02, make=_int_make,
))


# =============================================================================================
# Dipping refractor from reversed profiles (training)
# =============================================================================================
_DIP = (P("v1_km_s", 0, 1, digits=3), P("vd_km_s", 0, 1, digits=3), P("vu_km_s", 0, 1, digits=3))


def _dip_angles(p: dict[str, float]) -> tuple[float, float]:
    """(critical angle, dip) in radians from the down-dip and up-dip apparent velocities."""
    a, b = math.asin(p["v1_km_s"] / p["vd_km_s"]), math.asin(p["v1_km_s"] / p["vu_km_s"])
    return (a + b) / 2, (a - b) / 2


def _dip_make(rng: random.Random) -> dict[str, float]:
    while True:
        if rng.random() < 0.5:                   # sediments over bedrock, or crust over mantle
            v1 = rng.uniform(1.6, 3.0)
            v2 = rng.uniform(max(1.3 * v1, 2.5), min(2.2 * v1, 6.0))
        else:
            v1, v2 = rng.uniform(5.0, 6.4), rng.uniform(7.6, 8.4)
        dip = math.radians(rng.uniform(3.0, 15.0))
        ic = math.asin(v1 / v2)
        if ic - dip < math.radians(8):
            continue
        raw = {"v1_km_s": v1, "vd_km_s": v1 / math.sin(ic + dip), "vu_km_s": v1 / math.sin(ic - dip)}
        p = _rounded(_DIP, raw)
        if p["v1_km_s"] < p["vd_km_s"] < p["vu_km_s"]:
            return p


_DIP_SETUP = (
    "Reversed refraction profiles over a dipping planar interface give apparent head-wave velocities of "
    "{vd_km_s} km/s when shooting down-dip and {vu_km_s} km/s when shooting up-dip. The upper layer's P velocity "
    "is {v1_km_s} km/s. ",
    "A seismic refraction line is shot in both directions over an inclined refractor. The upper layer has a "
    "velocity of {v1_km_s} km/s; the head wave's apparent velocity is {vd_km_s} km/s for the down-dip shot and "
    "{vu_km_s} km/s for the up-dip shot. ",
)

_hard(Family(
    "dipping_refractor_velocity", "refraction seismology", lambda p: p["v1_km_s"] / math.sin(_dip_angles(p)[0]),
    _DIP, "km/s",
    (_DIP_SETUP[0] + "What is the true P velocity of the lower layer, in km/s?",
     _DIP_SETUP[1] + "Find the true velocity (km/s) of the refracting layer."),
    steps=3, rel_tol=0.01, make=_dip_make,
))

_hard(Family(
    "dipping_refractor_dip", "refraction seismology", lambda p: math.degrees(_dip_angles(p)[1]), _DIP, "°",
    (_DIP_SETUP[0] + "What is the dip of the interface, in degrees?",
     _DIP_SETUP[1] + "By how many degrees does the refractor dip?"),
    steps=3, rel_tol=0.02, abs_tol=0.2, make=_dip_make,
))


# =============================================================================================
# Gutenberg–Richter from a sample of magnitudes (training)
# =============================================================================================
_GR_M = tuple(P(f"m{i}", 0, 1, decimals=1) for i in range(1, 11))
_GR = (*_GR_M, P("mc", 0, 1, decimals=1), P("n_total", 0, 1, integer=True), P("m_target", 0, 1, decimals=1))


def _gr_b(p: dict[str, float]) -> float:
    mean = sum(p[q.name] for q in _GR_M) / len(_GR_M)
    return math.log10(math.e) / (mean - (p["mc"] - 0.05))


def _gr_count(p: dict[str, float]) -> float:
    return p["n_total"] * 10 ** (-_gr_b(p) * (p["m_target"] - p["mc"]))


def _gr_make(rng: random.Random) -> dict[str, float]:
    while True:
        b, mc = rng.uniform(0.8, 1.3), round(rng.uniform(1.5, 3.0), 1)
        beta = b * math.log(10)
        mags = [round(mc - 0.05 + rng.expovariate(beta) + 1e-9, 1) for _ in range(10)]   # 0.1 bins from Mc
        raw = {**{f"m{i + 1}": max(mc, m) for i, m in enumerate(mags)}, "mc": mc,
               "n_total": rng.uniform(200, 5000), "m_target": mc + rng.uniform(2.0, 3.5)}
        p = _rounded(_GR, raw)
        if len(set(mags)) >= 4 and 0.6 <= _gr_b(p) <= 1.8 and _gr_count(p) >= 0.3:
            return p


_GR_LIST = ", ".join("{m%d}" % i for i in range(1, 11))

_hard(Family(
    "gr_count_from_magnitude_sample", "statistical seismology", _gr_count, _GR, "",
    (
        "A catalogue is complete above Mc = {mc}, with magnitudes binned to 0.1. A random sample of ten of its "
        "events has magnitudes " + _GR_LIST + ". Estimate b with the Aki–Utsu formula b = log10(e)/(M̄ − (Mc − "
        "ΔM/2)), ΔM = 0.1, using the sample mean M̄. If the catalogue holds {n_total} events with M ≥ Mc, how "
        "many events with M ≥ {m_target} does the Gutenberg–Richter law with this b predict?",
        "Ten events drawn at random from a catalogue that is complete for M ≥ {mc} (0.1-unit bins) have "
        "magnitudes " + _GR_LIST + ". Use their mean in b = log10(e)/(M̄ − Mc + 0.05). The catalogue contains "
        "{n_total} events at or above Mc; with that b-value, how many should reach magnitude {m_target} or "
        "more?",
    ),
    steps=3, rel_tol=0.02, scaling={"n_total": 1}, make=_gr_make,
))


# =============================================================================================
# Poisson's ratio from a least-squares Wadati fit (training)
# =============================================================================================
_WAD4 = tuple(P(f"{k}{i}_s", 0, 1, decimals=2) for i in range(1, 5) for k in ("tp", "sp"))


def _wadati_fit_ratio(p: dict[str, float]) -> float:
    tp = [p[f"tp{i}_s"] for i in range(1, 5)]
    sp = [p[f"sp{i}_s"] for i in range(1, 5)]
    mt, ms = sum(tp) / 4, sum(sp) / 4
    slope = sum((a - mt) * (b - ms) for a, b in zip(tp, sp)) / sum((a - mt) ** 2 for a in tp)
    return 1 + slope


def _poisson_from_wadati(p: dict[str, float]) -> float:
    r = _wadati_fit_ratio(p)
    return (r * r - 2) / (2 * (r * r - 1))


def _wad4_make(rng: random.Random) -> dict[str, float]:
    while True:
        t0, r, vp = rng.uniform(0, 20), rng.uniform(1.65, 1.90), rng.uniform(5.8, 6.6)
        ds = sorted(rng.uniform(15, 250) for _ in range(4))
        if min(b - a for a, b in zip(ds, ds[1:])) < 15:
            continue
        raw = {}
        for i, d in enumerate(ds, 1):
            raw[f"tp{i}_s"] = t0 + d / vp
            raw[f"sp{i}_s"] = (r - 1) * d / vp + rng.uniform(-0.15, 0.15)
        p = _rounded(_WAD4, raw)
        if 1.55 <= _wadati_fit_ratio(p) <= 2.0:
            return p


_WAD4_DATA = "; ".join("station %d: P at {tp%d_s} s, S−P {sp%d_s} s" % (i, i, i) for i in range(1, 5))

_hard(Family(
    "poisson_ratio_from_wadati_fit", "earthquake location", _poisson_from_wadati, _WAD4, "",
    (
        "Four stations record a local earthquake on a common clock (origin time unknown): " + _WAD4_DATA + ". "
        "Fit a least-squares line to the Wadati diagram (S−P interval against P arrival time), take vp/vs = 1 + "
        "slope, and give Poisson's ratio ν = (r² − 2)/(2(r² − 1)) with r = vp/vs.",
        "P arrival times and S−P intervals at four stations are — " + _WAD4_DATA + ". Using the least-squares "
        "slope of S−P against P arrival time (vp/vs = 1 + slope), compute Poisson's ratio of the crust, "
        "ν = (r² − 2)/(2(r² − 1)), r = vp/vs.",
    ),
    steps=4, rel_tol=0.0, abs_tol=0.005, make=_wad4_make,
))


# =============================================================================================
# Normal-incidence energy through two interfaces (training)
# =============================================================================================
def _two_interface_energy(p: dict[str, float]) -> float:
    z1, z2, z3 = (p[f"rho{i}_kg_m3"] * p[f"v{i}_m_s"] for i in (1, 2, 3))
    r12, r23 = ((z2 - z1) / (z2 + z1)) ** 2, ((z3 - z2) / (z3 + z2)) ** 2
    return (1 - r12) ** 2 * r23


def _two_interface_valid(p: dict[str, float]) -> bool:
    z1, z2, z3 = (p[f"rho{i}_kg_m3"] * p[f"v{i}_m_s"] for i in (1, 2, 3))
    return ((z3 - z2) / (z3 + z2)) ** 2 >= 0.003 and ((z2 - z1) / (z2 + z1)) ** 2 >= 0.003


_hard(Family(
    "second_interface_reflected_energy", "reflection seismology", _two_interface_energy,
    (P("rho1_kg_m3", 1800, 2400), P("v1_m_s", 1600, 2800), P("rho2_kg_m3", 2100, 2600), P("v2_m_s", 2400, 4200),
     P("rho3_kg_m3", 2300, 2800), P("v3_m_s", 3000, 6000)), "",
    (
        "A plane P wave travelling down through layer 1 (density {rho1_kg_m3} kg/m³, velocity {v1_m_s} m/s) meets, "
        "at normal incidence, layer 2 ({rho2_kg_m3} kg/m³, {v2_m_s} m/s) and below it layer 3 ({rho3_kg_m3} "
        "kg/m³, {v3_m_s} m/s). Ignoring attenuation, spreading and multiples, what fraction of the incident energy "
        "comes back up into layer 1 as the reflection from the layer 2–3 interface?",
        "Normal-incidence energy budget: layer 1 has ρ = {rho1_kg_m3} kg/m³ and v = {v1_m_s} m/s, layer 2 has "
        "ρ = {rho2_kg_m3} kg/m³ and v = {v2_m_s} m/s, layer 3 has ρ = {rho3_kg_m3} kg/m³ and v = {v3_m_s} m/s. "
        "The wave crosses the 1–2 interface down and back up. What fraction of the original energy returns to "
        "layer 1 from the 2–3 reflection (no multiples, attenuation or spreading)?",
    ),
    steps=4, rel_tol=0.01, valid=_two_interface_valid,
))


# =============================================================================================
# Q from the spectral ratio at two frequencies (training)
# =============================================================================================
_QSR = (P("dt_s", 0, 1, digits=3), P("fa_hz", 0, 1, digits=2), P("ra", 0, 1, digits=3), P("fb_hz", 0, 1, digits=2),
        P("rb", 0, 1, digits=3))


def _q_spectral(p: dict[str, float]) -> float:
    return math.pi * p["dt_s"] * (p["fb_hz"] - p["fa_hz"]) / math.log(p["ra"] / p["rb"])


def _qsr_make(rng: random.Random) -> dict[str, float]:
    while True:
        q, dt, c = math.exp(rng.uniform(math.log(80), math.log(800))), rng.uniform(5, 60), rng.uniform(0.3, 0.9)
        fa = rng.uniform(0.5, 3.0)
        fb = fa + rng.uniform(2.0, 8.0)
        x = math.pi * (fb - fa) * dt / q
        if not 0.3 <= x <= 4.0:
            continue
        raw = {"dt_s": dt, "fa_hz": fa, "fb_hz": fb, "ra": c * math.exp(-math.pi * fa * dt / q),
               "rb": c * math.exp(-math.pi * fb * dt / q)}
        p = _rounded(_QSR, raw)
        if p["rb"] >= 0.005 and p["fb_hz"] > p["fa_hz"] and abs(_q_spectral(p) / q - 1) < 0.03:
            return p


_hard(Family(
    "q_from_spectral_ratio", "attenuation", _q_spectral, _QSR, "",
    (
        "Two stations lie on the same ray path, and the S wave takes {dt_s} s longer to reach the far one. After "
        "correcting for geometric spreading, the far-to-near spectral amplitude ratio is {ra} at {fa_hz} Hz and "
        "{rb} at {fb_hz} Hz. Assuming A(f) ∝ exp(−πft/Q) with frequency-independent Q and site terms that do not "
        "depend on frequency, what is Q between the two stations?",
        "Spectral ratios of the same S arrival at two stations (far over near, spreading removed) are {ra} at "
        "{fa_hz} Hz and {rb} at {fb_hz} Hz. The extra travel time to the far station is {dt_s} s. For "
        "attenuation of the form exp(−πft/Q), with constant Q and frequency-independent site factors, estimate "
        "Q.",
    ),
    steps=4, rel_tol=0.02, scaling={"dt_s": 1}, make=_qsr_make,
))


# =============================================================================================
# Source depth from the pP − P delay (training)
# =============================================================================================
def _pp_depth(p: dict[str, float]) -> float:
    return p["dt_s"] / (2 * _vertical_slowness(p["v_km_s"], p["p_s_deg"] / DEG_KM))


_hard(Family(
    "depth_from_pp_delay", "global seismology", _pp_depth,
    (P("dt_s", 3.0, 60.0), P("p_s_deg", 4.5, 8.5, digits=3), P("v_km_s", 5.8, 8.0, digits=3)), "km",
    (
        "At a teleseismic station the depth phase pP arrives {dt_s} s after P. The P ray parameter is {p_s_deg} "
        "s/° (convert with 111.19 km per degree and treat the Earth as flat near the source), and the P velocity "
        "above the source is {v_km_s} km/s. If pP leaves the source upward with the same ray parameter and "
        "reflects at the free surface, how deep is the earthquake, in km?",
        "The pP − P delay on a distant seismogram is {dt_s} s. With a ray parameter of {p_s_deg} s/° (111.19 km "
        "per degree, flat-Earth geometry near the source) and a uniform P velocity of {v_km_s} km/s between the "
        "source and the surface, estimate the focal depth (km).",
    ),
    steps=3, rel_tol=0.02, scaling={"dt_s": 1},
))


# =============================================================================================
# Held-out section: not published. The 8 held-out T1 families behind D-Num-hard live in
# seismology_hard_heldout.py, which stays out of git so that the test problems cannot be regenerated or
# trained on (README, "Data & licensing"). It is imported at the end of this module when present.
# =============================================================================================


# The held-out D-Num-hard families are not published; they register themselves when their private module is present.
# Imported last, because they use the helpers above (dpt.data.gen.common.import_private_module_if_present).
from dpt.data.gen.common import import_private_module_if_present  # noqa: E402

heldout_dnum_hard_families_present = import_private_module_if_present("dpt.data.gen.seismology_hard_heldout")
