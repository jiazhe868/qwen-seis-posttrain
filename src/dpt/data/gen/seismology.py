"""Seismology problem families for T1 (training) and D-Num (held out). Registered on import.

Conventions that differ between textbooks are stated in the question (the moment-magnitude constant, the
energy–magnitude relation, the local-magnitude formula, Brune's source radius), so every problem has one
correct answer. Where a sign convention would make the answer ambiguous, the question asks for a quantity
without one (the reflected energy fraction, not the reflection coefficient).

Held-out families (``split="heldout"``) are different kinds of problem, not rewordings of training ones:
tsunami travel time, Omori aftershock counts, the Fresnel zone, Dix interval velocity, Brune corner
frequency, earthquake recurrence from slip rate, ground strain, Young's modulus, exceedance probability
and the surface-wave magnitude. Some combine steps the training families teach (moment from magnitude,
the Gutenberg–Richter rate); none appears in training as a whole. Their definitions are not published:
they live in ``seismology_heldout.py``, kept out of git so that the D-Num problems cannot be regenerated
(README, "Data & licensing").
"""

from __future__ import annotations

import math
import random

from dpt.data.gen.common import MW_RULE
from dpt.data.gen.families import Family, Param, register

R_EARTH_KM = 6371.0
G = 9.81
LOG10_E = math.log10(math.e)

P = Param


def _m0(mw: float) -> float:
    """Seismic moment [N·m] from Mw, IASPEI standard: Mw = (2/3)(log10 M0 − 9.1)."""
    return 10 ** (1.5 * mw + 9.1)


def _mw(m0: float) -> float:
    return (2.0 / 3.0) * (math.log10(m0) - 9.1)


def _ratio_ok(lo: float = 1.6, hi: float = 1.9):
    return lambda p: lo <= p["vp_km_s"] / p["vs_km_s"] <= hi


def _haversine_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    f1, f2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    h = math.sin((f2 - f1) / 2) ** 2 + math.cos(f1) * math.cos(f2) * math.sin(dl / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))


def _azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    f1, f2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(f2)
    x = math.cos(f1) * math.sin(f2) - math.sin(f1) * math.cos(f2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


VP = P("vp_km_s", 5.5, 8.0, digits=3)
VS = P("vs_km_s", 3.0, 4.7, digits=3)
V1_CRUST = P("v1_km_s", 5.6, 6.8, digits=3)
V2_MANTLE = P("v2_km_s", 7.6, 8.4, digits=3)
H_CRUST = P("thickness_km", 8.0, 45.0, digits=3)
LAT1, LON1 = P("lat1_deg", -60.0, 60.0, decimals=1), P("lon1_deg", -179.0, 179.0, decimals=1)
LAT2, LON2 = P("lat2_deg", -60.0, 60.0, decimals=1), P("lon2_deg", -179.0, 179.0, decimals=1)
RIGIDITY = P("rigidity_gpa", 25.0, 40.0, digits=3)
MW = P("mw", 4.0, 8.5, decimals=1)
RHO = P("density_kg_m3", 2000.0, 3400.0, digits=3)


# =============================================================================================
# Body-wave travel times and location
# =============================================================================================
register(Family(
    "direct_travel_time", "body-wave travel times",
    lambda p: p["distance_km"] / p["velocity_km_s"],
    (P("distance_km", 20, 400), P("velocity_km_s", 3.0, 8.5, digits=3)), "s",
    (
        "A P wave travels {distance_km} km through the crust at an average velocity of {velocity_km_s} km/s. "
        "Assuming a straight ray path, how long does the trip take? Give the answer in seconds.",
        "An S wave crosses {distance_km} km of rock at a constant speed of {velocity_km_s} km/s. What is its "
        "travel time in seconds?",
        "Seismic energy propagates at {velocity_km_s} km/s from a source to a station {distance_km} km away "
        "along a straight path. Find the travel time in s.",
    ),
    scaling={"distance_km": 1, "velocity_km_s": -1},
))

register(Family(
    "sp_interval", "body-wave travel times",
    lambda p: p["distance_km"] * (1 / p["vs_km_s"] - 1 / p["vp_km_s"]),
    (P("distance_km", 10, 300), VP, VS), "s",
    (
        "A station lies {distance_km} km from an earthquake along a straight path. The P-wave speed is "
        "{vp_km_s} km/s and the S-wave speed is {vs_km_s} km/s. What S−P interval does the station record, "
        "in seconds?",
        "For a source–receiver distance of {distance_km} km in a uniform medium with vp = {vp_km_s} km/s and "
        "vs = {vs_km_s} km/s, how many seconds separate the P and S arrivals?",
    ),
    scaling={"distance_km": 1}, valid=_ratio_ok(),
))

register(Family(
    "sp_hypocentral_distance", "earthquake location",
    lambda p: p["ts_minus_tp_s"] * p["vp_km_s"] * p["vs_km_s"] / (p["vp_km_s"] - p["vs_km_s"]),
    (P("ts_minus_tp_s", 2.0, 40.0), VP, VS), "km",
    (
        "A seismogram shows the S wave arriving {ts_minus_tp_s} s after the P wave. Assuming straight rays "
        "with vp = {vp_km_s} km/s and vs = {vs_km_s} km/s, how far is the station from the hypocenter, in km?",
        "The S−P time at a station is {ts_minus_tp_s} s. With P and S speeds of {vp_km_s} km/s and "
        "{vs_km_s} km/s, estimate the hypocentral distance in kilometres.",
        "Using an S-minus-P interval of {ts_minus_tp_s} s, a P velocity of {vp_km_s} km/s and an S velocity of "
        "{vs_km_s} km/s, compute the distance to the earthquake (km).",
    ),
    scaling={"ts_minus_tp_s": 1}, valid=_ratio_ok(),
))


def _epi_from_sp(p):
    r = p["ts_minus_tp_s"] * p["vp_km_s"] * p["vs_km_s"] / (p["vp_km_s"] - p["vs_km_s"])
    return math.sqrt(r * r - p["depth_km"] ** 2)


register(Family(
    "epicentral_from_sp_and_depth", "earthquake location", _epi_from_sp,
    (P("ts_minus_tp_s", 3.0, 30.0), VP, VS, P("depth_km", 5.0, 30.0, digits=3)), "km",
    (
        "An earthquake at {depth_km} km depth produces an S−P time of {ts_minus_tp_s} s at a station. "
        "Treat the crust as uniform with vp = {vp_km_s} km/s and vs = {vs_km_s} km/s and the rays as "
        "straight. What is the epicentral distance in km?",
        "A station records S−P = {ts_minus_tp_s} s from an event whose focal depth is {depth_km} km. With "
        "vp = {vp_km_s} km/s and vs = {vs_km_s} km/s in a homogeneous half-space, how far from the station "
        "(measured along the surface) is the epicenter, in kilometres?",
    ),
    steps=2,
    valid=lambda p: _ratio_ok()(p) and p["ts_minus_tp_s"] * p["vp_km_s"] * p["vs_km_s"]
    / (p["vp_km_s"] - p["vs_km_s"]) > 1.3 * p["depth_km"],
))

register(Family(
    "p_travel_time_from_sp", "earthquake location",
    lambda p: p["ts_minus_tp_s"] * p["vs_km_s"] / (p["vp_km_s"] - p["vs_km_s"]),
    (P("ts_minus_tp_s", 2.0, 40.0), VP, VS), "s",
    (
        "A station measures an S−P time of {ts_minus_tp_s} s. If vp = {vp_km_s} km/s and vs = {vs_km_s} km/s "
        "along a straight path, how many seconds after the origin time did the P wave arrive?",
        "Given S−P = {ts_minus_tp_s} s, a P speed of {vp_km_s} km/s and an S speed of {vs_km_s} km/s, find the "
        "P-wave travel time from the hypocenter to the station, in seconds.",
    ),
    steps=2, scaling={"ts_minus_tp_s": 1}, valid=_ratio_ok(),
))

_WAD = (P("tp_a_s", 0, 0, digits=4), P("sp_a_s", 0, 0, digits=3), P("tp_b_s", 0, 0, digits=4),
        P("sp_b_s", 0, 0, digits=3))


def _wadati_make(rng: random.Random) -> dict[str, float]:
    t0, r, vp = rng.uniform(0, 30), rng.uniform(1.65, 1.85), rng.uniform(5.8, 6.6)
    da, db = rng.uniform(20, 80), rng.uniform(120, 260)
    raw = {"tp_a_s": t0 + da / vp, "sp_a_s": (r - 1) * da / vp, "tp_b_s": t0 + db / vp, "sp_b_s": (r - 1) * db / vp}
    return {q.name: q.round(raw[q.name]) for q in _WAD}


register(Family(
    "wadati_vp_vs", "earthquake location",
    lambda p: 1 + (p["sp_b_s"] - p["sp_a_s"]) / (p["tp_b_s"] - p["tp_a_s"]),
    _WAD, "",
    (
        "Two stations record the same earthquake on a common clock (the origin time is unknown). At station A "
        "the P wave arrives at t = {tp_a_s} s and the S−P time is {sp_a_s} s; at station B the P wave arrives "
        "at t = {tp_b_s} s and the S−P time is {sp_b_s} s. Using a Wadati diagram (S−P time against P arrival "
        "time), estimate the vp/vs ratio.",
        "On a Wadati diagram, station A plots at P arrival {tp_a_s} s with S−P {sp_a_s} s and station B at P "
        "arrival {tp_b_s} s with S−P {sp_b_s} s. What vp/vs ratio does the line through them imply?",
    ),
    steps=2, rel_tol=0.0, abs_tol=0.02, make=_wadati_make,
))

register(Family(
    "arc_length_from_degrees", "global seismology",
    lambda p: p["delta_deg"] * math.pi / 180 * R_EARTH_KM,
    (P("delta_deg", 5.0, 120.0, decimals=1),), "km",
    (
        "A teleseismic station is {delta_deg}° from an earthquake. Taking the Earth as a sphere of radius "
        "6371 km, what is the great-circle distance in kilometres?",
        "Convert an epicentral distance of {delta_deg} degrees into kilometres along the Earth's surface "
        "(spherical Earth, R = 6371 km).",
    ),
    scaling={"delta_deg": 1},
))

register(Family(
    "great_circle_distance", "global seismology",
    lambda p: _haversine_deg(p["lat1_deg"], p["lon1_deg"], p["lat2_deg"], p["lon2_deg"]),
    (LAT1, LON1, LAT2, LON2), "°",
    (
        "An earthquake occurs at latitude {lat1_deg}°, longitude {lon1_deg}° and is recorded at a station at "
        "latitude {lat2_deg}°, longitude {lon2_deg}° (negative values are south and west). On a spherical "
        "Earth, what is the epicentral distance in degrees?",
        "Event coordinates: ({lat1_deg}°, {lon1_deg}°); station coordinates: ({lat2_deg}°, {lon2_deg}°), given "
        "as (latitude, longitude) with south and west negative. Compute the great-circle angular distance "
        "between them, in degrees.",
    ),
    steps=2, rel_tol=0.005, abs_tol=0.1,
    valid=lambda p: _haversine_deg(p["lat1_deg"], p["lon1_deg"], p["lat2_deg"], p["lon2_deg"]) > 3,
))


def _az_valid(p):
    d = _haversine_deg(p["lat1_deg"], p["lon1_deg"], p["lat2_deg"], p["lon2_deg"])
    az = _azimuth_deg(p["lat1_deg"], p["lon1_deg"], p["lat2_deg"], p["lon2_deg"])
    return 5 < d < 150 and 3 < az < 357


register(Family(
    "station_azimuth", "global seismology",
    lambda p: _azimuth_deg(p["lat1_deg"], p["lon1_deg"], p["lat2_deg"], p["lon2_deg"]),
    (LAT1, LON1, LAT2, LON2), "°",
    (
        "An earthquake at latitude {lat1_deg}°, longitude {lon1_deg}° is recorded at a station at latitude "
        "{lat2_deg}°, longitude {lon2_deg}° (south and west negative). What is the azimuth from the epicenter "
        "to the station, in degrees clockwise from north (0–360°), on a spherical Earth?",
        "Compute the initial great-circle bearing (azimuth, degrees clockwise from north, between 0 and 360) "
        "from an epicenter at ({lat1_deg}°, {lon1_deg}°) to a seismometer at ({lat2_deg}°, {lon2_deg}°); "
        "coordinates are (latitude, longitude) with south and west negative.",
    ),
    steps=2, rel_tol=0.0, abs_tol=1.0, valid=_az_valid,
))

# =============================================================================================
# Refraction, reflection and ray geometry
# =============================================================================================
register(Family(
    "snell_transmitted_angle", "ray theory",
    lambda p: math.degrees(math.asin(p["v2_km_s"] / p["v1_km_s"] * math.sin(math.radians(p["incidence_deg"])))),
    (P("v1_km_s", 2.0, 6.0, digits=3), P("v2_km_s", 3.0, 8.0, digits=3), P("incidence_deg", 5.0, 60.0, decimals=0)),
    "°",
    (
        "A P wave in a layer with velocity {v1_km_s} km/s strikes a planar interface at an angle of incidence "
        "of {incidence_deg}°. The layer below has a P velocity of {v2_km_s} km/s. What is the angle of the "
        "transmitted P wave from the normal, in degrees?",
        "Using Snell's law, find the refraction angle (degrees from the interface normal) of a ray passing "
        "from a medium with v = {v1_km_s} km/s into one with v = {v2_km_s} km/s, if it arrives at "
        "{incidence_deg}° from the normal.",
    ),
    rel_tol=0.0, abs_tol=0.3,
    valid=lambda p: abs(p["v2_km_s"] - p["v1_km_s"]) > 0.3
    and p["v2_km_s"] / p["v1_km_s"] * math.sin(math.radians(p["incidence_deg"])) < 0.97,
))

register(Family(
    "critical_angle", "ray theory",
    lambda p: math.degrees(math.asin(p["v1_km_s"] / p["v2_km_s"])),
    (V1_CRUST, V2_MANTLE), "°",
    (
        "A crustal layer with P velocity {v1_km_s} km/s overlies mantle with P velocity {v2_km_s} km/s. What is "
        "the critical angle of incidence for P waves at the Moho, in degrees?",
        "Find the critical angle (in degrees) for a P wave travelling in rock of velocity {v1_km_s} km/s that "
        "meets a faster layer of velocity {v2_km_s} km/s.",
    ),
    rel_tol=0.0, abs_tol=0.3,
))

register(Family(
    "ray_parameter", "ray theory",
    lambda p: math.sin(math.radians(p["incidence_deg"])) / p["velocity_km_s"],
    (P("incidence_deg", 5.0, 80.0, decimals=0), P("velocity_km_s", 3.0, 8.5, digits=3)), "s/km",
    (
        "A ray leaves a source in a layer with velocity {velocity_km_s} km/s at an angle of {incidence_deg}° "
        "from the vertical. What is its ray parameter in s/km?",
        "Compute the horizontal slowness (ray parameter) of a P wave travelling at {velocity_km_s} km/s with "
        "an incidence angle of {incidence_deg}° measured from the vertical. Answer in s/km.",
    ),
    scaling={"velocity_km_s": -1},
))

register(Family(
    "crossover_distance", "refraction seismology",
    lambda p: 2 * p["thickness_km"] * math.sqrt((p["v2_km_s"] + p["v1_km_s"]) / (p["v2_km_s"] - p["v1_km_s"])),
    (H_CRUST, V1_CRUST, V2_MANTLE), "km",
    (
        "A {thickness_km} km thick layer with P velocity {v1_km_s} km/s lies over a half-space with P velocity "
        "{v2_km_s} km/s. At what source–receiver distance does the head wave overtake the direct wave (the "
        "crossover distance), in km?",
        "For a single horizontal layer of thickness {thickness_km} km (v1 = {v1_km_s} km/s) above a faster "
        "half-space (v2 = {v2_km_s} km/s), compute the crossover distance of the refraction survey in "
        "kilometres.",
    ),
    scaling={"thickness_km": 1},
))

register(Family(
    "thickness_from_crossover", "refraction seismology",
    lambda p: p["crossover_km"] / 2 * math.sqrt((p["v2_km_s"] - p["v1_km_s"]) / (p["v2_km_s"] + p["v1_km_s"])),
    (P("crossover_km", 40.0, 300.0), V1_CRUST, V2_MANTLE), "km",
    (
        "In a refraction profile the direct wave (velocity {v1_km_s} km/s) and the head wave (velocity "
        "{v2_km_s} km/s) cross over at {crossover_km} km. Assuming one horizontal layer over a half-space, how "
        "thick is the layer in km?",
        "A refraction survey finds a crossover distance of {crossover_km} km, a top-layer velocity of "
        "{v1_km_s} km/s and a refractor velocity of {v2_km_s} km/s. Estimate the depth to the refractor in "
        "kilometres (single flat layer).",
    ),
    scaling={"crossover_km": 1},
))


def _intercept(p):
    v1, v2 = p["v1_km_s"], p["v2_km_s"]
    return 2 * p["thickness_km"] * math.sqrt(v2 * v2 - v1 * v1) / (v1 * v2)


register(Family(
    "intercept_time", "refraction seismology", _intercept,
    (H_CRUST, V1_CRUST, V2_MANTLE), "s",
    (
        "A horizontal layer {thickness_km} km thick with velocity {v1_km_s} km/s overlies a half-space with "
        "velocity {v2_km_s} km/s. What is the intercept time of the head-wave travel-time line, in seconds?",
        "Find the zero-offset intercept time (in s) of the refracted (head) wave for a {thickness_km} km layer "
        "of velocity {v1_km_s} km/s over a refractor of velocity {v2_km_s} km/s.",
    ),
    steps=2, scaling={"thickness_km": 1},
))


def _head_valid(p):
    ic = math.asin(p["v1_km_s"] / p["v2_km_s"])
    return p["distance_km"] > 1.1 * 2 * p["thickness_km"] * math.tan(ic)


register(Family(
    "head_wave_time", "refraction seismology",
    lambda p: p["distance_km"] / p["v2_km_s"] + _intercept(p),
    (P("distance_km", 60.0, 400.0), H_CRUST, V1_CRUST, V2_MANTLE), "s",
    (
        "The crust is a single {thickness_km} km thick layer with P velocity {v1_km_s} km/s above mantle with P "
        "velocity {v2_km_s} km/s. When does the Pn head wave arrive at a station {distance_km} km from a "
        "surface source, in seconds after the origin time?",
        "Compute the travel time (s) of the head wave refracted along a flat interface at {thickness_km} km "
        "depth to a receiver {distance_km} km away, given v1 = {v1_km_s} km/s above and v2 = {v2_km_s} km/s "
        "below; source at the surface.",
    ),
    steps=2, valid=_head_valid,
))

register(Family(
    "two_way_time", "reflection seismology",
    lambda p: 2 * p["depth_m"] / p["velocity_m_s"],
    (P("depth_m", 200.0, 6000.0), P("velocity_m_s", 1500.0, 5500.0)), "s",
    (
        "A horizontal reflector lies {depth_m} m below the surface in rock with an average P velocity of "
        "{velocity_m_s} m/s. What is the zero-offset two-way travel time of the reflection, in seconds?",
        "How long (two-way, in s) does a vertically travelling P wave take to reach a reflector at {depth_m} m "
        "depth and return, if the velocity is {velocity_m_s} m/s?",
    ),
    scaling={"depth_m": 1, "velocity_m_s": -1},
))

_NMO = (P("t0_s", 0.4, 4.0), P("offset_m", 100.0, 4000.0), P("vnmo_m_s", 1600.0, 4500.0))

register(Family(
    "nmo_traveltime", "reflection seismology",
    lambda p: math.sqrt(p["t0_s"] ** 2 + (p["offset_m"] / p["vnmo_m_s"]) ** 2),
    _NMO, "s",
    (
        "A reflection has a zero-offset two-way time of {t0_s} s and a stacking (NMO) velocity of "
        "{vnmo_m_s} m/s. Using the hyperbolic moveout equation, at what two-way time does it arrive at an "
        "offset of {offset_m} m, in seconds?",
        "For t0 = {t0_s} s, offset x = {offset_m} m and NMO velocity {vnmo_m_s} m/s, compute the reflection "
        "travel time t(x) in seconds.",
    ),
))

register(Family(
    "nmo_correction", "reflection seismology",
    lambda p: 1000 * (math.sqrt(p["t0_s"] ** 2 + (p["offset_m"] / p["vnmo_m_s"]) ** 2) - p["t0_s"]),
    _NMO, "ms",
    (
        "What normal-moveout correction (in milliseconds) must be removed from a trace at {offset_m} m offset "
        "for a reflection with zero-offset time {t0_s} s and NMO velocity {vnmo_m_s} m/s?",
        "A reflector has t0 = {t0_s} s and v_NMO = {vnmo_m_s} m/s. How much later (in ms) does the reflection "
        "arrive at {offset_m} m offset than at zero offset?",
    ),
    steps=2,
    valid=lambda p: math.sqrt(p["t0_s"] ** 2 + (p["offset_m"] / p["vnmo_m_s"]) ** 2) - p["t0_s"] > 0.005,
))

register(Family(
    "quarter_wavelength_resolution", "reflection seismology",
    lambda p: p["velocity_m_s"] / p["frequency_hz"] / 4,
    (P("velocity_m_s", 1500.0, 5500.0), P("frequency_hz", 10.0, 80.0)), "m",
    (
        "Seismic data with a dominant frequency of {frequency_hz} Hz image a layer whose velocity is "
        "{velocity_m_s} m/s. Using the quarter-wavelength criterion, what is the minimum bed thickness that "
        "can be resolved, in metres?",
        "Estimate the vertical resolution (λ/4, in m) of a wavelet with dominant frequency {frequency_hz} Hz in "
        "rock with velocity "
        "{velocity_m_s} m/s.",
    ),
    steps=2, scaling={"velocity_m_s": 1, "frequency_hz": -1},
))


def _energy_fraction(p):
    z1, z2 = p["rho1_kg_m3"] * p["v1_m_s"], p["rho2_kg_m3"] * p["v2_m_s"]
    return ((z2 - z1) / (z2 + z1)) ** 2


register(Family(
    "reflected_energy_fraction", "reflection seismology", _energy_fraction,
    (P("rho1_kg_m3", 1900.0, 2700.0), P("v1_m_s", 1800.0, 4500.0), P("rho2_kg_m3", 2000.0, 2900.0),
     P("v2_m_s", 2000.0, 6000.0)), "",
    (
        "A P wave hits a horizontal boundary at normal incidence. The upper layer has density {rho1_kg_m3} "
        "kg/m^3 and P velocity {v1_m_s} m/s; the lower layer has density {rho2_kg_m3} kg/m^3 and velocity "
        "{v2_m_s} m/s. What fraction of the incident energy is reflected? Give a number between 0 and 1.",
        "Compute the reflected energy fraction (between 0 and 1) at normal incidence for an interface between "
        "rock 1 (ρ = {rho1_kg_m3} kg/m^3, v = {v1_m_s} m/s) and rock 2 (ρ = {rho2_kg_m3} kg/m^3, "
        "v = {v2_m_s} m/s).",
    ),
    rel_tol=0.02, abs_tol=0.002, valid=lambda p: _energy_fraction(p) > 0.002,
))

# =============================================================================================
# Elastic moduli
# =============================================================================================
register(Family(
    "poisson_ratio_from_velocities", "elastic properties",
    lambda p: ((p["vp_km_s"] / p["vs_km_s"]) ** 2 - 2) / (2 * ((p["vp_km_s"] / p["vs_km_s"]) ** 2 - 1)),
    (VP, VS), "",
    (
        "A rock has a P-wave velocity of {vp_km_s} km/s and an S-wave velocity of {vs_km_s} km/s. What is its "
        "Poisson's ratio?",
        "Given vp = {vp_km_s} km/s and vs = {vs_km_s} km/s for an isotropic elastic solid, compute Poisson's "
        "ratio.",
    ),
    rel_tol=0.01, abs_tol=0.005, valid=_ratio_ok(1.5, 2.3),
))

register(Family(
    "shear_modulus_from_vs", "elastic properties",
    lambda p: p["density_kg_m3"] * (p["vs_km_s"] * 1000) ** 2 / 1e9,
    (RHO, P("vs_km_s", 1.5, 4.8, digits=3)), "GPa",
    (
        "Rock with density {density_kg_m3} kg/m^3 transmits S waves at {vs_km_s} km/s. What is its shear "
        "modulus in GPa?",
        "Compute the rigidity (in GPa) of a material whose shear-wave speed is {vs_km_s} km/s and density "
        "{density_kg_m3} kg/m^3.",
    ),
    scaling={"density_kg_m3": 1, "vs_km_s": 2},
))

register(Family(
    "bulk_modulus_from_velocities", "elastic properties",
    lambda p: p["density_kg_m3"] * ((p["vp_km_s"] * 1000) ** 2 - 4 / 3 * (p["vs_km_s"] * 1000) ** 2) / 1e9,
    (RHO, VP, VS), "GPa",
    (
        "An isotropic rock has density {density_kg_m3} kg/m^3, P velocity {vp_km_s} km/s and S velocity "
        "{vs_km_s} km/s. What is its bulk modulus in GPa?",
        "From vp = {vp_km_s} km/s, vs = {vs_km_s} km/s and ρ = {density_kg_m3} kg/m^3, compute the "
        "incompressibility K (GPa).",
    ),
    steps=2, valid=_ratio_ok(1.6, 2.2),
))

register(Family(
    "vp_from_moduli", "elastic properties",
    lambda p: math.sqrt((p["bulk_gpa"] + 4 / 3 * p["shear_gpa"]) * 1e9 / p["density_kg_m3"]) / 1000,
    (P("bulk_gpa", 10.0, 130.0), P("shear_gpa", 5.0, 70.0), RHO), "km/s",
    (
        "A rock has bulk modulus {bulk_gpa} GPa, shear modulus {shear_gpa} GPa and density {density_kg_m3} "
        "kg/m^3. What is its P-wave velocity in km/s?",
        "Compute the compressional-wave speed (km/s) of an isotropic solid with K = {bulk_gpa} GPa, "
        "μ = {shear_gpa} GPa and ρ = {density_kg_m3} kg/m^3.",
    ),
    steps=2, valid=lambda p: 0.6 < p["bulk_gpa"] / p["shear_gpa"] < 4,
))

# =============================================================================================
# Earthquake source
# =============================================================================================
_FAULT = (RIGIDITY, P("length_km", 5.0, 300.0, log=True), P("width_km", 4.0, 30.0),
          P("slip_m", 0.1, 10.0, log=True, digits=3))


def _fault_m0(p):
    return p["rigidity_gpa"] * 1e9 * p["length_km"] * 1e3 * p["width_km"] * 1e3 * p["slip_m"]


register(Family(
    "seismic_moment_from_fault", "earthquake source", _fault_m0, _FAULT, "N·m",
    (
        "A fault ruptures over a length of {length_km} km and a down-dip width of {width_km} km with an "
        "average slip of {slip_m} m. The rigidity is {rigidity_gpa} GPa. What is the seismic moment in N·m?",
        "Compute the scalar seismic moment (N·m) of an earthquake with rupture area {length_km} km × "
        "{width_km} km, mean slip {slip_m} m and shear modulus {rigidity_gpa} GPa.",
    ),
    scaling={"rigidity_gpa": 1, "length_km": 1, "width_km": 1, "slip_m": 1},
    valid=lambda p: p["width_km"] <= p["length_km"],
))

register(Family(
    "mw_from_moment", "earthquake source", lambda p: _mw(p["m0_nm"]),
    (P("m0_nm", 1e13, 1e23, log=True, digits=3),), "",
    (
        "An earthquake has a seismic moment of {m0_nm} N·m. What is its moment magnitude? " + MW_RULE,
        "Convert a scalar moment of {m0_nm} N·m into moment magnitude Mw. " + MW_RULE,
    ),
    rel_tol=0.0, abs_tol=0.05,
))

register(Family(
    "moment_from_mw", "earthquake source", lambda p: _m0(p["mw"]),
    (P("mw", 3.0, 9.0, decimals=1),), "N·m",
    (
        "What seismic moment, in N·m, corresponds to a moment magnitude of {mw}? " + MW_RULE,
        "An earthquake is reported as Mw {mw}. Compute its scalar moment in N·m. " + MW_RULE,
    ),
    rel_tol=0.02,
))

register(Family(
    "mw_from_fault", "earthquake source", lambda p: _mw(_fault_m0(p)), _FAULT, "",
    (
        "A rupture {length_km} km long and {width_km} km wide slips by {slip_m} m on average in rock with "
        "rigidity {rigidity_gpa} GPa. What is the moment magnitude? " + MW_RULE,
        "Estimate Mw for an earthquake with fault length {length_km} km, width {width_km} km, average slip "
        "{slip_m} m and shear modulus {rigidity_gpa} GPa. " + MW_RULE,
    ),
    steps=2, rel_tol=0.0, abs_tol=0.05, valid=lambda p: p["width_km"] <= p["length_km"],
))

_SLIP = (MW, RIGIDITY, P("length_km", 10.0, 400.0, log=True), P("width_km", 8.0, 30.0))


def _slip_valid(p):
    d = _m0(p["mw"]) / (p["rigidity_gpa"] * 1e9 * p["length_km"] * 1e3 * p["width_km"] * 1e3)
    return p["width_km"] <= p["length_km"] and 0.05 < d < 20


register(Family(
    "slip_from_mw", "earthquake source",
    lambda p: _m0(p["mw"]) / (p["rigidity_gpa"] * 1e9 * p["length_km"] * 1e3 * p["width_km"] * 1e3),
    _SLIP, "m",
    (
        "An Mw {mw} earthquake ruptures a fault {length_km} km long and {width_km} km wide. With a rigidity "
        "of {rigidity_gpa} GPa, what is the average slip in metres? " + MW_RULE,
        "Find the mean fault slip (m) for a moment magnitude {mw} event on a {length_km} km × {width_km} km "
        "rupture, assuming μ = {rigidity_gpa} GPa. " + MW_RULE,
    ),
    steps=2, rel_tol=0.02, valid=_slip_valid,
))


def _stress_drop(p):
    return 7 / 16 * _m0(p["mw"]) / (p["radius_km"] * 1e3) ** 3 / 1e6


register(Family(
    "stress_drop_circular", "earthquake source", _stress_drop,
    (P("mw", 3.5, 7.5, decimals=1), P("radius_km", 0.3, 25.0, log=True, digits=3)), "MPa",
    (
        "An Mw {mw} earthquake ruptures a circular crack of radius {radius_km} km. Using Δσ = (7/16) M0 / a³, "
        "what is the static stress drop in MPa? " + MW_RULE,
        "Estimate the stress drop (MPa) of a circular rupture with radius {radius_km} km and moment magnitude "
        "{mw}, using Eshelby's relation Δσ = 7M0/(16a³). " + MW_RULE,
    ),
    steps=2, rel_tol=0.02, valid=lambda p: 0.1 < _stress_drop(p) < 100,
))

register(Family(
    "radiated_energy_from_magnitude", "earthquake source",
    lambda p: 10 ** (1.5 * p["mw"] + 4.8),
    (P("mw", 3.0, 9.0, decimals=1),), "J",
    (
        "Using the Gutenberg–Richter energy relation log10 E = 1.5 M + 4.8 (E in joules), how much seismic "
        "energy does a magnitude {mw} earthquake radiate?",
        "Compute the radiated energy in J of an M {mw} earthquake from log10 E = 1.5M + 4.8.",
    ),
    rel_tol=0.02,
))

register(Family(
    "energy_ratio_between_magnitudes", "earthquake source",
    lambda p: 10 ** (1.5 * (p["m_large"] - p["m_small"])),
    (P("m_small", 3.0, 7.0, decimals=1), P("m_large", 4.0, 9.0, decimals=1)), "",
    (
        "How many times more energy does a magnitude {m_large} earthquake radiate than a magnitude {m_small} "
        "earthquake? Use log10 E = 1.5M + constant.",
        "If radiated energy scales as log10 E = 1.5M + 4.8, what is the energy ratio between an M {m_large} "
        "and an M {m_small} event?",
    ),
    rel_tol=0.02, valid=lambda p: 0.4 <= p["m_large"] - p["m_small"] <= 3.0,
))

register(Family(
    "local_magnitude", "earthquake magnitude",
    lambda p: math.log10(p["amplitude_mm"]) + 1.11 * math.log10(p["distance_km"]) + 0.00189 * p["distance_km"]
    - 2.09,
    (P("amplitude_mm", 0.01, 500.0, log=True, digits=3), P("distance_km", 10.0, 500.0)), "",
    (
        "A Wood–Anderson seismograph {distance_km} km (hypocentral distance) from an earthquake records a peak "
        "amplitude of {amplitude_mm} mm. Using ML = log10 A + 1.11 log10 r + 0.00189 r − 2.09 (A in mm, r in "
        "km), what is the local magnitude?",
        "Compute ML from A = {amplitude_mm} mm at r = {distance_km} km with the Hutton–Boore relation "
        "ML = log10 A + 1.11 log10 r + 0.00189 r − 2.09.",
    ),
    steps=2, rel_tol=0.0, abs_tol=0.05,
))

# =============================================================================================
# Statistical seismology
# =============================================================================================
_GR = (P("a_value", 3.0, 6.5, decimals=2), P("b_value", 0.8, 1.2, decimals=2), P("magnitude", 3.0, 7.0, decimals=1))


def _gr_rate(p):
    return 10 ** (p["a_value"] - p["b_value"] * p["magnitude"])


register(Family(
    "gr_annual_count", "statistical seismology", _gr_rate, _GR, "",
    (
        "A region's seismicity follows log10 N = {a_value} − {b_value} M, where N is the number of earthquakes "
        "per year with magnitude ≥ M. How many earthquakes of magnitude {magnitude} or larger are expected per "
        "year?",
        "With Gutenberg–Richter parameters a = {a_value} and b = {b_value} (annual rates), compute the expected "
        "yearly number of events with M ≥ {magnitude}.",
    ),
    rel_tol=0.02, valid=lambda p: 0.01 < _gr_rate(p) < 1e5,
))

register(Family(
    "gr_return_period", "statistical seismology", lambda p: 1 / _gr_rate(p), _GR, "yr",
    (
        "Seismicity in a region obeys log10 N = {a_value} − {b_value} M, with N the annual number of events of "
        "magnitude ≥ M. What is the mean return period of earthquakes with M ≥ {magnitude}, in years?",
        "Given annual Gutenberg–Richter values a = {a_value}, b = {b_value}, how many years on average separate "
        "earthquakes of magnitude {magnitude} or greater?",
    ),
    steps=2, rel_tol=0.02, valid=lambda p: 1 <= 1 / _gr_rate(p) <= 20000,
))


def _aki(p):
    return LOG10_E / (p["mean_magnitude"] - (p["mc"] - p["bin_width"] / 2))


register(Family(
    "aki_b_value", "statistical seismology", _aki,
    (P("mean_magnitude", 1.8, 4.6, decimals=2), P("mc", 1.5, 4.0, decimals=1), P("bin_width", 0.1, 0.1, decimals=1)),
    "",
    (
        "A catalogue complete above Mc = {mc} has magnitudes binned at {bin_width} units and a mean magnitude "
        "(for M ≥ Mc) of {mean_magnitude}. Using Aki's maximum-likelihood estimator with Utsu's correction for "
        "binning, b = log10(e) / (M̄ − (Mc − ΔM/2)), what is the b-value?",
        "Estimate the Gutenberg–Richter b-value by maximum likelihood from a mean magnitude of "
        "{mean_magnitude}, a completeness magnitude of {mc} and a bin width ΔM = {bin_width} "
        "(b = log10 e / (M̄ − Mc + ΔM/2)).",
    ),
    rel_tol=0.0, abs_tol=0.02, valid=lambda p: 0.6 < _aki(p) < 1.6,
))

# =============================================================================================
# Site effects, ground motion and attenuation
# =============================================================================================
register(Family(
    "site_resonance_frequency", "site effects",
    lambda p: p["vs_m_s"] / (4 * p["thickness_m"]),
    (P("vs_m_s", 120.0, 600.0), P("thickness_m", 5.0, 120.0)), "Hz",
    (
        "A soft soil layer {thickness_m} m thick with shear-wave velocity {vs_m_s} m/s rests on bedrock. What is "
        "its fundamental resonance frequency for vertically propagating S waves, in Hz?",
        "Estimate the site's fundamental frequency (Hz) from the quarter-wavelength rule for a {thickness_m} m "
        "sediment layer with Vs = {vs_m_s} m/s over rock.",
    ),
    scaling={"vs_m_s": 1, "thickness_m": -1},
))

register(Family(
    "soil_thickness_from_resonance", "site effects",
    lambda p: p["vs_m_s"] / (4 * p["f0_hz"]),
    (P("vs_m_s", 120.0, 600.0), P("f0_hz", 0.5, 10.0, digits=3)), "m",
    (
        "Ambient-noise H/V measurements show a resonance peak at {f0_hz} Hz over sediments with an average "
        "shear-wave velocity of {vs_m_s} m/s. Using the quarter-wavelength relation, how thick are the "
        "sediments, in metres?",
        "A single soil layer over bedrock resonates at {f0_hz} Hz. If its Vs is {vs_m_s} m/s, what is its "
        "thickness in m?",
    ),
    scaling={"vs_m_s": 1, "f0_hz": -1},
))

_VS30 = tuple(P(n, 0, 0, digits=3) for n in ("h1_m", "v1_m_s", "h2_m", "v2_m_s", "h3_m", "v3_m_s"))


def _vs30_make(rng: random.Random) -> dict[str, float]:
    h1, h2 = round(rng.uniform(2, 12), 1), round(rng.uniform(3, 12), 1)
    h3 = round(30 - h1 - h2, 1)
    v1 = rng.uniform(120, 300)
    v2 = v1 + rng.uniform(30, 300)
    v3 = v2 + rng.uniform(50, 400)
    raw = {"h1_m": h1, "h2_m": h2, "h3_m": h3, "v1_m_s": v1, "v2_m_s": v2, "v3_m_s": v3}
    return {q.name: (raw[q.name] if q.name.startswith("h") else q.round(raw[q.name])) for q in _VS30}


register(Family(
    "vs30", "site effects",
    lambda p: 30 / (p["h1_m"] / p["v1_m_s"] + p["h2_m"] / p["v2_m_s"] + p["h3_m"] / p["v3_m_s"]),
    _VS30, "m/s",
    (
        "The top 30 m at a site consist of {h1_m} m of soil with Vs = {v1_m_s} m/s, then {h2_m} m with "
        "Vs = {v2_m_s} m/s, then {h3_m} m with Vs = {v3_m_s} m/s. What is the time-averaged shear-wave "
        "velocity Vs30, in m/s?",
        "Compute Vs30 (m/s) for a profile of three layers: {h1_m} m at {v1_m_s} m/s, {h2_m} m at {v2_m_s} m/s "
        "and {h3_m} m at {v3_m_s} m/s.",
    ),
    steps=2, make=_vs30_make, valid=lambda p: p["h3_m"] >= 3,
))

register(Family(
    "impedance_amplification", "site effects",
    lambda p: p["rho_rock_kg_m3"] * p["vs_rock_m_s"] / (p["rho_soil_kg_m3"] * p["vs_soil_m_s"]),
    (P("rho_rock_kg_m3", 2300.0, 2800.0), P("vs_rock_m_s", 800.0, 3000.0), P("rho_soil_kg_m3", 1600.0, 2100.0),
     P("vs_soil_m_s", 100.0, 500.0)), "",
    (
        "For vertically incident S waves, the peak amplification of an undamped soil layer at resonance equals "
        "the impedance contrast with the underlying rock. The soil has density {rho_soil_kg_m3} kg/m^3 and Vs "
        "{vs_soil_m_s} m/s; the rock has density {rho_rock_kg_m3} kg/m^3 and Vs {vs_rock_m_s} m/s. What is the "
        "amplification factor?",
        "Compute the rock-to-soil shear impedance ratio (ρ_r Vs_r)/(ρ_s Vs_s) for rock with ρ = "
        "{rho_rock_kg_m3} kg/m^3, Vs = {vs_rock_m_s} m/s and soil with ρ = {rho_soil_kg_m3} kg/m^3, "
        "Vs = {vs_soil_m_s} m/s.",
    ),
))

_HARM = (P("frequency_hz", 0.5, 10.0, digits=3), P("displacement_mm", 0.5, 50.0, digits=3))

register(Family(
    "harmonic_peak_acceleration", "ground motion",
    lambda p: (2 * math.pi * p["frequency_hz"]) ** 2 * p["displacement_mm"] / 1000,
    _HARM, "m/s^2",
    (
        "Ground motion at a site is approximately sinusoidal with frequency {frequency_hz} Hz and peak "
        "displacement {displacement_mm} mm. What is the peak ground acceleration in m/s^2?",
        "For harmonic shaking of amplitude {displacement_mm} mm at {frequency_hz} Hz, compute the peak "
        "acceleration (m/s^2).",
    ),
    steps=2, scaling={"frequency_hz": 2, "displacement_mm": 1},
))

register(Family(
    "harmonic_peak_velocity", "ground motion",
    lambda p: 2 * math.pi * p["frequency_hz"] * p["displacement_mm"] / 10,
    _HARM, "cm/s",
    (
        "A seismometer records sinusoidal ground displacement with amplitude {displacement_mm} mm at "
        "{frequency_hz} Hz. What is the peak ground velocity in cm/s?",
        "Compute the peak particle velocity (cm/s) of harmonic motion with {displacement_mm} mm amplitude and "
        "{frequency_hz} Hz frequency.",
    ),
    scaling={"frequency_hz": 1, "displacement_mm": 1},
))

register(Family(
    "body_wave_spreading", "attenuation",
    lambda p: p["a1_um"] * p["r1_km"] / p["r2_km"],
    (P("a1_um", 1.0, 100.0), P("r1_km", 10.0, 50.0), P("r2_km", 60.0, 500.0)), "µm",
    (
        "A body wave has an amplitude of {a1_um} µm at {r1_km} km from the source. Considering only geometric "
        "spreading (amplitude ∝ 1/r), what is its amplitude at {r2_km} km, in µm?",
        "If amplitude decays as 1/r for body waves, what amplitude (µm) is expected at {r2_km} km for a wave "
        "measured at {a1_um} µm at {r1_km} km?",
    ),
    scaling={"a1_um": 1, "r1_km": 1, "r2_km": -1},
))


def _anelastic(p):
    return p["r0_km"] / p["r_km"] * math.exp(-math.pi * p["frequency_hz"] * (p["r_km"] - p["r0_km"])
                                             / (p["q"] * p["velocity_km_s"]))


register(Family(
    "anelastic_amplitude_ratio", "attenuation", _anelastic,
    (P("r0_km", 10.0, 30.0), P("r_km", 50.0, 300.0), P("frequency_hz", 1.0, 10.0, digits=3), P("q", 100.0, 1000.0),
     P("velocity_km_s", 3.2, 3.8, digits=3)), "",
    (
        "An S wave of frequency {frequency_hz} Hz travels at {velocity_km_s} km/s through crust with quality "
        "factor Q = {q}. Its amplitude decays as A(r) = A0 (r0/r) exp(−π f (r − r0) / (Q v)). What is the "
        "ratio A(r)/A0 between r = {r_km} km and the reference distance r0 = {r0_km} km?",
        "Combining 1/r geometric spreading with anelastic attenuation exp(−π f t/Q), where t is the extra "
        "travel time, compute the amplitude ratio of a wave of frequency {frequency_hz} Hz at {r_km} km relative to "
        "{r0_km} km, for Q = {q} and v = {velocity_km_s} km/s.",
    ),
    steps=2, rel_tol=0.02, valid=lambda p: _anelastic(p) > 1e-4,
))

register(Family(
    "q_from_amplitude_decay", "attenuation",
    lambda p: math.pi * p["frequency_hz"] * p["travel_time_s"] / math.log(p["amplitude_ratio"]),
    (P("frequency_hz", 1.0, 20.0, digits=3), P("travel_time_s", 5.0, 60.0),
     P("amplitude_ratio", 1.2, 10.0, digits=3)), "",
    (
        "Ignoring geometric spreading, a wave of frequency {frequency_hz} Hz loses amplitude by a factor of "
        "{amplitude_ratio} over a travel time of {travel_time_s} s. Using A = A0 exp(−π f t / Q), what is the "
        "quality factor Q?",
        "A wave's amplitude falls to 1/{amplitude_ratio} of its initial value after {travel_time_s} s of "
        "travel at {frequency_hz} Hz, from attenuation alone. Estimate Q.",
    ),
))

register(Family(
    "seismic_wavelength", "seismic waves",
    lambda p: p["velocity_km_s"] * 1000 / p["frequency_hz"],
    (P("velocity_km_s", 0.2, 8.0, digits=3), P("frequency_hz", 0.05, 20.0, log=True, digits=3)), "m",
    (
        "What is the wavelength, in metres, of a seismic wave with frequency {frequency_hz} Hz travelling at "
        "{velocity_km_s} km/s?",
        "A wave with frequency {frequency_hz} Hz propagates at {velocity_km_s} km/s. Compute its wavelength in m.",
    ),
    scaling={"velocity_km_s": 1, "frequency_hz": -1},
))

register(Family(
    "surface_wave_group_delay", "seismic waves",
    lambda p: p["distance_km"] * (1 / p["u_slow_km_s"] - 1 / p["u_fast_km_s"]),
    (P("distance_km", 500.0, 6000.0), P("u_slow_km_s", 2.8, 3.6, digits=3), P("u_fast_km_s", 3.0, 4.2, digits=3)),
    "s",
    (
        "Rayleigh waves at one period travel with group velocity {u_slow_km_s} km/s and at another period with "
        "{u_fast_km_s} km/s. How many seconds apart do the two arrive at a station {distance_km} km from the "
        "source?",
        "Because of dispersion, two wave packets with group velocities {u_fast_km_s} km/s and {u_slow_km_s} km/s "
        "leave a source together. What is their arrival-time difference (s) after {distance_km} km?",
    ),
    steps=2, valid=lambda p: p["u_fast_km_s"] - p["u_slow_km_s"] > 0.08,
))

_ARRAY = (P("dx_km", 0, 0, digits=3), P("dt_s", 0, 0, digits=3), P("velocity_km_s", 0, 0, digits=3))


def _array_make(rng: random.Random) -> dict[str, float]:
    i, v, dx = math.radians(rng.uniform(10, 70)), rng.uniform(5.8, 6.6), rng.uniform(1, 20)
    raw = {"dx_km": dx, "velocity_km_s": v, "dt_s": dx * math.sin(i) / v}
    return {q.name: q.round(raw[q.name]) for q in _ARRAY}


register(Family(
    "incidence_from_array_delay", "seismic arrays",
    lambda p: math.degrees(math.asin(p["velocity_km_s"] * p["dt_s"] / p["dx_km"])),
    _ARRAY, "°",
    (
        "A plane P wave crosses two stations {dx_km} km apart along its direction of propagation; it reaches the "
        "far station {dt_s} s after the near one. The P velocity beneath the stations is {velocity_km_s} km/s. "
        "What is the angle of incidence from the vertical, in degrees?",
        "From an inter-station delay of {dt_s} s over a {dx_km} km baseline aligned with the back-azimuth, and a "
        "near-surface velocity of {velocity_km_s} km/s, compute the incidence angle (degrees from vertical) of "
        "the incoming plane wave.",
    ),
    steps=2, rel_tol=0.0, abs_tol=0.5, make=_array_make,
    valid=lambda p: p["velocity_km_s"] * p["dt_s"] / p["dx_km"] < 0.95,
))

# =============================================================================================
# Held-out section: not published. The 10 held-out T1 families behind D-Num live in
# seismology_heldout.py, which stays out of git so that the test problems cannot be regenerated or
# trained on (README, "Data & licensing"). It is imported at the end of this module when present.
# =============================================================================================


# The held-out D-Num families are not published; they register themselves when their private module is present.
# Imported last, because they use the helpers above (dpt.data.gen.common.import_private_module_if_present).
from dpt.data.gen.common import import_private_module_if_present  # noqa: E402

heldout_dnum_families_present = import_private_module_if_present("dpt.data.gen.seismology_heldout")
