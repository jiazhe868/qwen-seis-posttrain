"""Library of T2 seismic data-processing task kinds: spec, reference implementation, input generator.

A *kind* builds one :class:`Variant` per call: a fully specified function (signature, units, edge
behaviour), its reference implementation as Python source, 1–2 small example inputs (shown in the prompt
and used as the visible tests) and the inputs of the hidden tests. ``dpt.data.gen.t2`` runs the reference
on the inputs to get the expected values, writes the test files and checks them in the sandbox.

Rules every kind follows:

* The spec states everything the tests check: argument units, conventions (sign, window, rounding,
  tie-breaking), and what to return when nothing qualifies. Inputs avoid boundary cases the spec leaves
  open (exact ties, samples exactly on a window edge).
* The reference is plain Python (``math`` only), defines exactly one public function named by the
  placeholder ``ENTRY``, and is deterministic.
* ``key`` names the semantic variant (kind + options, not names or wording), so generation can cap how
  many instances of one spec it emits.

``split="heldout"`` kinds are used only for D-Code. They are different computations, not rewordings of
training kinds. Their definitions are not published: they live in ``t2_tasks_heldout.py``, kept out of git
(README, "Data & licensing").
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

G0 = 9.80665                        # standard gravity [m/s²], stated in every spec that uses it
R_EARTH = 6371.0                    # mean Earth radius [km], stated in every spec that uses it


@dataclass
class Variant:
    entry: str
    spec: str
    reference: str | None                      # source defining ``entry``; None → ``fn`` (solver-backed)
    examples: list[tuple]
    hidden: list[tuple]
    key: str
    rel_tol: float = 1e-6
    abs_scale: float = 1e-7                    # abs tolerance as a fraction of the largest |expected|
    abs_tol: float = 1e-12
    fn: Callable[..., Any] | None = None       # in-process reference when there is no source
    family: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Kind:
    name: str
    topic: str
    split: str
    build: Callable[[random.Random], Variant]


KINDS: dict[str, Kind] = {}


def kind(name: str, topic: str, split: str = "train"):
    def deco(fn: Callable[[random.Random], Variant]) -> Callable[[random.Random], Variant]:
        if name in KINDS:
            raise ValueError(f"T2 kind {name!r} already registered")
        KINDS[name] = Kind(name, topic, split, fn)
        return fn
    return deco


# --------------------------------------------------------------------------- helpers


def pick(rng: random.Random, seq: Sequence[Any]) -> Any:
    return seq[rng.randrange(len(seq))]


def _r(x: float, d: int = 4) -> float:
    return round(float(x), d)


def ref(src: str, entry: str) -> str:
    return src.strip("\n").replace("ENTRY", entry) + "\n"


_STYLES = (
    "Write a Python function `{sig}` to {what}.",
    "Implement `{sig}`. It should {what}.",
    "Seismic data-processing utility: `{sig}` must {what}.",
)


def phrase(rng: random.Random, sig: str, what: str, *details: str) -> str:
    """The spec text: a signature sentence in one of a few styles, then the details."""
    head = pick(rng, _STYLES).format(sig=sig, what=what)
    return " ".join([head, *[d for d in details if d]])


def noise(rng: random.Random, n: int, s: float = 1.0) -> list[float]:
    return [_r(rng.gauss(0.0, s)) for _ in range(n)]


def trace(rng: random.Random, n: int, *, offset: float = 0.0, trend: float = 0.0, amp: float = 1.0) -> list[float]:
    """A few sinusoids plus noise, an offset and a linear trend; 4 decimals."""
    comps = [(rng.uniform(0.3, 1.0) * amp, rng.uniform(0.02, 0.23), rng.uniform(0, 2 * math.pi))
             for _ in range(rng.randint(1, 3))]
    out = []
    for i in range(n):
        v = offset + trend * i + sum(a * math.sin(2 * math.pi * f * i + ph) for a, f, ph in comps)
        out.append(_r(v + rng.gauss(0, 0.1 * amp)))
    return out


def burst(rng: random.Random, n: int, onset: int, *, noise_s: float = 0.05, amp: float = 1.0) -> list[float]:
    """Noise, then a decaying oscillation starting at ``onset`` (an event)."""
    f, decay = rng.uniform(0.05, 0.2), rng.uniform(0.02, 0.08)
    out = []
    for i in range(n):
        v = rng.gauss(0, noise_s)
        if i >= onset:
            k = i - onset
            v += amp * math.exp(-decay * k) * math.sin(2 * math.pi * f * k + 0.3)
        out.append(_r(v))
    return out


def accel(rng: random.Random, n: int, peak: float) -> list[float]:
    """A strong-motion-like acceleration record [m/s²]: enveloped oscillation, peak ≈ ``peak``."""
    f = rng.uniform(0.04, 0.15)
    t0 = n * rng.uniform(0.2, 0.4)
    width = n * rng.uniform(0.1, 0.25)
    out = []
    for i in range(n):
        env = math.exp(-((i - t0) / width) ** 2)
        out.append(_r(peak * env * math.sin(2 * math.pi * f * i + 0.7) + rng.gauss(0, 0.01 * peak)))
    return out


def gr_mags(rng: random.Random, n: int, mc: float, b: float, dm: float) -> list[float]:
    """Gutenberg–Richter magnitudes ≥ mc − dm/2, binned to ``dm`` (some below mc, to be excluded)."""
    beta = b * math.log(10)
    out = []
    for _ in range(n):
        m = (mc - 0.3) + rng.expovariate(beta)
        out.append(round(round(m / dm) * dm, 2))
    return out


def latlon(rng: random.Random) -> tuple[float, float]:
    return _r(rng.uniform(-60, 60), 3), _r(rng.uniform(-179, 179), 3)


# =========================================================================== training kinds


@kind("remove_offset", "trace preprocessing")
def _remove_offset(rng):
    center = pick(rng, ("mean", "median", "pre_event"))
    ret = pick(rng, ("trace", "value"))
    names = {"mean": ("demean", "remove_mean", "remove_offset"), "median": ("remove_median", "median_center"),
             "pre_event": ("remove_pre_event_offset", "baseline_correct")}[center]
    if ret == "value":
        names = {"mean": ("trace_offset", "dc_offset"), "median": ("median_offset",),
                 "pre_event": ("pre_event_offset", "baseline_level")}[center]
    entry = pick(rng, names)
    sig = f"{entry}(x, k)" if center == "pre_event" else f"{entry}(x)"
    what = {"mean": "the mean of `x`", "median": "the median of `x`",
            "pre_event": "the mean of the first `k` samples of `x` (the pre-event noise, 1 ≤ k ≤ len(x))"}[center]
    spec = phrase(rng, sig, "remove the offset of a seismic trace" if ret == "trace" else
                  "measure the offset of a seismic trace",
                  f"`x` is a non-empty list of floats. The offset is {what}.",
                  "For an even number of samples the median is the average of the two middle values."
                  if center == "median" else "",
                  "Return a new list of the same length with the offset subtracted from every sample."
                  if ret == "trace" else "Return the offset as a float.")
    off = {"mean": "sum(x) / len(x)",
           "median": "(s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))",
           "pre_event": "sum(x[:k]) / k"}[center]
    pre = "    s = sorted(x)\n    n = len(s)\n" if center == "median" else ""
    args_sig = "x, k" if center == "pre_event" else "x"
    tail = "    return [v - m for v in x]\n" if ret == "trace" else "    return m\n"
    src = f"def ENTRY({args_sig}):\n{pre}    m = {off}\n{tail}"

    def args(n):
        x = trace(rng, n, offset=rng.uniform(-50, 50), amp=rng.uniform(0.5, 20))
        return (x, rng.randint(1, max(1, n // 2))) if center == "pre_event" else (x,)
    ex = [args(rng.randint(5, 7))]
    hid = [args(rng.randint(30, 90)) for _ in range(5)] + [args(1)]
    return Variant(entry, spec, ref(src, entry), ex, hid, f"remove_offset|{center}|{ret}")


@kind("linear_detrend", "trace preprocessing")
def _linear_detrend(rng):
    mode = pick(rng, ("residual", "fit"))
    if mode == "residual":
        entry = pick(rng, ("detrend", "remove_linear_trend", "linear_detrend"))
        spec = phrase(rng, f"{entry}(x)", "remove the least-squares straight line from a trace",
                      "`x` is a list of at least 2 floats sampled at equal intervals; use the sample index "
                      "i = 0, 1, … as the abscissa. Fit x[i] ≈ a + b·i by ordinary least squares and return the "
                      "list of residuals x[i] − (a + b·i).")
        src = '''
def ENTRY(x):
    n = len(x)
    mi = (n - 1) / 2.0
    mx = sum(x) / n
    sxx = sum((i - mi) ** 2 for i in range(n))
    sxy = sum((i - mi) * (v - mx) for i, v in enumerate(x))
    b = sxy / sxx
    a = mx - b * mi
    return [v - (a + b * i) for i, v in enumerate(x)]
'''
        ex = [(trace(rng, 6, trend=rng.uniform(-1, 1)),)]
        hid = [(trace(rng, rng.randint(20, 90), trend=rng.uniform(-0.3, 0.3), offset=rng.uniform(-5, 5)),)
               for _ in range(5)] + [([_r(rng.uniform(-3, 3)), _r(rng.uniform(-3, 3))],)]
    else:
        entry = pick(rng, ("trend_line", "fit_linear_trend"))
        spec = phrase(rng, f"{entry}(x, dt)", "fit a straight line to a trace sampled every `dt` seconds",
                      "Sample i is at time t = i·dt (i = 0, 1, …; `x` has at least 2 samples). Fit x ≈ a + b·t by "
                      "ordinary least squares and return the tuple (b, a): the slope in units of x per second and "
                      "the value of the line at t = 0.")
        src = '''
def ENTRY(x, dt):
    n = len(x)
    t = [i * dt for i in range(n)]
    mt = sum(t) / n
    mx = sum(x) / n
    stt = sum((ti - mt) ** 2 for ti in t)
    stx = sum((ti - mt) * (v - mx) for ti, v in zip(t, x))
    b = stx / stt
    return (b, mx - b * mt)
'''
        ex = [(trace(rng, 6, trend=rng.uniform(-1, 1)), pick(rng, (0.01, 0.02, 0.5)))]
        hid = [(trace(rng, rng.randint(20, 90), trend=rng.uniform(-0.3, 0.3), offset=rng.uniform(-5, 5)),
                pick(rng, (0.005, 0.01, 0.02, 0.025, 0.1))) for _ in range(6)]
    return Variant(entry, spec, ref(src, entry), ex, hid, f"linear_detrend|{mode}")


@kind("cosine_taper", "trace preprocessing")
def _cosine_taper(rng):
    fixed = pick(rng, (None, 0.05, 0.1, 0.2, 0.25))
    shape = pick(rng, ("cosine", "linear"))
    entry = pick(rng, ("cosine_taper", "taper", "hann_taper") if shape == "cosine" else ("linear_taper", "ramp_taper"))
    sig = f"{entry}(x)" if fixed else f"{entry}(x, p)"
    pdesc = (f"with p = {fixed} (the fraction tapered at each end)" if fixed
             else "where `p` (0 ≤ p ≤ 0.5) is the fraction tapered at each end")
    wdef = "w = 0.5·(1 − cos(π·i/m))" if shape == "cosine" else "w = i/m"
    spec = phrase(rng, sig, f"apply a {shape} taper to both ends of a trace {pdesc}",
                  "With n = len(x) and m = floor(p·n): if m = 0 return an unchanged copy; otherwise, for "
                  f"i = 0 … m−1, multiply both x[i] and x[n−1−i] by {wdef}. Samples outside "
                  "the two tapered ends are unchanged. Return a new list; do not modify `x`.")
    wexpr = "0.5 * (1.0 - math.cos(math.pi * i / m))" if shape == "cosine" else "i / m"
    head = "def ENTRY(x):\n    p = " + repr(fixed) + "\n" if fixed else "def ENTRY(x, p):\n"
    src = ("import math\n\n\n" + head +
           "    n = len(x)\n    m = int(math.floor(p * n))\n    y = list(x)\n    if m == 0:\n        return y\n"
           "    for i in range(m):\n        w = " + wexpr + "\n        y[i] *= w\n        y[n - 1 - i] *= w\n"
           "    return y\n")

    def args(n):
        x = trace(rng, n)
        return (x,) if fixed else (x, pick(rng, (0.05, 0.1, 0.15, 0.2, 0.25)))
    ex = [args(10)]
    hid = [args(rng.randint(20, 100)) for _ in range(6)]
    return Variant(entry, spec, ref(src, entry), ex, hid, f"cosine_taper|{fixed}|{shape}")


@kind("rms_amplitude", "amplitude measures")
def _rms(rng):
    mode = pick(rng, ("all", "window"))
    measure = pick(rng, ("rms", "mean_abs"))
    mdesc = "sqrt of the mean of the squared samples" if measure == "rms" else "the mean of the absolute values"
    mexpr = ("math.sqrt(sum(v * v for v in sel) / len(sel))" if measure == "rms" else
             "sum(abs(v) for v in sel) / len(sel)")
    label = "root-mean-square" if measure == "rms" else "mean absolute"
    if mode == "all":
        entry = pick(rng, ("rms", "rms_amplitude") if measure == "rms" else ("mean_abs_amplitude",))
        spec = phrase(rng, f"{entry}(x)", f"return the {label} amplitude of a trace",
                      f"`x` is a list of floats: return {mdesc}. Return 0.0 for an empty list.")
        src = ("import math\n\n\ndef ENTRY(x):\n    sel = x\n    if not sel:\n        return 0.0\n"
               "    return " + mexpr + "\n")
        ex = [(trace(rng, 5),)]
        hid = [(trace(rng, rng.randint(10, 100), amp=rng.uniform(0.1, 30)),) for _ in range(5)] + [([],)]
    else:
        entry = pick(rng, ("window_rms", "rms_in_window") if measure == "rms" else ("window_mean_abs",))
        spec = phrase(rng, f"{entry}(x, dt, t_start, t_end)",
                      f"return the {label} amplitude of the samples of a trace inside a time window",
                      "Sample i of `x` is at time i·dt seconds. Use the samples with t_start ≤ i·dt < t_end and "
                      f"return {mdesc}; return 0.0 if no sample is in the window.")
        src = ("import math\n\n\ndef ENTRY(x, dt, t_start, t_end):\n"
               "    sel = [v for i, v in enumerate(x) if t_start <= i * dt < t_end]\n"
               "    if not sel:\n        return 0.0\n    return " + mexpr + "\n")

        def args(n, empty=False):
            dt = pick(rng, (0.01, 0.02, 0.05))
            a = rng.randint(0, n // 2)
            b = rng.randint(a + 1, n)
            if empty:
                a, b = n + 2, n + 5
            return (trace(rng, n), dt, _r((a - 0.5) * dt, 6), _r((b - 0.5) * dt, 6))
        ex = [args(8)]
        hid = [args(rng.randint(20, 100)) for _ in range(5)] + [args(30, empty=True)]
    return Variant(entry, spec, ref(src, entry), ex, hid, f"rms_amplitude|{mode}|{measure}")


@kind("peak_acceleration", "strong motion")
def _pga(rng):
    out = pick(rng, ("m/s²", "g", "cm/s²"))
    with_time = rng.random() < 0.4
    entry = pick(rng, ("pga", "peak_ground_acceleration", "peak_acceleration"))
    conv = {"m/s²": "", "g": f" divided by g = {G0} m/s² (so the result is in g)",
            "cm/s²": " converted to cm/s² (multiply by 100)"}[out]
    ret = (f"Return the tuple (peak, time): the largest absolute sample value of `acc`{conv}, and the time "
           "i·dt in seconds of the first sample where it occurs." if with_time else
           f"Return the largest absolute sample value of `acc`{conv}.")
    spec = phrase(rng, f"{entry}(acc, dt)" if with_time else f"{entry}(acc)",
                  "compute the peak ground acceleration of an accelerogram",
                  "`acc` is a non-empty list of accelerations in m/s²"
                  + (" sampled every `dt` seconds." if with_time else "."), ret)
    factor = {"m/s²": "1.0", "g": f"1.0 / {G0}", "cm/s²": "100.0"}[out]
    src = (f'''
def ENTRY(acc, dt):
    k = max(range(len(acc)), key=lambda i: (abs(acc[i]), -i))
    return (abs(acc[k]) * {factor}, k * dt)
''' if with_time else f'''
def ENTRY(acc):
    return max(abs(a) for a in acc) * {factor}
''')

    def args(n):
        a = accel(rng, n, rng.uniform(0.2, 6.0))
        return (a, pick(rng, (0.005, 0.01, 0.02))) if with_time else (a,)
    return Variant(entry, f"{spec}", ref(src, entry), [args(8)], [args(rng.randint(30, 120)) for _ in range(6)],
                   f"peak_acceleration|{out}|{with_time}")


@kind("peak_velocity", "strong motion")
def _pgv(rng):
    qty = pick(rng, ("velocity", "displacement"))
    out = pick(rng, ("m/s", "cm/s") if qty == "velocity" else ("m", "cm"))
    demean = rng.random() < 0.4
    entry = pick(rng, ("pgv_from_acceleration", "peak_ground_velocity", "pgv") if qty == "velocity"
                 else ("pgd_from_acceleration", "peak_ground_displacement", "pgd"))
    integ = ("Integrate with the trapezoidal rule starting from zero velocity: v[0] = 0 and "
             "v[i] = v[i−1] + 0.5·(a[i−1] + a[i])·dt.")
    if qty == "displacement":
        integ += " Integrate v the same way to displacement d (d[0] = 0)."
    conv = {"m/s": "m/s.", "cm/s": "cm/s (multiply by 100).", "m": "m.", "cm": "cm (multiply by 100)."}[out]
    spec = phrase(rng, f"{entry}(acc, dt)",
                  f"integrate an accelerogram and return the peak absolute {qty}",
                  "`acc` is a list of at least 2 accelerations in m/s² sampled every `dt` seconds." +
                  (" First subtract the mean of `acc` from every sample." if demean else ""),
                  integ, f"Return max |{'v' if qty == 'velocity' else 'd'}[i]| in {conv}")
    src = (
        "def ENTRY(acc, dt):\n    a = list(acc)\n"
        + ("    m = sum(a) / len(a)\n    a = [x - m for x in a]\n" if demean else "")
        + "    v = [0.0]\n    for i in range(1, len(a)):\n        v.append(v[-1] + 0.5 * (a[i - 1] + a[i]) * dt)\n"
        + ("    d = [0.0]\n    for i in range(1, len(v)):\n        d.append(d[-1] + 0.5 * (v[i - 1] + v[i]) * dt)\n"
           "    v = d\n" if qty == "displacement" else "")
        + "    return max(abs(x) for x in v) * " + ("100.0" if out.startswith("cm") else "1.0") + "\n")

    def args(n):
        return (accel(rng, n, rng.uniform(0.2, 6.0)), pick(rng, (0.005, 0.01, 0.02)))
    return Variant(entry, spec, ref(src, entry), [args(8)], [args(rng.randint(30, 120)) for _ in range(6)],
                   f"peak_velocity|{qty}|{out}|{demean}")


@kind("arias_intensity", "strong motion")
def _arias(rng):
    unit_in = pick(rng, ("m/s²", "g"))
    unit_out = pick(rng, ("m/s", "cm/s"))
    entry = pick(rng, ("arias_intensity", "arias"))
    spec = phrase(rng, f"{entry}(acc, dt)", "compute the Arias intensity of an accelerogram",
                  f"`acc` is a list of at least 2 accelerations in {unit_in} sampled every `dt` seconds" +
                  (f" (convert to m/s² with g = {G0} m/s²)." if unit_in == "g" else "."),
                  f"Ia = π/(2g) · ∫ a(t)² dt with g = {G0} m/s², a in m/s², and the integral by the trapezoidal "
                  "rule over consecutive samples. Return Ia in " +
                  ("m/s." if unit_out == "m/s" else "cm/s (multiply by 100)."))
    src = ("import math\n\n\ndef ENTRY(acc, dt):\n"
           f"    a = [x * {repr(G0) if unit_in == 'g' else '1.0'} for x in acc]\n"
           "    s = sum(0.5 * (a[i - 1] ** 2 + a[i] ** 2) * dt for i in range(1, len(a)))\n"
           f"    return math.pi / (2.0 * {G0!r}) * s * {'100.0' if unit_out == 'cm/s' else '1.0'}\n")

    def args(n):
        peak = rng.uniform(0.2, 6.0) / (G0 if unit_in == "g" else 1.0)
        return ([_r(v, 5) for v in accel(rng, n, peak)], pick(rng, (0.005, 0.01, 0.02)))
    return Variant(entry, spec, ref(src, entry), [args(8)], [args(rng.randint(30, 120)) for _ in range(6)],
                   f"arias_intensity|{unit_in}|{unit_out}")


@kind("significant_duration", "strong motion")
def _sig_duration(rng):
    lo, hi = pick(rng, ((0.05, 0.95), (0.05, 0.75), (0.25, 0.75)))
    unit = pick(rng, ("seconds", "samples"))
    entry = pick(rng, ("significant_duration", "arias_duration"))
    sig = f"{entry}(acc, dt)" if unit == "seconds" else f"{entry}(acc)"
    spec = phrase(rng, sig,
                  f"return the {int(lo * 100)}–{int(hi * 100)}% significant duration of an accelerogram",
                  "`acc` is a list of accelerations" + (" sampled every `dt` seconds" if unit == "seconds" else "") +
                  ", not all zero. Build the cumulative sum c[0] = 0, c[i] = c[i−1] + 0.5·(a[i−1]² + a[i]²)"
                  + ("·dt" if unit == "seconds" else "") + ", and normalize by its last value. Let i1 "
                  f"be the first index where the normalized value is ≥ {lo} and i2 the first index where it is "
                  f"≥ {hi}.", "Return (i2 − i1)·dt in seconds." if unit == "seconds" else
                  "Return the number of samples i2 − i1 (an int).")
    src = ("def ENTRY(acc" + (", dt" if unit == "seconds" else "") + "):\n    c = [0.0]\n"
           "    for i in range(1, len(acc)):\n"
           "        c.append(c[-1] + 0.5 * (acc[i - 1] ** 2 + acc[i] ** 2)" + (" * dt" if unit == "seconds" else "")
           + ")\n    total = c[-1]\n"
           f"    i1 = next(i for i, v in enumerate(c) if v / total >= {lo!r})\n"
           f"    i2 = next(i for i, v in enumerate(c) if v / total >= {hi!r})\n"
           "    return " + ("(i2 - i1) * dt" if unit == "seconds" else "i2 - i1") + "\n")

    def args(n):
        a = accel(rng, n, rng.uniform(0.2, 6.0))
        return (a, pick(rng, (0.005, 0.01, 0.02))) if unit == "seconds" else (a,)
    return Variant(entry, spec, ref(src, entry), [args(12)], [args(rng.randint(40, 120)) for _ in range(6)],
                   f"significant_duration|{lo}|{hi}|{unit}", abs_tol=1e-9)


@kind("sta_lta", "event detection")
def _sta_lta(rng):
    mode = pick(rng, ("trigger_index", "trigger_time", "ratio"))
    cf = pick(rng, ("energy", "abs"))
    fixed = pick(rng, (None, 3.0, 4.0)) if mode != "ratio" else None
    cfd = "x[i]²" if cf == "energy" else "|x[i]|"
    cfe = "v * v" if cf == "energy" else "abs(v)"
    common = (f"Use the characteristic function c[i] = {cfd}. With trailing windows ending at sample i (inclusive), "
              "STA[i] is the mean of c over the last ns samples and LTA[i] the mean over the last nl samples "
              "(ns < nl); both are defined only for i ≥ nl − 1.")
    thr_arg = "" if fixed else ", threshold"
    thr = repr(fixed) if fixed else "threshold"
    thr_txt = f"{fixed}" if fixed else "threshold"
    reach = f"{fixed}" if fixed else "a threshold"
    if mode == "ratio":
        entry = pick(rng, ("sta_lta_ratio", "sta_lta"))
        spec = phrase(rng, f"{entry}(x, ns, nl)", "compute the classic STA/LTA ratio of a trace", common,
                      "Return the list of STA[i]/LTA[i] for i = nl − 1, …, len(x) − 1 (0.0 where LTA[i] = 0).")
        src = (f"def ENTRY(x, ns, nl):\n    c = [{cfe} for v in x]\n    out = []\n"
               "    for i in range(nl - 1, len(c)):\n        sta = sum(c[i - ns + 1:i + 1]) / ns\n"
               "        lta = sum(c[i - nl + 1:i + 1]) / nl\n        out.append(sta / lta if lta > 0 else 0.0)\n"
               "    return out\n")
    elif mode == "trigger_index":
        entry = pick(rng, ("sta_lta_trigger", "first_trigger"))
        spec = phrase(rng, f"{entry}(x, ns, nl{thr_arg})",
                      f"return the first sample at which the STA/LTA ratio reaches {reach}",
                      common, f"Return the smallest i ≥ nl − 1 with LTA[i] > 0 and STA[i]/LTA[i] ≥ {thr_txt}, or −1 "
                      "if there is none.")
        src = (f"def ENTRY(x, ns, nl{thr_arg}):\n    c = [{cfe} for v in x]\n"
               "    for i in range(nl - 1, len(c)):\n        sta = sum(c[i - ns + 1:i + 1]) / ns\n"
               "        lta = sum(c[i - nl + 1:i + 1]) / nl\n"
               f"        if lta > 0 and sta / lta >= {thr}:\n            return i\n    return -1\n")
    else:
        entry = pick(rng, ("sta_lta_trigger_time", "trigger_onset_time"))
        spec = phrase(rng, f"{entry}(x, dt, sta, lta{thr_arg})",
                      f"return the time at which the STA/LTA ratio first reaches {reach}",
                      "`x` is sampled every `dt` seconds; the window lengths are ns = round(sta/dt) and "
                      "nl = round(lta/dt) samples (`sta` and `lta` in seconds).", common,
                      f"Return i·dt for the smallest i ≥ nl − 1 with LTA[i] > 0 and STA[i]/LTA[i] ≥ {thr_txt}, or "
                      "None if there is none.")
        src = (f"def ENTRY(x, dt, sta, lta{thr_arg}):\n    ns, nl = int(round(sta / dt)), int(round(lta / dt))\n"
               f"    c = [{cfe} for v in x]\n    for i in range(nl - 1, len(c)):\n"
               "        s = sum(c[i - ns + 1:i + 1]) / ns\n        lt = sum(c[i - nl + 1:i + 1]) / nl\n"
               f"        if lt > 0 and s / lt >= {thr}:\n            return i * dt\n    return None\n")

    def args(n, quiet=False):
        ns = rng.randint(2, 5)
        nl = rng.randint(4 * ns, 8 * ns)
        n = max(n, nl + 15)
        onset = rng.randint(nl + 3, n - 10)
        x = noise(rng, n, 0.05) if quiet else burst(rng, n, onset, noise_s=0.05, amp=rng.uniform(0.5, 3))
        t = () if fixed else (pick(rng, (2.5, 3.0, 3.5, 4.0)),)
        if mode == "ratio":
            return (x, ns, nl)
        if mode == "trigger_index":
            return (x, ns, nl, *t)
        dt = pick(rng, (0.01, 0.02, 0.05))
        return (x, dt, _r(ns * dt, 6), _r(nl * dt, 6), *t)
    ex = [args(30)]
    hid = [args(rng.randint(60, 120)) for _ in range(5)] + [args(80, quiet=True)]
    return Variant(entry, spec, ref(src, entry), ex, hid, f"sta_lta|{mode}|{cf}|{fixed}")


@kind("zero_crossings", "trace measures")
def _zero_crossings(rng):
    mode = pick(rng, ("count", "rate", "upward", "downward"))
    entry = pick(rng, {"count": ("zero_crossings", "count_zero_crossings"), "rate": ("zero_crossing_rate",),
                       "upward": ("upward_zero_crossings",), "downward": ("downward_zero_crossings",)}[mode])
    rule = {"count": "a crossing is a pair of consecutive samples whose product is negative",
            "rate": "a crossing is a pair of consecutive samples whose product is negative",
            "upward": "an upward crossing is a pair of consecutive samples with x[i−1] < 0 < x[i]",
            "downward": "a downward crossing is a pair of consecutive samples with x[i−1] > 0 > x[i]"}[mode]
    ret = {"count": "Return the number of crossings.",
           "rate": "Return the number of crossings divided by the record length (len(x) − 1)·dt, in crossings "
                   "per second.",
           "upward": "Return the number of upward crossings.",
           "downward": "Return the number of downward crossings."}[mode]
    sig = f"{entry}(x, dt)" if mode == "rate" else f"{entry}(x)"
    spec = phrase(rng, sig, "count zero crossings in a trace", f"`x` is a list of at least 2 floats; {rule}.", ret)
    cond = {"count": "x[i - 1] * x[i] < 0", "rate": "x[i - 1] * x[i] < 0", "upward": "x[i - 1] < 0 < x[i]",
            "downward": "x[i - 1] > 0 > x[i]"}[mode]
    body = f"    n = sum(1 for i in range(1, len(x)) if {cond})\n"
    src = (f"def ENTRY(x, dt):\n{body}    return n / ((len(x) - 1) * dt)\n" if mode == "rate"
           else f"def ENTRY(x):\n{body}    return n\n")

    def args(n):
        x = trace(rng, n)
        return (x, pick(rng, (0.01, 0.02, 0.05))) if mode == "rate" else (x,)
    return Variant(entry, spec, ref(src, entry), [args(8)], [args(rng.randint(20, 100)) for _ in range(6)],
                   f"zero_crossings|{mode}")


@kind("xcorr_lag", "waveform comparison")
def _xcorr(rng):
    unit = pick(rng, ("samples", "seconds"))
    with_cc = rng.random() < 0.4
    entry = pick(rng, ("xcorr_lag", "cross_correlation_shift", "time_shift"))
    sig = f"{entry}(x, y, maxlag)" if unit == "samples" else f"{entry}(x, y, maxlag, dt)"
    res = "L" if unit == "samples" else "L·dt in seconds (`maxlag` is in samples)"
    spec = phrase(rng, sig, "find the lag that maximizes the cross-correlation of two traces",
                  "`x` and `y` are lists of the same length n, not all zero. For each integer lag L with "
                  "−maxlag ≤ L ≤ maxlag, c(L) = Σ x[i]·y[i+L] over all i with 0 ≤ i < n and 0 ≤ i+L < n. Find the L "
                  "with the largest c(L); if several tie, take the one with the smallest |L|, and the negative one "
                  "of a ± pair.",
                  (f"Return the tuple ({res}, c(L) / sqrt(Σx²·Σy²)), the second being the normalized correlation at "
                   "that lag with both sums over all n samples." if with_cc else f"Return {res}") +
                  (" (L is an int)." if unit == "samples" and not with_cc else "."))
    lag = "L" if unit == "samples" else "L * dt"
    tail = ("    norm = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))\n"
            f"    return ({lag}, best[0][0] / norm)\n" if with_cc else f"    return {lag}\n")
    src = ("import math\n\n\ndef ENTRY(x, y, maxlag" + (", dt" if unit == "seconds" else "") + "):\n"
           "    n = len(x)\n    best = None\n    for L in range(-maxlag, maxlag + 1):\n"
           "        c = sum(x[i] * y[i + L] for i in range(n) if 0 <= i + L < n)\n"
           "        k = (c, -abs(L), -L)\n        if best is None or k > best[0]:\n            best = (k, L)\n"
           "    L = best[1]\n" + tail)

    def args(n):
        x = burst(rng, n, rng.randint(n // 5, n // 3), noise_s=0.02)
        s = rng.randint(-n // 6, n // 6)
        y = [_r((x[i - s] if 0 <= i - s < n else 0.0) + rng.gauss(0, 0.01)) for i in range(n)]
        maxlag = abs(s) + rng.randint(2, 8)
        return (x, y, maxlag) if unit == "samples" else (x, y, maxlag, pick(rng, (0.01, 0.02, 0.05)))
    return Variant(entry, spec, ref(src, entry), [args(12)], [args(rng.randint(40, 100)) for _ in range(6)],
                   f"xcorr_lag|{unit}|{with_cc}")


@kind("dominant_frequency", "spectral analysis")
def _dominant_frequency(rng):
    ret = pick(rng, ("frequency", "frequency_amplitude", "period"))
    names = ("dominant_period",) if ret == "period" else ("dominant_frequency", "peak_frequency", "spectral_peak")
    entry = pick(rng, names)
    spec = phrase(rng, f"{entry}(x, dt)", "find the dominant " + ("period" if ret == "period" else "frequency") +
                  " of a trace from its discrete Fourier transform",
                  "`x` has n ≥ 4 samples spaced `dt` seconds apart. Compute X[k] = Σ_j x[j]·exp(−2πi·j·k/n) for "
                  "k = 1 … floor(n/2) (k = 0 is excluded) and find the k with the largest |X[k]| (the smallest k "
                  "if several tie). Its frequency is k/(n·dt) Hz.",
                  {"frequency": "Return that frequency in Hz.",
                   "frequency_amplitude": "Return the tuple (frequency, amplitude) with amplitude = 2·|X[k]|/n.",
                   "period": "Return the corresponding period n·dt/k in seconds."}[ret])
    tail = {"frequency": "f", "frequency_amplitude": "(f, 2.0 * best / n)", "period": "1.0 / f"}[ret]
    src = ("import cmath\nimport math\n\n\ndef ENTRY(x, dt):\n    n = len(x)\n    best_k, best = 1, -1.0\n"
           "    for k in range(1, n // 2 + 1):\n"
           "        s = sum(x[j] * cmath.exp(-2j * math.pi * j * k / n) for j in range(n))\n"
           "        if abs(s) > best + 1e-12 * max(1.0, best):\n            best_k, best = k, abs(s)\n"
           "    f = best_k / (n * dt)\n    return " + tail + "\n")

    def args(n):
        k0 = rng.randint(2, n // 2 - 2)
        a = rng.uniform(0.5, 3)
        ph = rng.uniform(0, 2 * math.pi)
        x = [_r(a * math.sin(2 * math.pi * k0 * j / n + ph) + rng.gauss(0, 0.1)) for j in range(n)]
        return (x, pick(rng, (0.01, 0.02, 0.05, 0.1)))
    return Variant(entry, spec, ref(src, entry), [args(8)], [args(rng.randint(24, 96)) for _ in range(6)],
                   f"dominant_frequency|{ret}")


@kind("rotate_components", "three-component processing")
def _rotate(rng):
    direction = pick(rng, ("ne_to_rt", "rt_to_ne"))
    angle = pick(rng, ("baz", "az"))
    ang_arg = angle
    ang_txt = ("`baz` is the back-azimuth in degrees (station to event). With b = baz in radians"
               if angle == "baz" else
               "`az` is the event-to-station azimuth in degrees; the back-azimuth is baz = az + 180°. With "
               "b = baz in radians")
    bexpr = "math.radians(baz)" if angle == "baz" else "math.radians(az + 180.0)"
    if direction == "ne_to_rt":
        entry = pick(rng, ("rotate_ne_rt", "ne_to_rt"))
        spec = phrase(rng, f"{entry}(n, e, {ang_arg})",
                      "rotate north and east components into radial and transverse",
                      f"`n` and `e` are equal-length lists; {ang_txt}: r[i] = −e[i]·sin b − n[i]·cos b and "
                      "t[i] = −e[i]·cos b + n[i]·sin b. Return the tuple (r, t) of two lists.")
        src = (f"import math\n\n\ndef ENTRY(n, e, {ang_arg}):\n    b = {bexpr}\n"
               "    r = [-ei * math.sin(b) - ni * math.cos(b) for ni, ei in zip(n, e)]\n"
               "    t = [-ei * math.cos(b) + ni * math.sin(b) for ni, ei in zip(n, e)]\n    return (r, t)\n")
    else:
        entry = pick(rng, ("rotate_rt_ne", "rt_to_ne"))
        spec = phrase(rng, f"{entry}(r, t, {ang_arg})",
                      "rotate radial and transverse components back to north and east",
                      f"`r` and `t` are equal-length lists; {ang_txt}, the forward rotation was "
                      "r = −e·sin b − n·cos b, t = −e·cos b + n·sin b; invert it and return the tuple (n, e) of two "
                      "lists.")
        src = (f"import math\n\n\ndef ENTRY(r, t, {ang_arg}):\n    b = {bexpr}\n"
               "    n = [-ri * math.cos(b) + ti * math.sin(b) for ri, ti in zip(r, t)]\n"
               "    e = [-ri * math.sin(b) - ti * math.cos(b) for ri, ti in zip(r, t)]\n    return (n, e)\n")

    def args(n):
        return (trace(rng, n), trace(rng, n), _r(rng.uniform(0, 360), 2))
    return Variant(entry, spec, ref(src, entry), [args(4)], [args(rng.randint(10, 60)) for _ in range(6)],
                   f"rotate_components|{direction}|{angle}")


@kind("great_circle", "source–station geometry")
def _great_circle(rng):
    mode = pick(rng, ("km", "deg", "many_km", "many_deg"))
    hav = ("Use the haversine formula: a = sin²(Δφ/2) + cos φ1·cos φ2·sin²(Δλ/2), central angle "
           "c = 2·atan2(√a, √(1−a)), with latitudes φ and longitudes λ in radians.")
    res = f"R·c in km with R = {R_EARTH} km" if mode.endswith("km") else "the central angle c in degrees"
    rexpr = f"{R_EARTH!r} * c" if mode.endswith("km") else "math.degrees(c)"
    core = ("    p1, p2 = math.radians(LA1), math.radians(LA2)\n    dp, dl = p2 - p1, math.radians(LO2 - LO1)\n"
            "    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2\n"
            "    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))\n")
    if mode.startswith("many"):
        entry = pick(rng, ("station_distances", "distances_to_stations"))
        spec = phrase(rng, f"{entry}(lat0, lon0, stations)",
                      "return the epicentral distance from an event to each station",
                      "The event is at (`lat0`, `lon0`) in degrees; `stations` is a list of (lat, lon) pairs in "
                      "degrees.", hav, f"Return a list with {res} for each station, in the order of `stations`.")
        body = core.replace("LA1", "lat0").replace("LO1", "lon0").replace("LA2", "lat").replace("LO2", "lon")
        src = ("import math\n\n\ndef ENTRY(lat0, lon0, stations):\n    out = []\n    for lat, lon in stations:\n"
               + "".join("    " + line + "\n" for line in body.splitlines()) + f"        out.append({rexpr})\n"
               "    return out\n")

        def args(k):
            la, lo = latlon(rng)
            return (la, lo, [latlon(rng) for _ in range(k)])
        ex, hid = [args(2)], [args(rng.randint(3, 10)) for _ in range(6)]
    else:
        names = ("epicentral_distance", "haversine") if mode == "km" else ("central_angle", "epicentral_distance_deg")
        entry = pick(rng, names)
        spec = phrase(rng, f"{entry}(lat1, lon1, lat2, lon2)",
                      "return the great-circle distance between two points given in degrees", hav, f"Return {res}.")
        body = core.replace("LA1", "lat1").replace("LO1", "lon1").replace("LA2", "lat2").replace("LO2", "lon2")
        src = "import math\n\n\ndef ENTRY(lat1, lon1, lat2, lon2):\n" + body + f"    return {rexpr}\n"

        def args(_):
            return (*latlon(rng), *latlon(rng))
        ex, hid = [args(0)], [args(0) for _ in range(6)]
    return Variant(entry, spec, ref(src, entry), ex, hid, f"great_circle|{mode}")


@kind("azimuth", "source–station geometry")
def _azimuth(rng):
    back = rng.random() < 0.5
    signed = rng.random() < 0.4
    entry = pick(rng, ("back_azimuth",) if back else ("azimuth", "initial_bearing"))
    frm, to = ("station (lat2, lon2)", "event (lat1, lon1)") if back else ("point 1", "point 2")
    rng_txt = "in (−180, 180]" if signed else "in [0, 360)"
    spec = phrase(rng, f"{entry}(lat1, lon1, lat2, lon2)",
                  f"return the {'back-azimuth' if back else 'azimuth'} in degrees",
                  "Coordinates are in degrees; point 1 is "
                  + ("the event and point 2 the station. " if back else "the start and point 2 the target. ") +
                  f"Compute the initial great-circle bearing from the {frm} toward the {to}: "
                  "θ = atan2(sin Δλ·cos φB, cos φA·sin φB − sin φA·cos φB·cos Δλ), with A the start, B the target, "
                  f"Δλ = λB − λA (radians). Return θ in degrees {rng_txt}.")
    tail = "math.degrees(th)" if signed else "math.degrees(th) % 360.0"
    src = ("import math\n\n\ndef ENTRY(lat1, lon1, lat2, lon2):\n"
           + ("    lat1, lon1, lat2, lon2 = lat2, lon2, lat1, lon1\n" if back else "") +
           "    a, b = math.radians(lat1), math.radians(lat2)\n    dl = math.radians(lon2 - lon1)\n"
           "    th = math.atan2(math.sin(dl) * math.cos(b),\n"
           "                    math.cos(a) * math.sin(b) - math.sin(a) * math.cos(b) * math.cos(dl))\n"
           f"    return {tail}\n")

    def args():
        return (*latlon(rng), *latlon(rng))
    return Variant(entry, spec, ref(src, entry), [args()], [args() for _ in range(6)], f"azimuth|{back}|{signed}",
                   abs_tol=1e-7)


@kind("wadati_fit", "earthquake location")
def _wadati(rng):
    both = rng.random() < 0.6
    entry = pick(rng, ("wadati", "wadati_fit", "vp_vs_from_wadati"))
    spec = phrase(rng, f"{entry}(tp, ts)", "estimate Vp/Vs" + (" and the origin time" if both else "") +
                  " from a Wadati diagram",
                  "`tp` and `ts` are equal-length lists (at least 2) of P and S arrival times in seconds at "
                  "different stations. Fit y = m·x + c by ordinary least squares with x = tp and y = ts − tp.",
                  "Return the tuple (1 + m, −c/m): Vp/Vs and the origin time in seconds." if both else
                  "Return Vp/Vs = 1 + m.")
    src = '''
def ENTRY(tp, ts):
    y = [s - p for p, s in zip(tp, ts)]
    n = len(tp)
    mx, my = sum(tp) / n, sum(y) / n
    m = sum((p - mx) * (v - my) for p, v in zip(tp, y)) / sum((p - mx) ** 2 for p in tp)
    c = my - m * mx
    return (1.0 + m, -c / m) if BOTH else 1.0 + m
'''.replace("BOTH", repr(both))

    def args(k):
        t0, vp = rng.uniform(0, 60), rng.uniform(5.5, 7.5)
        vs = vp / rng.uniform(1.65, 1.85)
        ds = sorted(rng.uniform(10, 300) for _ in range(k))
        tp = [_r(t0 + d / vp + rng.gauss(0, 0.05), 3) for d in ds]
        ts = [_r(t0 + d / vs + rng.gauss(0, 0.08), 3) for d in ds]
        return (tp, ts)
    return Variant(entry, spec, ref(src, entry), [args(3)], [args(rng.randint(4, 12)) for _ in range(6)],
                   f"wadati_fit|{both}")


@kind("b_value", "earthquake statistics")
def _b_value(rng):
    corr = rng.random() < 0.6
    with_a = rng.random() < 0.4
    entry = pick(rng, ("gutenberg_richter_fit", "gr_parameters") if with_a else ("b_value", "aki_b_value",
                                                                                 "b_value_mle"))
    sig = f"{entry}(mags, mc, dm)" if corr else f"{entry}(mags, mc)"
    bdef = ("b = log10(e) / (mean(M) − (mc − dm/2)), Aki's estimator with Utsu's correction for magnitudes "
            "binned to width `dm`" if corr else "b = log10(e) / (mean(M) − mc) (Aki's estimator)")
    spec = phrase(rng, sig, "estimate the Gutenberg–Richter " + ("a- and b-values" if with_a else "b-value") +
                  " by maximum likelihood",
                  "Use only the magnitudes M in `mags` with M ≥ mc − 1e-9 (at least one exists); N is their number.",
                  f"Return the tuple (a, b) with {bdef} and a = log10(N) + b·mc." if with_a else f"Return {bdef}.")
    shift = "dm / 2.0" if corr else "0.0"
    tail = "(math.log10(len(m)) + b * mc, b)" if with_a else "b"
    src = ("import math\n\n\ndef ENTRY(mags, mc" + (", dm" if corr else "") + "):\n"
           "    m = [x for x in mags if x >= mc - 1e-9]\n"
           f"    b = math.log10(math.e) / (sum(m) / len(m) - (mc - {shift}))\n    return {tail}\n")

    def args(n):
        dm = pick(rng, (0.1, 0.05))
        mc = _r(round(rng.uniform(1.0, 3.0) / dm) * dm, 2)
        mags = gr_mags(rng, n, mc, rng.uniform(0.7, 1.3), dm)
        return (mags, mc, dm) if corr else (mags, mc)
    return Variant(entry, spec, ref(src, entry), [args(8)], [args(rng.randint(30, 120)) for _ in range(6)],
                   f"b_value|{corr}|{with_a}")


@kind("time_averaged_vs", "site characterization")
def _vsz(rng):
    z = pick(rng, (None, 30.0, 10.0, 20.0))
    ret = pick(rng, ("velocity", "time"))
    if ret == "velocity":
        entry = pick(rng, ("vs30", "time_averaged_vs") if z in (None, 30.0) else (f"vs{int(z)}", "time_averaged_vs"))
    else:
        entry = pick(rng, ("shear_travel_time", "vertical_s_time"))
    sig = f"{entry}(h, v, z)" if z is None else f"{entry}(h, v)"
    zd = "`z` metres" if z is None else f"{int(z)} m"
    layers = ("`v` lists the layer shear-wave velocities in m/s from the surface down; `h` lists the thicknesses "
              "in m of all layers except the last, which is a half-space extending to any depth (so "
              "len(h) = len(v) − 1).")
    spec = phrase(rng, sig, (f"compute the time-averaged shear-wave velocity of the top {zd}" if ret == "velocity"
                             else f"compute the vertical shear-wave travel time through the top {zd}"), layers,
                  (f"Return D / Σ(h_i'/v_i), where D = {zd} and h_i' is the part of layer i above depth D." if
                   ret == "velocity" else
                   f"Return Σ(h_i'/v_i) in seconds, where h_i' is the part of layer i above depth {zd}."))
    src = ("def ENTRY(h, v, z" + ("" if z is None else f"={z}") + "):\n    t, top = 0.0, 0.0\n"
           "    for i, vi in enumerate(v):\n        bottom = top + h[i] if i < len(h) else float(\"inf\")\n"
           "        part = min(bottom, z) - top\n        if part > 0:\n            t += part / vi\n"
           "        top = bottom\n        if top >= z:\n            break\n"
           "    return " + ("z / t" if ret == "velocity" else "t") + "\n")

    def args():
        k = rng.randint(1, 5)
        hs = [_r(rng.uniform(1, 15), 1) for _ in range(k)]
        vs = sorted(_r(rng.uniform(120, 1200), 0) for _ in range(k + 1))
        return (hs, vs, pick(rng, (10.0, 20.0, 30.0, 50.0))) if z is None else (hs, vs)
    return Variant(entry, spec, ref(src, entry), [args()], [args() for _ in range(6)], f"time_averaged_vs|{z}|{ret}")


@kind("travel_time_table", "travel times")
def _tt_table(rng):
    many = rng.random() < 0.4
    outside = pick(rng, ("none", "clamp"))
    entry = pick(rng, ("interp_travel_time", "travel_time_from_table"))
    sig = f"{entry}(dists, times, queries)" if many else f"{entry}(dists, times, d)"
    out_txt = ("outside that range return None" if outside == "none" else
               "outside that range return the time at the nearest end of the table")
    spec = phrase(rng, sig, "interpolate a travel-time table linearly",
                  "`dists` is a strictly increasing list of distances (at least 2) and `times` the travel times "
                  "at those distances. For a distance inside [dists[0], dists[-1]], return the linear interpolation "
                  f"between the two neighbouring entries (the table value at a node); {out_txt}.",
                  "`queries` is a list of distances: return a list of results." if many else "")
    lo, hi = ("None", "None") if outside == "none" else ("times[0]", "times[-1]")
    src = ("def _one(dists, times, d):\n"
           f"    if d < dists[0]:\n        return {lo}\n    if d > dists[-1]:\n        return {hi}\n"
           "    for i in range(1, len(dists)):\n        if d <= dists[i]:\n"
           "            f = (d - dists[i - 1]) / (dists[i] - dists[i - 1])\n"
           "            return times[i - 1] + f * (times[i] - times[i - 1])\n    return times[-1]\n\n\n"
           + ("def ENTRY(dists, times, queries):\n    return [_one(dists, times, d) for d in queries]\n" if many
              else "def ENTRY(dists, times, d):\n    return _one(dists, times, d)\n"))

    def table(k):
        ds = sorted({_r(rng.uniform(0, 100) * j, 1) for j in range(1, k + 1)})
        v = rng.uniform(5, 9)
        ts = [_r(d / v + 0.002 * d ** 1.2, 3) for d in ds]
        return ds, ts

    def args():
        ds, ts = table(rng.randint(4, 10))
        qs = [_r(rng.uniform(ds[0] - 5, ds[-1] + 5), 2) for _ in range(rng.randint(3, 6))] + [ds[1]]
        return (ds, ts, qs) if many else (ds, ts, pick(rng, qs))
    hid = [args() for _ in range(6)]
    if not many:      # make sure both out-of-range branches are tested
        ds, ts = table(5)
        hid += [(ds, ts, _r(ds[-1] + 3, 2)), (ds, ts, _r(ds[0] - 3, 2))]
    return Variant(entry, spec, ref(src, entry), [args()], hid, f"travel_time_table|{many}|{outside}")


@kind("threshold_picker", "event detection")
def _picker(rng):
    as_time = rng.random() < 0.5
    k = pick(rng, (3.0, 4.0, 5.0, 6.0))
    entry = pick(rng, ("simple_picker", "threshold_pick", "first_arrival"))
    sig = f"{entry}(x, n0, dt)" if as_time else f"{entry}(x, n0)"
    spec = phrase(rng, sig, "pick the first arrival where a trace leaves its noise level",
                  "The first `n0` samples of `x` are noise: compute their mean μ and population standard deviation "
                  f"σ (divide by n0). Find the first index i ≥ n0 with |x[i] − μ| > {k}·σ.",
                  "Return i·dt in seconds, or None if no sample qualifies." if as_time else
                  "Return i, or −1 if no sample qualifies.")
    src = '''
import math


def ENTRY(x, n0, *rest):
    mu = sum(x[:n0]) / n0
    sd = math.sqrt(sum((v - mu) ** 2 for v in x[:n0]) / n0)
    for i in range(n0, len(x)):
        if abs(x[i] - mu) > K * sd:
            return HIT
    return MISS
'''.replace("K", repr(k))
    src = src.replace(", *rest", ", dt" if as_time else "").replace("HIT", "i * dt" if as_time else "i").replace(
        "MISS", "None" if as_time else "-1")

    def args(n, quiet=False):
        n0 = rng.randint(10, 25)
        n = max(n, n0 + 12)
        x = noise(rng, n, 0.05) if quiet else burst(rng, n, rng.randint(n0 + 3, n - 5), amp=rng.uniform(0.5, 2))
        return (x, n0, pick(rng, (0.01, 0.02))) if as_time else (x, n0)
    return Variant(entry, spec, ref(src, entry), [args(30)], [args(rng.randint(50, 110)) for _ in range(5)]
                   + [args(60, quiet=True)], f"threshold_picker|{as_time}|{k}")


@kind("moving_average", "trace preprocessing")
def _moving_average(rng):
    fixed = pick(rng, (None, 3, 5, 7))
    stat = pick(rng, ("mean", "median"))
    names = ("moving_average", "smooth", "running_mean") if stat == "mean" else ("median_filter", "running_median")
    entry = pick(rng, names)
    sig = f"{entry}(x)" if fixed else f"{entry}(x, w)"
    wd = f"w = {fixed}" if fixed else "an odd window length `w` ≥ 1"
    spec = phrase(rng, sig, f"smooth a trace with a centred moving {stat} of {wd}",
                  f"With h = (w − 1)/2, output sample i is the {stat} of x[j] for max(0, i−h) ≤ j ≤ min(n−1, i+h) "
                  "(the window shrinks at both ends)" + (", and the median of an even number of values is the "
                                                         "mean of the two middle ones" if stat == "median" else "")
                  + ". Return a list of the same length as `x`.")
    agg = ("sum(seg) / len(seg)" if stat == "mean" else
           "(s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2]))")
    src = ("def ENTRY(x, w" + (f"={fixed}" if fixed else "") + "):\n    h = (w - 1) // 2\n    n = len(x)\n"
           "    out = []\n    for i in range(n):\n        seg = x[max(0, i - h):min(n, i + h + 1)]\n"
           + ("        s = sorted(seg)\n" if stat == "median" else "") + f"        out.append({agg})\n    return out\n")

    def args(n):
        x = trace(rng, n)
        return (x,) if fixed else (x, pick(rng, (3, 5, 7, 9)))
    return Variant(entry, spec, ref(src, entry), [args(6)], [args(rng.randint(15, 80)) for _ in range(6)],
                   f"moving_average|{fixed}|{stat}")


@kind("differentiate_integrate", "trace preprocessing")
def _diff_int(rng):
    mode = pick(rng, ("differentiate", "integrate", "second_derivative"))
    if mode == "differentiate":
        entry = pick(rng, ("differentiate", "time_derivative"))
        spec = phrase(rng, f"{entry}(x, dt)", "differentiate a trace sampled every `dt` seconds",
                      "`x` has at least 2 samples. Use central differences (x[i+1] − x[i−1])/(2·dt) inside, the "
                      "forward difference (x[1] − x[0])/dt at the first sample and the backward difference "
                      "(x[n−1] − x[n−2])/dt at the last. Return a list of the same length.")
        src = ("def ENTRY(x, dt):\n    n = len(x)\n    out = [(x[1] - x[0]) / dt]\n    for i in range(1, n - 1):\n"
               "        out.append((x[i + 1] - x[i - 1]) / (2 * dt))\n    out.append((x[n - 1] - x[n - 2]) / dt)\n"
               "    return out\n")
    elif mode == "integrate":
        entry = pick(rng, ("integrate", "cumulative_trapezoid"))
        spec = phrase(rng, f"{entry}(x, dt)", "integrate a trace sampled every `dt` seconds",
                      "Return the cumulative trapezoidal integral y with y[0] = 0 and "
                      "y[i] = y[i−1] + 0.5·(x[i−1] + x[i])·dt, a list of the same length as `x` (non-empty).")
        src = ("def ENTRY(x, dt):\n    y = [0.0]\n    for i in range(1, len(x)):\n"
               "        y.append(y[-1] + 0.5 * (x[i - 1] + x[i]) * dt)\n    return y\n")
    else:
        entry = pick(rng, ("second_derivative", "curvature"))
        spec = phrase(rng, f"{entry}(x, dt)", "compute the second time derivative of a trace sampled every `dt` "
                      "seconds", "`x` has at least 3 samples. Return the list (x[i+1] − 2·x[i] + x[i−1])/dt² for "
                      "i = 1 … n−2 (length n − 2).")
        src = ("def ENTRY(x, dt):\n"
               "    return [(x[i + 1] - 2 * x[i] + x[i - 1]) / (dt * dt) for i in range(1, len(x) - 1)]\n")

    def args(n):
        return (trace(rng, n), pick(rng, (0.01, 0.02, 0.05, 0.1)))
    return Variant(entry, spec, ref(src, entry), [args(5)], [args(rng.randint(10, 80)) for _ in range(6)],
                   f"differentiate_integrate|{mode}", abs_scale=1e-6)


@kind("magnitude_counts", "earthquake statistics")
def _mag_counts(rng):
    mode = pick(rng, ("cumulative", "binned"))
    entry = pick(rng, ("cumulative_counts", "gr_counts") if mode == "cumulative" else ("magnitude_histogram",
                                                                                       "binned_counts"))
    if mode == "cumulative":
        spec = phrase(rng, f"{entry}(mags, thresholds)",
                      "count earthquakes at or above each magnitude threshold",
                      "Return a list with, for each m in `thresholds`, the number of values M in `mags` with "
                      "M ≥ m − 1e-9.")
        src = '''
def ENTRY(mags, thresholds):
    return [sum(1 for x in mags if x >= m - 1e-9) for m in thresholds]
'''
    else:
        spec = phrase(rng, f"{entry}(mags, m0, dm, nbins)", "bin earthquake magnitudes",
                      "Bin k (k = 0 … nbins−1) covers m0 + k·dm ≤ M < m0 + (k+1)·dm; compare with a 1e-9 "
                      "tolerance (M ≥ edge − 1e-9 and M < next edge − 1e-9). Return the list of nbins counts; "
                      "magnitudes outside all bins are ignored.")
        src = '''
def ENTRY(mags, m0, dm, nbins):
    out = [0] * nbins
    for x in mags:
        for k in range(nbins):
            lo, hi = m0 + k * dm, m0 + (k + 1) * dm
            if x >= lo - 1e-9 and x < hi - 1e-9:
                out[k] += 1
                break
    return out
'''

    def args(n):
        mc = _r(rng.uniform(1.0, 3.0), 1)
        mags = gr_mags(rng, n, mc, rng.uniform(0.8, 1.2), 0.1)
        if mode == "cumulative":
            return (mags, [_r(mc + 0.5 * j, 1) for j in range(rng.randint(2, 6))])
        return (mags, mc, 0.5, rng.randint(3, 6))
    return Variant(entry, spec, ref(src, entry), [args(8)], [args(rng.randint(20, 120)) for _ in range(6)],
                   f"magnitude_counts|{mode}")


@kind("total_moment", "earthquake source")
def _total_moment(rng):
    out = pick(rng, ("mw", "nm", "dyne_cm"))
    entry = pick(rng, {"mw": ("equivalent_magnitude", "cumulative_mw"), "nm": ("total_moment", "summed_moment"),
                       "dyne_cm": ("total_moment_dyne_cm",)}[out])
    spec = phrase(rng, f"{entry}(mws)", "sum the seismic moments of a list of earthquakes",
                  "Each moment magnitude Mw in `mws` has moment M0 = 10^(1.5·Mw + 9.1) N·m.",
                  {"mw": "Return the moment magnitude of the summed moment, Mw = (2/3)·(log10 ΣM0 − 9.1).",
                   "nm": "Return the summed moment in N·m.",
                   "dyne_cm": "Return the summed moment in dyne·cm (1 N·m = 1e7 dyne·cm)."}[out])
    tail = {"mw": "(2.0 / 3.0) * (math.log10(s) - 9.1)", "nm": "s", "dyne_cm": "s * 1e7"}[out]
    src = f'''
import math


def ENTRY(mws):
    s = sum(10 ** (1.5 * m + 9.1) for m in mws)
    return {tail}
'''

    def args(n):
        return ([_r(rng.uniform(2.0, 7.5), 1) for _ in range(n)],)
    return Variant(entry, spec, ref(src, entry), [args(3)], [args(rng.randint(2, 30)) for _ in range(6)],
                   f"total_moment|{out}")


@kind("clipping", "data quality")
def _clipping(rng):
    mode = pick(rng, ("fraction", "longest_run", "runs"))
    entry = pick(rng, {"fraction": ("clipped_fraction", "clip_fraction"), "longest_run": ("longest_clipped_run",),
                       "runs": ("clipped_runs", "find_clipping")}[mode])
    spec = phrase(rng, f"{entry}(x, level)", "detect clipping in a digitized trace",
                  "A sample is clipped when |x[i]| ≥ level.",
                  {"fraction": "Return the fraction of clipped samples (0.0 for an empty list).",
                   "longest_run": "Return the length of the longest run of consecutive clipped samples (0 if none).",
                   "runs": "Return a list of (start index, length) tuples, one per maximal run of consecutive "
                           "clipped samples, in order (an empty list if none)."}[mode])
    src = {"fraction": "def ENTRY(x, level):\n    if not x:\n        return 0.0\n"
                       "    return sum(1 for v in x if abs(v) >= level) / len(x)\n",
           "longest_run": "def ENTRY(x, level):\n    best = run = 0\n    for v in x:\n"
                          "        run = run + 1 if abs(v) >= level else 0\n        best = max(best, run)\n"
                          "    return best\n",
           "runs": "def ENTRY(x, level):\n    out, start = [], None\n    for i, v in enumerate(list(x) + [0.0]):\n"
                   "        hit = i < len(x) and abs(v) >= level\n        if hit and start is None:\n"
                   "            start = i\n        elif not hit and start is not None:\n"
                   "            out.append((start, i - start))\n            start = None\n    return out\n"}[mode]

    def args(n):
        lvl = _r(rng.uniform(0.6, 1.2), 3)
        x = [max(-lvl, min(lvl, v)) for v in trace(rng, n, amp=rng.uniform(0.8, 2.5))]
        return (x, lvl)
    return Variant(entry, spec, ref(src, entry), [args(8)], [args(rng.randint(20, 100)) for _ in range(6)],
                   f"clipping|{mode}")


@kind("rms_envelope", "amplitude measures")
def _rms_envelope(rng):
    centred = rng.random() < 0.5
    entry = pick(rng, ("rms_envelope", "running_rms") if not centred else ("centred_rms_envelope", "moving_rms"))
    win = ("x[j]² for max(0, i−w+1) ≤ j ≤ i (`w` ≥ 1 samples; the window is shorter at the start)" if not centred
           else "x[j]² for max(0, i−h) ≤ j ≤ min(n−1, i+h) with h = (w − 1)/2 for an odd `w` (the window shrinks "
                "at both ends)")
    spec = phrase(rng, f"{entry}(x, w)", f"compute a {'centred' if centred else 'trailing'} moving-RMS envelope",
                  f"Output sample i is sqrt of the mean of {win}. Return a list of the same length as `x`.")
    seg = "x[max(0, i - w + 1):i + 1]" if not centred else "x[max(0, i - (w - 1) // 2):i + (w - 1) // 2 + 1]"
    src = ("import math\n\n\ndef ENTRY(x, w):\n    out = []\n    for i in range(len(x)):\n"
           f"        seg = {seg}\n        out.append(math.sqrt(sum(v * v for v in seg) / len(seg)))\n    return out\n")

    def args(n):
        return (trace(rng, n), pick(rng, (3, 5, 7, 9)) if centred else rng.randint(2, 10))
    return Variant(entry, spec, ref(src, entry), [args(6)], [args(rng.randint(15, 80)) for _ in range(6)],
                   f"rms_envelope|{centred}")


@kind("resample_linear", "trace preprocessing")
def _resample(rng):
    method = pick(rng, ("linear", "nearest"))
    entry = pick(rng, ("resample_linear", "resample") if method == "linear" else ("resample_nearest",))
    how = ("the linear interpolation of `x` at t (with j = floor(t/dt_in), between x[j] and x[j+1]; use x[n−1] "
           "when j ≥ n−1)" if method == "linear" else
           "x[j] with j = floor(t/dt_in + 0.5), the nearest sample (capped at n−1)")
    spec = phrase(rng, f"{entry}(x, dt_in, dt_out)", f"resample a trace ({method} interpolation)",
                  "Sample i of `x` (at least 2 samples) is at time i·dt_in. Output sample k is at t = k·dt_out for "
                  f"k = 0 … K with K = floor((n−1)·dt_in/dt_out + 1e-9); its value is {how}. Return the list of "
                  "output samples.")
    if method == "linear":
        body = ("        j = int(math.floor(t / dt_in))\n        if j >= n - 1:\n            out.append(x[n - 1])\n"
                "            continue\n        f = t / dt_in - j\n        out.append(x[j] + f * (x[j + 1] - x[j]))\n")
    else:
        body = "        j = min(n - 1, int(math.floor(t / dt_in + 0.5)))\n        out.append(x[j])\n"
    src = ("import math\n\n\ndef ENTRY(x, dt_in, dt_out):\n    n = len(x)\n"
           "    K = int(math.floor((n - 1) * dt_in / dt_out + 1e-9))\n    out = []\n    for k in range(K + 1):\n"
           "        t = k * dt_out\n" + body + "    return out\n")
    ratios = ((0.01, 2.5), (0.01, 2.0), (0.02, 0.5), (0.01, 0.4), (0.05, 3.0)) if method == "linear" else \
        ((0.01, 2.0), (0.01, 3.0), (0.02, 4.0), (0.05, 2.0))

    def args(n):
        dt_in, ratio = pick(rng, ratios)
        return (trace(rng, n), dt_in, _r(dt_in * ratio, 6))
    return Variant(entry, spec, ref(src, entry), [args(5)], [args(rng.randint(10, 60)) for _ in range(6)],
                   f"resample_linear|{method}", rel_tol=1e-6, abs_scale=1e-6)


@kind("peak_to_peak", "amplitude measures")
def _p2p(rng):
    half = rng.random() < 0.4
    by_index = rng.random() < 0.4
    entry = pick(rng, ("half_peak_to_peak", "window_amplitude") if half else ("peak_to_peak", "window_peak_to_peak"))
    sig = f"{entry}(x, i0, i1)" if by_index else f"{entry}(x, dt, t_start, t_end)"
    win = ("Use the samples x[i0:i1] (a Python slice)" if by_index else
           "Sample i is at time i·dt. Use the samples with t_start ≤ i·dt < t_end")
    spec = phrase(rng, sig, "return the " + ("half " if half else "") + "peak-to-peak amplitude of a trace inside "
                  "a window", f"{win} and return " + ("(max − min)/2" if half else "max − min") +
                  "; return 0.0 if no sample is in the window.")
    sel = "x[i0:i1]" if by_index else "[v for i, v in enumerate(x) if t_start <= i * dt < t_end]"
    src = ("def ENTRY(" + ("x, i0, i1" if by_index else "x, dt, t_start, t_end") + "):\n"
           f"    sel = {sel}\n    if not sel:\n        return 0.0\n"
           "    return " + ("(max(sel) - min(sel)) / 2.0" if half else "max(sel) - min(sel)") + "\n")

    def args(n, empty=False):
        a = rng.randint(0, n // 2)
        b = rng.randint(a + 2, n)
        if empty:
            a, b = n + 1, n + 4
        x = trace(rng, n)
        if by_index:
            return (x, a, b)
        dt = pick(rng, (0.01, 0.02, 0.05))
        return (x, dt, _r((a - 0.5) * dt, 6), _r((b - 0.5) * dt, 6))
    return Variant(entry, spec, ref(src, entry), [args(8)], [args(rng.randint(20, 100)) for _ in range(5)]
                   + [args(30, empty=True)], f"peak_to_peak|{half}|{by_index}")


@kind("signal_to_noise", "data quality")
def _snr(rng):
    mode = pick(rng, ("rms", "peak"))
    scale = pick(rng, ("db", "ratio"))
    entry = pick(rng, ("snr_db", "signal_to_noise_db") if scale == "db" else ("snr", "signal_to_noise_ratio"))
    measure = "RMS amplitude (sqrt of the mean square)" if mode == "rms" else "peak absolute amplitude"
    spec = phrase(rng, f"{entry}(x, noise_end, sig_start, sig_end)",
                  "compute a signal-to-noise ratio" + (" in decibels" if scale == "db" else ""),
                  "The noise window is x[0:noise_end] and the signal window x[sig_start:sig_end] (Python slices, "
                  f"both non-empty, noise not all zero). With A the {measure} of each window, return " +
                  ("20·log10(A_signal / A_noise)." if scale == "db" else "A_signal / A_noise."))
    meas = "math.sqrt(sum(v * v for v in w) / len(w))" if mode == "rms" else "max(abs(v) for v in w)"
    ratio = "_a(x[sig_start:sig_end]) / _a(x[0:noise_end])"
    src = (f"import math\n\n\ndef _a(w):\n    return {meas}\n\n\ndef ENTRY(x, noise_end, sig_start, sig_end):\n"
           "    return " + (f"20.0 * math.log10({ratio})" if scale == "db" else ratio) + "\n")

    def args(n):
        ne = rng.randint(8, n // 3)
        x = burst(rng, n, rng.randint(ne + 2, n // 2), amp=rng.uniform(0.3, 3))
        s0 = rng.randint(ne, n // 2)
        return (x, ne, s0, rng.randint(s0 + 5, n))
    return Variant(entry, spec, ref(src, entry), [args(24)], [args(rng.randint(40, 110)) for _ in range(6)],
                   f"signal_to_noise|{mode}|{scale}")


@kind("event_rate", "earthquake statistics")
def _event_rate(rng):
    entry = pick(rng, ("event_rate", "binned_event_counts", "seismicity_rate"))
    per_day = rng.random() < 0.5
    spec = phrase(rng, f"{entry}(times, t0, width, nbins)", "bin an earthquake catalog in time",
                  "`times` are event times in days. Bin k (k = 0 … nbins−1) covers t0 + k·width ≤ t < "
                  "t0 + (k+1)·width; events outside all bins are ignored.",
                  "Return the list of event rates per day (count / width)." if per_day else
                  "Return the list of nbins counts.")
    src = '''
def ENTRY(times, t0, width, nbins):
    out = [0] * nbins
    for t in times:
        k = int((t - t0) // width)
        if 0 <= k < nbins and t0 + k * width <= t < t0 + (k + 1) * width:
            out[k] += 1
    return RESULT
'''.replace("RESULT", "[c / width for c in out]" if per_day else "out")

    def args(n):
        t0 = _r(rng.uniform(0, 100), 2)
        width = pick(rng, (1.0, 2.0, 5.0, 7.0))
        nb = rng.randint(3, 8)
        ts = sorted(_r(t0 + rng.uniform(-width, (nb + 1) * width), 3) for _ in range(n))
        ts = [t for t in ts if abs(((t - t0) / width) - round((t - t0) / width)) > 1e-6]
        return (ts, t0, width, nb)
    return Variant(entry, spec, ref(src, entry), [args(6)], [args(rng.randint(10, 80)) for _ in range(6)],
                   f"event_rate|{per_day}")


@kind("hv_spectral_ratio", "site characterization")
def _hv(rng):
    hdef = pick(rng, ("quadratic", "geometric"))
    ret = pick(rng, ("curve", "peak"))
    entry = pick(rng, ("hv_ratio", "hvsr") if ret == "curve" else ("hv_peak", "hvsr_peak"))
    htxt = "H = sqrt((N² + E²)/2)" if hdef == "quadratic" else "H = sqrt(N·E)"
    sig = f"{entry}(freqs, n, e, v)" if ret == "peak" else f"{entry}(n, e, v)"
    spec = phrase(rng, sig, "compute the horizontal-to-vertical spectral ratio of ambient noise",
                  "`n`, `e` and `v` are equal-length lists of positive Fourier amplitudes of the north, east and "
                  f"vertical components at the same frequencies. At each frequency {htxt} and the ratio is H/V.",
                  "Return the tuple (f0, A0): the frequency in `freqs` where the ratio is largest (the first if "
                  "several tie) and that ratio." if ret == "peak" else "Return the list of ratios.")
    hexpr = "math.sqrt((a * a + b * b) / 2.0)" if hdef == "quadratic" else "math.sqrt(a * b)"
    body = f"    r = [{hexpr} / c for a, b, c in zip(n, e, v)]\n"
    tail = ("    k = max(range(len(r)), key=lambda i: (r[i], -i))\n    return (freqs[k], r[k])\n" if ret == "peak"
            else "    return r\n")
    src = "import math\n\n\ndef ENTRY(" + ("freqs, n, e, v" if ret == "peak" else "n, e, v") + "):\n" + body + tail

    def args(k):
        fr = [_r(0.2 * 1.25 ** i, 4) for i in range(k)]
        f0 = pick(rng, fr[1:-1])
        pk = rng.uniform(2, 8)
        amp = [1 + (pk - 1) * math.exp(-(math.log(f / f0) / 0.3) ** 2) for f in fr]
        v = [_r(rng.uniform(0.5, 2), 4) for _ in fr]
        n = [_r(a * vv * rng.uniform(0.9, 1.1), 4) for a, vv in zip(amp, v)]
        e = [_r(a * vv * rng.uniform(0.9, 1.1), 4) for a, vv in zip(amp, v)]
        return (fr, n, e, v) if ret == "peak" else (n, e, v)
    return Variant(entry, spec, ref(src, entry), [args(5)], [args(rng.randint(12, 30)) for _ in range(6)],
                   f"hv_spectral_ratio|{hdef}|{ret}")


@kind("apparent_velocity", "array processing")
def _apparent_velocity(rng):
    ret = pick(rng, ("velocity", "slowness_intercept", "velocity_ms"))
    entry = pick(rng, ("apparent_velocity", "moveout_velocity") if ret != "slowness_intercept" else
                 ("slowness_fit", "fit_moveout"))
    spec = phrase(rng, f"{entry}(x, t)", "fit the moveout of an arrival across a linear array",
                  "`x` are station offsets along the line in km and `t` the arrival times in s (at least 2, not all "
                  "the same offset). Fit t = t0 + s·x by ordinary least squares.",
                  {"velocity": "Return the apparent velocity 1/s in km/s.",
                   "velocity_ms": "Return the apparent velocity 1/s in m/s.",
                   "slowness_intercept": "Return the tuple (s, t0): the slowness in s/km and the intercept in s."}[ret])
    tail = {"velocity": "1.0 / s", "velocity_ms": "1000.0 / s", "slowness_intercept": "(s, mt - s * mx)"}[ret]
    src = ("def ENTRY(x, t):\n    n = len(x)\n    mx, mt = sum(x) / n, sum(t) / n\n"
           "    s = sum((a - mx) * (b - mt) for a, b in zip(x, t)) / sum((a - mx) ** 2 for a in x)\n"
           f"    return {tail}\n")

    def args(k):
        v = rng.uniform(3, 12)
        t0 = rng.uniform(0, 20)
        xs = sorted(_r(rng.uniform(0, 50), 3) for _ in range(k))
        return (xs, [_r(t0 + xx / v + rng.gauss(0, 0.02), 4) for xx in xs])
    return Variant(entry, spec, ref(src, entry), [args(3)], [args(rng.randint(4, 15)) for _ in range(6)],
                   f"apparent_velocity|{ret}")


@kind("network_local_magnitude", "magnitude")
def _ml(rng):
    agg = pick(rng, ("mean", "median"))
    per_station = rng.random() < 0.3
    entry = pick(rng, ("station_ml",) if per_station else ("network_ml", "local_magnitude"))
    spec = phrase(rng, f"{entry}(amps_mm, dists_km)", "compute local magnitudes from Wood–Anderson amplitudes",
                  "For each station, ML = log10(A) + 1.11·log10(r) + 0.00189·r − 2.09 (Hutton & Boore, 1987), with A "
                  "the zero-to-peak Wood–Anderson amplitude in mm and r the hypocentral distance in km.",
                  "Return the list of station magnitudes." if per_station else
                  f"Return the {agg} of the station magnitudes"
                  + (" (for an even count, the mean of the two middle values)." if agg == "median" else "."))
    tail = {"list": "    return ml\n", "mean": "    return sum(ml) / len(ml)\n",
            "median": "    s = sorted(ml)\n    n = len(s)\n"
                      "    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])\n"}[
        "list" if per_station else agg]
    src = ("import math\n\n\ndef ENTRY(amps_mm, dists_km):\n"
           "    ml = [math.log10(a) + 1.11 * math.log10(r) + 0.00189 * r - 2.09 for a, r in zip(amps_mm, dists_km)]\n"
           + tail)

    def args(k):
        m = rng.uniform(1.5, 5.5)
        rs = [_r(rng.uniform(10, 400), 1) for _ in range(k)]
        amps = [float(f"{10 ** (m - 1.11 * math.log10(r) - 0.00189 * r + 2.09 + rng.gauss(0, 0.15)):.4g}")
                for r in rs]
        return (amps, rs)
    return Variant(entry, spec, ref(src, entry), [args(3)], [args(rng.randint(3, 12)) for _ in range(6)],
                   f"network_local_magnitude|{'list' if per_station else agg}")


@kind("travel_time_residuals", "earthquake location")
def _residuals(rng):
    ret = pick(rng, ("list", "rms"))
    names = ("travel_time_residuals", "arrival_residuals") if ret == "list" else ("residual_rms", "location_rms")
    entry = pick(rng, names)
    spec = phrase(rng, f"{entry}(stations, arrivals, x0, y0, t0, v)",
                  "compute P-arrival residuals for a trial epicentre",
                  "`stations` lists (x, y) positions in km and `arrivals` the observed P times in s; the trial "
                  "source is at (x0, y0) on the surface with origin time t0, and the velocity is `v` km/s. The "
                  "residual is observed − (t0 + distance/v) with 2-D distances.",
                  "Return the list of residuals." if ret == "list" else
                  "Return the RMS of the residuals (sqrt of their mean square).")
    tail = "    return r\n" if ret == "list" else "    return math.sqrt(sum(v_ * v_ for v_ in r) / len(r))\n"
    src = ("import math\n\n\ndef ENTRY(stations, arrivals, x0, y0, t0, v):\n"
           "    r = [a - (t0 + math.hypot(sx - x0, sy - y0) / v) for (sx, sy), a in zip(stations, arrivals)]\n" + tail)

    def args(k):
        st = [(_r(rng.uniform(-80, 80), 2), _r(rng.uniform(-80, 80), 2)) for _ in range(k)]
        ex, ey, t0, v = rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(0, 30), rng.uniform(5, 7)
        arr = [_r(t0 + math.hypot(sx - ex, sy - ey) / v + rng.gauss(0, 0.1), 3) for sx, sy in st]
        return (st, arr, _r(ex + rng.gauss(0, 3), 2), _r(ey + rng.gauss(0, 3), 2), _r(t0 + rng.gauss(0, 0.3), 2),
                _r(v, 2))
    return Variant(entry, spec, ref(src, entry), [args(3)], [args(rng.randint(4, 12)) for _ in range(6)],
                   f"travel_time_residuals|{ret}")


@kind("catalog_select", "earthquake statistics")
def _catalog_select(rng):
    radial = rng.random() < 0.5
    ret = pick(rng, ("indices", "count"))
    entry = pick(rng, ("select_events", "filter_catalog") if ret == "indices" else ("count_events",))
    sig = (f"{entry}(events, mmin, t1, t2, lat0, lon0, radius_km)" if radial else f"{entry}(events, mmin, t1, t2)")
    rad = (f" and lie within `radius_km` of (`lat0`, `lon0`) by the haversine great-circle distance with "
           f"R = {R_EARTH} km" if radial else "")
    spec = phrase(rng, sig, "select earthquakes from a catalog",
                  "`events` is a list of (time_days, lat_deg, lon_deg, magnitude). Keep the events with "
                  f"magnitude ≥ mmin − 1e-9, t1 ≤ time < t2{rad}.",
                  "Return the list of their indices in `events`, in order." if ret == "indices" else
                  "Return how many there are.")
    dist = ("    p1 = math.radians(lat0)\n"
            "    def d(lat, lon):\n        p2 = math.radians(lat)\n"
            "        a = (math.sin((p2 - p1) / 2) ** 2\n"
            "             + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon - lon0) / 2) ** 2)\n"
            f"        return {R_EARTH!r} * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))\n") if radial else ""
    cond = "m >= mmin - 1e-9 and t1 <= t < t2" + (" and d(la, lo) <= radius_km" if radial else "")
    src = ("import math\n\n\ndef ENTRY(events, mmin, t1, t2" + (", lat0, lon0, radius_km" if radial else "") + "):\n"
           + dist + f"    sel = [i for i, (t, la, lo, m) in enumerate(events) if {cond}]\n"
           + ("    return sel\n" if ret == "indices" else "    return len(sel)\n"))

    def args(k):
        la0, lo0 = latlon(rng)
        ev = [(_r(rng.uniform(0, 365), 3), _r(la0 + rng.uniform(-1.5, 1.5), 3),
               _r(lo0 + rng.uniform(-1.5, 1.5), 3), _r(rng.uniform(1, 6), 1)) for _ in range(k)]
        t1 = _r(rng.uniform(0, 100), 2)
        base = (ev, _r(rng.uniform(1.5, 3.0), 1), t1, _r(t1 + rng.uniform(120, 260), 2))
        return base + ((la0, lo0, pick(rng, (100.0, 150.0, 250.0))) if radial else ())
    return Variant(entry, spec, ref(src, entry), [args(4)], [args(rng.randint(10, 40)) for _ in range(6)],
                   f"catalog_select|{radial}|{ret}")


@kind("first_motion", "source mechanism")
def _first_motion(rng):
    ref_mode = pick(rng, ("absolute", "pre_mean"))
    entry = pick(rng, ("first_motion_polarity", "onset_polarity"))
    base = ("x[i]" if ref_mode == "absolute" else "x[i] − μ, where μ is the mean of x[0:onset]")
    spec = phrase(rng, f"{entry}(x, onset, level)", "determine the first-motion polarity of a P arrival",
                  "Starting at index `onset` (≥ 1), find the first sample i ≥ onset with |d| > level, where "
                  f"d = {base}. "
                  "Return +1 if that d is positive, −1 if it is negative, and 0 if no sample qualifies.")
    mu = "    mu = sum(x[:onset]) / onset\n" if ref_mode == "pre_mean" else "    mu = 0.0\n"
    src = ("def ENTRY(x, onset, level):\n" + mu + "    for v in x[onset:]:\n        d = v - mu\n"
           "        if abs(d) > level:\n            return 1 if d > 0 else -1\n    return 0\n")

    def args(n, quiet=False):
        onset = rng.randint(5, n // 2)
        x = burst(rng, n, onset, noise_s=0.02, amp=rng.uniform(0.3, 2) * pick(rng, (1, -1)))
        if ref_mode == "pre_mean":
            off = rng.uniform(-1, 1)
            x = [_r(v + off) for v in x]
        lvl = 10.0 if quiet else _r(rng.uniform(0.08, 0.2), 3)
        return (x, onset, lvl)
    return Variant(entry, spec, ref(src, entry), [args(16)], [args(rng.randint(30, 80)) for _ in range(5)]
                   + [args(40, quiet=True)], f"first_motion|{ref_mode}")


@kind("radiated_energy", "earthquake source")
def _energy(rng):
    ret = pick(rng, ("joules", "magnitude"))
    entry = pick(rng, ("total_radiated_energy", "summed_energy") if ret == "joules" else ("energy_equivalent_ms",))
    spec = phrase(rng, f"{entry}(ms_list)", "sum the radiated seismic energy of a sequence of earthquakes",
                  "Each surface-wave magnitude Ms in `ms_list` radiates E = 10^(1.5·Ms + 4.8) J "
                  "(Gutenberg–Richter energy relation).",
                  "Return the total energy in J." if ret == "joules" else
                  "Return the magnitude of a single event with the total energy, (log10 ΣE − 4.8)/1.5.")
    src = ("import math\n\n\ndef ENTRY(ms_list):\n    e = sum(10 ** (1.5 * m + 4.8) for m in ms_list)\n"
           + ("    return e\n" if ret == "joules" else "    return (math.log10(e) - 4.8) / 1.5\n"))

    def args(k):
        return ([_r(rng.uniform(3.0, 8.0), 1) for _ in range(k)],)
    return Variant(entry, spec, ref(src, entry), [args(3)], [args(rng.randint(2, 25)) for _ in range(6)],
                   f"radiated_energy|{ret}")


@kind("normalize_trace", "trace preprocessing")
def _normalize(rng):
    mode = pick(rng, ("peak", "rms", "energy"))
    entry = pick(rng, {"peak": ("normalize_peak", "normalize"), "rms": ("normalize_rms",),
                       "energy": ("normalize_energy", "unit_energy")}[mode])
    norm = {"peak": "the largest absolute value", "rms": "the RMS amplitude sqrt(mean(x²))",
            "energy": "sqrt(Σx²), so that the squares of the result sum to 1"}[mode]
    spec = phrase(rng, f"{entry}(x)", "normalize a trace",
                  f"Divide every sample of the non-empty list `x` by {norm}. If that value is 0, return a copy of "
                  "`x` unchanged.")
    nexpr = {"peak": "max(abs(v) for v in x)", "rms": "math.sqrt(sum(v * v for v in x) / len(x))",
             "energy": "math.sqrt(sum(v * v for v in x))"}[mode]
    src = (f"import math\n\n\ndef ENTRY(x):\n    s = {nexpr}\n    if s == 0:\n        return list(x)\n"
           "    return [v / s for v in x]\n")

    def args(n, zero=False):
        return ([0.0] * n,) if zero else (trace(rng, n, amp=rng.uniform(0.1, 50)),)
    return Variant(entry, spec, ref(src, entry), [args(5)], [args(rng.randint(10, 80)) for _ in range(5)]
                   + [args(6, zero=True)], f"normalize_trace|{mode}")


@kind("decimate", "trace preprocessing")
def _decimate(rng):
    smooth = rng.random() < 0.5
    entry = pick(rng, ("decimate", "downsample") if not smooth else ("smooth_decimate", "decimate_with_average"))
    rule = ("output sample k is x[k·q] for k·q < len(x)" if not smooth else
            "output sample k is the mean of x[k·q : k·q + q] (fewer samples in a short last block), for "
            "k·q < len(x)")
    spec = phrase(rng, f"{entry}(x, q)", "reduce the sampling rate of a trace by an integer factor `q`",
                  f"With q ≥ 1, {rule}. Return the list.")
    src = ("def ENTRY(x, q):\n    return [x[i] for i in range(0, len(x), q)]\n" if not smooth else
           "def ENTRY(x, q):\n    out = []\n    for i in range(0, len(x), q):\n        seg = x[i:i + q]\n"
           "        out.append(sum(seg) / len(seg))\n    return out\n")

    def args(n):
        return (trace(rng, n), rng.randint(2, 5))
    return Variant(entry, spec, ref(src, entry), [args(7)], [args(rng.randint(15, 90)) for _ in range(6)],
                   f"decimate|{smooth}")


@kind("zero_pad", "spectral analysis")
def _zero_pad(rng):
    mode = pick(rng, ("pow2", "length"))
    entry = pick(rng, ("pad_to_pow2", "next_pow2_pad") if mode == "pow2" else ("zero_pad", "pad_trace"))
    sig = f"{entry}(x)" if mode == "pow2" else f"{entry}(x, n)"
    spec = phrase(rng, sig, "zero-pad a trace for an FFT",
                  "Append zeros to the end of `x` until its length is the smallest power of two that is ≥ len(x) "
                  "(an empty list stays empty)." if mode == "pow2" else
                  "Append zeros to the end of `x` up to length `n`; if len(x) ≥ n, return the first n samples.",
                  "Return a new list.")
    src = ("def ENTRY(x):\n    if not x:\n        return []\n    m = 1\n    while m < len(x):\n        m *= 2\n"
           "    return list(x) + [0.0] * (m - len(x))\n" if mode == "pow2" else
           "def ENTRY(x, n):\n    return list(x[:n]) + [0.0] * max(0, n - len(x))\n")

    def args(n):
        x = trace(rng, n)
        return (x,) if mode == "pow2" else (x, rng.randint(max(1, n - 5), n + 20))
    hid = [args(rng.randint(3, 70)) for _ in range(5)] + [args(16)]
    return Variant(entry, spec, ref(src, entry), [args(5)], hid, f"zero_pad|{mode}")


@kind("band_power", "spectral analysis")
def _band_power(rng):
    frac = rng.random() < 0.5
    entry = pick(rng, ("band_power", "spectral_band_energy") if not frac else ("band_power_fraction",))
    spec = phrase(rng, f"{entry}(x, dt, fmin, fmax)", "measure the power of a trace in a frequency band",
                  "With n = len(x), X[k] = Σ_j x[j]·exp(−2πi·j·k/n) and f_k = k/(n·dt) for k = 0 … floor(n/2). "
                  "The band power is Σ|X[k]|² over the k with fmin ≤ f_k ≤ fmax.",
                  "Return the band power divided by the same sum over all k = 0 … floor(n/2) (0.0 if that sum is "
                  "0)." if frac else "Return the band power.")
    src = ("import cmath\nimport math\n\n\ndef ENTRY(x, dt, fmin, fmax):\n    n = len(x)\n    band = total = 0.0\n"
           "    for k in range(n // 2 + 1):\n"
           "        p = abs(sum(x[j] * cmath.exp(-2j * math.pi * j * k / n) for j in range(n))) ** 2\n"
           "        total += p\n        if fmin <= k / (n * dt) <= fmax:\n            band += p\n"
           + ("    return band / total if total > 0 else 0.0\n" if frac else "    return band\n"))

    def args(n):
        dt = pick(rng, (0.01, 0.02, 0.05))
        df = 1.0 / (n * dt)
        k1 = rng.randint(1, n // 4)
        k2 = rng.randint(k1 + 1, n // 2)
        return (trace(rng, n), dt, _r((k1 - 0.5) * df, 6), _r((k2 + 0.5) * df, 6))
    return Variant(entry, spec, ref(src, entry), [args(8)], [args(rng.randint(16, 64)) for _ in range(6)],
                   f"band_power|{frac}", abs_scale=1e-6)


@kind("interevent_times", "earthquake statistics")
def _interevent(rng):
    ret = pick(rng, ("list", "cov"))
    names = ("interevent_times", "waiting_times") if ret == "list" else ("interevent_cov", "burstiness_cov")
    entry = pick(rng, names)
    spec = phrase(rng, f"{entry}(times)", "analyse the waiting times between earthquakes",
                  "`times` is a list of at least 2 (3 for the coefficient) event times, not necessarily sorted. Sort "
                  "them and take the differences of consecutive times.",
                  "Return the list of differences." if ret == "list" else
                  "Return their coefficient of variation: the population standard deviation (divide by the "
                  "number of differences) divided by the mean.")
    src = ("import math\n\n\ndef ENTRY(times):\n    t = sorted(times)\n    d = [b - a for a, b in zip(t, t[1:])]\n"
           + ("    return d\n" if ret == "list" else
              "    m = sum(d) / len(d)\n    return math.sqrt(sum((v - m) ** 2 for v in d) / len(d)) / m\n"))

    def args(k):
        ts = [_r(rng.uniform(0, 1000), 3) for _ in range(k)]
        return (ts,)
    return Variant(entry, spec, ref(src, entry), [args(4)], [args(rng.randint(5, 40)) for _ in range(6)],
                   f"interevent_times|{ret}")


@kind("cumulative_moment", "earthquake source")
def _cum_moment(rng):
    ret = pick(rng, ("moment", "magnitude"))
    entry = pick(rng, ("cumulative_moment", "moment_release_curve") if ret == "moment" else ("cumulative_mw",))
    spec = phrase(rng, f"{entry}(times, mws)", "compute the cumulative seismic moment release of a sequence",
                  "`times` and `mws` are equal-length lists of event times and moment magnitudes (times distinct, "
                  "not necessarily sorted). Each event has M0 = 10^(1.5·Mw + 9.1) N·m. In time order, return the "
                  "running sum of M0 after each event" + (" in N·m." if ret == "moment" else
                                                          ", each converted back to Mw = (2/3)·(log10 ΣM0 − 9.1)."))
    src = ("import math\n\n\ndef ENTRY(times, mws):\n    s, out = 0.0, []\n"
           "    for t, m in sorted(zip(times, mws)):\n        s += 10 ** (1.5 * m + 9.1)\n"
           + ("        out.append(s)\n" if ret == "moment" else
              "        out.append((2.0 / 3.0) * (math.log10(s) - 9.1))\n") + "    return out\n")

    def args(k):
        ts = rng.sample(range(10000), k)
        return ([t / 10 for t in ts], [_r(rng.uniform(2, 7), 1) for _ in range(k)])
    return Variant(entry, spec, ref(src, entry), [args(3)], [args(rng.randint(3, 25)) for _ in range(6)],
                   f"cumulative_moment|{ret}")


# =============================================================================================
# Held-out section: not published. The 11 held-out T2 kinds behind D-Code live in
# t2_tasks_heldout.py, which stays out of git so that the test problems cannot be regenerated or
# trained on (README, "Data & licensing"). It is imported at the end of this module when present.
# =============================================================================================


def kinds_by_split(split: str) -> list[Kind]:
    return [k for k in KINDS.values() if k.split == split]


# The held-out D-Code kinds are not published; they register themselves on import when their private module
# is present (dpt.data.gen.common.import_private_module_if_present). Imported last: they use the helpers above.
from dpt.data.gen.common import import_private_module_if_present  # noqa: E402

heldout_code_kinds_present = import_private_module_if_present("dpt.data.gen.t2_tasks_heldout")
