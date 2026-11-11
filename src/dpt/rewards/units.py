"""A small, self-contained unit registry for grading numeric answers.

Why not ``pint``: graders run inside every Ray reward worker and the sandbox
cluster, so they should have no heavy dependencies. More importantly, the
grader's behaviour must be fully specified by code we review. A third-party
registry that silently accepts ``"m"`` as minutes in some context, or that
changes alias tables between versions, would move the reward.

Model
-----
A :class:`Unit` is ``value_SI = (value + offset) * factor`` with a dimension
vector over the SI base quantities (L, M, T, Θ, I, N). Angles are
dimensionless (rad = 1, deg = π/180), so "rad/s" and "Hz" are dimensionally
equal, which is the SI convention.

Temperature is the one non-multiplicative case. ``degC``/``degF`` carry an
offset **only when they appear alone** (absolute temperature: 25 °C = 298.15 K).
Inside a compound unit (``°C/km`` geothermal gradient) they mean a temperature
*difference*, and the offset is dropped. That is the only physically
meaningful reading.

Parsing grammar (deliberately small):

    unit    := term (('/' term) | ([ *·⋅] term))*
    term    := '(' unit ')' exp? | symbol exp?
    exp     := '^' int | '^{' int '}' | '**' int | superscript digits (²³⁻¹…)
    symbol  := exact registry entry | prefix + prefixable base

``a/b/c`` means ``a/(b·c)``. Exact symbols win over prefix splits, so ``Pa``
is pascal and not peta-annum, ``min`` is minutes, and ``h`` is hours.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Dimension vector order: length, mass, time, temperature, current, amount
_DIMS = ("L", "M", "T", "Θ", "I", "N")
Dim = tuple[float, float, float, float, float, float]
_DLESS: Dim = (0, 0, 0, 0, 0, 0)


class UnitError(ValueError):
    """Unit string could not be parsed."""


class DimensionError(ValueError):
    """Units are not dimensionally compatible."""


@dataclass(frozen=True)
class Unit:
    factor: float  # multiply by this to get SI
    dim: Dim
    offset: float = 0.0  # only non-zero for standalone degC / degF
    symbol: str = ""

    def __mul__(self, other: "Unit") -> "Unit":
        return Unit(
            self.factor * other.factor, _add(self.dim, other.dim), 0.0, f"{self.symbol}*{other.symbol}"
        )

    def __truediv__(self, other: "Unit") -> "Unit":
        return Unit(
            self.factor / other.factor, _sub(self.dim, other.dim), 0.0, f"{self.symbol}/{other.symbol}"
        )

    def __pow__(self, p: float) -> "Unit":
        dim = tuple(d * p for d in self.dim)
        return Unit(self.factor**p, dim, 0.0, f"{self.symbol}^{p:g}")  # type: ignore[arg-type]

    @property
    def is_dimensionless(self) -> bool:
        return all(abs(d) < 1e-12 for d in self.dim)


def _add(a: Dim, b: Dim) -> Dim:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def _sub(a: Dim, b: Dim) -> Dim:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def _d(L=0, M=0, T=0, K=0, I=0, N=0) -> Dim:  # noqa: E741 - physics letters
    return (L, M, T, K, I, N)


# ---------------------------------------------------------------------------------------------
# Registry: symbol -> (factor to SI, dimension, prefixable)
# ---------------------------------------------------------------------------------------------
_G = 9.80665
_BASE: dict[str, tuple[float, Dim, bool]] = {
    # dimensionless
    "1": (1.0, _DLESS, False),
    # "7.97 \\text{(dimensionless)}": the label says there is no unit (models write it; the value still counts)
    "unitless": (1.0, _DLESS, False),
    "dimensionless": (1.0, _DLESS, False),
    # counts and magnitudes named after the number ("800 aftershocks", "6.7 magnitude"); still dimensionless,
    # so "800 aftershocks/day" against a count label is a dimension mismatch, as it should be
    "magnitude": (1.0, _DLESS, False),
    "aftershock": (1.0, _DLESS, False),
    "aftershocks": (1.0, _DLESS, False),
    "event": (1.0, _DLESS, False),
    "events": (1.0, _DLESS, False),
    "earthquake": (1.0, _DLESS, False),
    "earthquakes": (1.0, _DLESS, False),
    "%": (1e-2, _DLESS, False),
    "percent": (1e-2, _DLESS, False),
    "‰": (1e-3, _DLESS, False),
    "permil": (1e-3, _DLESS, False),
    "ppm": (1e-6, _DLESS, False),
    "ppb": (1e-9, _DLESS, False),
    "rad": (1.0, _DLESS, True),
    "sr": (1.0, _DLESS, False),
    "deg": (math.pi / 180.0, _DLESS, False),
    "°": (math.pi / 180.0, _DLESS, False),
    "arcmin": (math.pi / 180.0 / 60.0, _DLESS, False),
    "arcsec": (math.pi / 180.0 / 3600.0, _DLESS, False),
    # length
    "m": (1.0, _d(L=1), True),
    "Å": (1e-10, _d(L=1), False),
    "angstrom": (1e-10, _d(L=1), False),
    "ft": (0.3048, _d(L=1), False),
    "mi": (1609.344, _d(L=1), False),
    "nmi": (1852.0, _d(L=1), False),
    "au": (1.495978707e11, _d(L=1), False),
    # mass
    "g": (1e-3, _d(M=1), True),
    "t": (1e3, _d(M=1), False),  # tonne
    "lb": (0.45359237, _d(M=1), False),
    # time
    "s": (1.0, _d(T=1), True),
    "sec": (1.0, _d(T=1), False),
    "min": (60.0, _d(T=1), False),
    "h": (3600.0, _d(T=1), False),
    "hr": (3600.0, _d(T=1), False),
    "day": (86400.0, _d(T=1), False),
    "d": (86400.0, _d(T=1), False),
    "yr": (3.15576e7, _d(T=1), True),  # Julian year
    "a": (3.15576e7, _d(T=1), True),  # annum: ka, Ma, Ga are standard in geoscience
    # temperature (absolute scale; offsets handled specially)
    "K": (1.0, _d(K=1), True),
    "degC": (1.0, _d(K=1), False),
    "°C": (1.0, _d(K=1), False),
    "℃": (1.0, _d(K=1), False),
    "degF": (5.0 / 9.0, _d(K=1), False),
    "°F": (5.0 / 9.0, _d(K=1), False),
    # current / amount
    "A": (1.0, _d(I=1), True),
    "mol": (1.0, _d(N=1), True),
    # derived
    "Hz": (1.0, _d(T=-1), True),
    "N": (1.0, _d(L=1, M=1, T=-2), True),
    "dyn": (1e-5, _d(L=1, M=1, T=-2), False),
    "Pa": (1.0, _d(L=-1, M=1, T=-2), True),
    "bar": (1e5, _d(L=-1, M=1, T=-2), True),
    "atm": (101325.0, _d(L=-1, M=1, T=-2), False),
    "psi": (6894.757293168, _d(L=-1, M=1, T=-2), False),
    "J": (1.0, _d(L=2, M=1, T=-2), True),
    "erg": (1e-7, _d(L=2, M=1, T=-2), False),
    "eV": (1.602176634e-19, _d(L=2, M=1, T=-2), True),
    "cal": (4.184, _d(L=2, M=1, T=-2), True),
    "W": (1.0, _d(L=2, M=1, T=-3), True),
    "C": (1.0, _d(T=1, I=1), True),
    "V": (1.0, _d(L=2, M=1, T=-3, I=-1), True),
    "Ω": (1.0, _d(L=2, M=1, T=-3, I=-2), True),
    "ohm": (1.0, _d(L=2, M=1, T=-3, I=-2), True),
    "S": (1.0, _d(L=-2, M=-1, T=3, I=2), True),
    "T": (1.0, _d(M=1, T=-2, I=-1), True),  # tesla (nT is the geomagnetic workhorse)
    "Gal": (1e-2, _d(L=1, T=-2), True),  # gravity: 1 Gal = 1 cm/s², mGal common
    "gn": (_G, _d(L=1, T=-2), False),  # standard gravity as a unit (not "g", which is gram)
    "L": (1e-3, _d(L=3), True),
    "l": (1e-3, _d(L=3), False),
}

_PREFIXES: dict[str, float] = {
    "Y": 1e24,
    "Z": 1e21,
    "E": 1e18,
    "P": 1e15,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "h": 1e2,
    "da": 1e1,
    "d": 1e-1,
    "c": 1e-2,
    "m": 1e-3,
    "µ": 1e-6,
    "μ": 1e-6,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}

# Spelled-out aliases → canonical symbols.
_ALIASES: dict[str, str] = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "centimeter": "cm",
    "centimeters": "cm",
    "millimeter": "mm",
    "millimeters": "mm",
    "second": "s",
    "seconds": "s",
    "secs": "s",
    "minute": "min",
    "minutes": "min",
    "mins": "min",
    "hour": "h",
    "hours": "h",
    "hrs": "h",
    "days": "day",
    "year": "yr",
    "years": "yr",
    "yrs": "yr",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "tonne": "t",
    "tonnes": "t",
    "kelvin": "K",
    "celsius": "degC",
    "fahrenheit": "degF",
    "degree": "deg",
    "degrees": "deg",
    "radian": "rad",
    "radians": "rad",
    "hertz": "Hz",
    "newton": "N",
    "newtons": "N",
    "pascal": "Pa",
    "pascals": "Pa",
    "joule": "J",
    "joules": "J",
    "watt": "W",
    "watts": "W",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",
    "ohms": "ohm",
    "tesla": "T",
    "米": "m",
    "千米": "km",
    "公里": "km",
    "厘米": "cm",
    "毫米": "mm",
    "秒": "s",
    "分钟": "min",
    "小时": "h",
    "天": "day",
    "年": "yr",
    "克": "g",
    "千克": "kg",
    "公斤": "kg",
    "吨": "t",
    "度": "deg",
    "赫兹": "Hz",
    "帕": "Pa",
    "兆帕": "MPa",
    "焦耳": "J",
    "瓦": "W",
}

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUP_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")


def _normalize_unit_string(s: str) -> str:
    s = s.strip()
    # Degree signs before generic \circ handling (order matters: "^\circ" must not become "^°").
    s = s.replace("^{\\circ}", "°").replace("^\\circ", "°").replace("\\circ", "°").replace("\\degree", "°")
    s = _SUP_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPTS), s)
    # Exponent braces next (m^{-1} → m^-1) so LaTeX wrappers below contain no nested braces.
    s = re.sub(r"\^\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", r"^\1", s)
    # LaTeX wrappers: \mathrm{km}, \text{ s}; repeat for nested wrappers.
    for _ in range(3):
        s = re.sub(r"\\(?:mathrm|text|textrm|rm|mbox|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\cdot", "·").replace("\\times", "·").replace("\\,", " ").replace("\\;", " ")
    s = s.replace("\\!", "").replace("~", " ").replace("\\ ", " ").replace("\\%", "%")
    s = s.replace("\\mu", "µ").replace("\\Omega", "Ω").replace("\\AA", "Å")
    # Any remaining braces act as parentheses.
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"°\s*([CF])\b", r"deg\1", s)
    s = s.replace("℃", "degC")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _lookup_symbol(sym: str) -> Unit:
    sym = _ALIASES.get(sym, _ALIASES.get(sym.lower(), sym))
    if sym in _BASE:
        f, dim, _ = _BASE[sym]
        return Unit(f, dim, 0.0, sym)
    # Longest prefix first ("da" before "d").
    for p in sorted(_PREFIXES, key=len, reverse=True):
        if sym.startswith(p) and len(sym) > len(p):
            base = sym[len(p) :]
            if base in _BASE and _BASE[base][2]:
                f, dim, _ = _BASE[base]
                return Unit(f * _PREFIXES[p], dim, 0.0, sym)
    raise UnitError(f"unknown unit symbol {sym!r}")


def _lookup_with_trailing_exponent(sym: str) -> Unit:
    """Accept sloppy exponents written without a caret (``cm3``, ``s2``) when unambiguous."""
    try:
        return _lookup_symbol(sym)
    except UnitError:
        m = re.fullmatch(r"(.*?[^\d\-])(-?\d)", sym)
        if m:
            return _lookup_symbol(m.group(1)) ** float(m.group(2))
        raise


_TOKEN_RE = re.compile(r"\s*(\(|\)|/|\*\*|\^|\*|·|⋅|[-+]?\d+(?:\.\d+)?|[^\s()/*^·⋅]+)")


class _Parser:
    def __init__(self, text: str):
        self.toks = [t for t in _TOKEN_RE.findall(text) if t.strip()]
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse(self) -> Unit:
        u = self.unit()
        if self.peek() is not None:
            raise UnitError(f"unexpected token {self.peek()!r}")
        return u

    def unit(self) -> Unit:
        u = self.term()
        while True:
            t = self.peek()
            if t is None or t == ")":
                return u
            if t == "/":
                self.take()
                u = u / self.term()
            elif t in ("*", "·", "⋅"):
                self.take()
                u = u * self.term()
            else:  # implicit multiplication by juxtaposition: "m s^-1"
                u = u * self.term()

    def term(self) -> Unit:
        t = self.peek()
        if t is None:
            raise UnitError("unexpected end of unit")
        if t == "(":
            self.take()
            u = self.unit()
            if self.peek() != ")":
                raise UnitError("unbalanced parenthesis")
            self.take()
        else:
            self.take()
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t):
                if t == "1":
                    u = Unit(1.0, _DLESS, 0.0, "1")
                else:
                    raise UnitError(f"bare number {t!r} inside unit")
            else:
                u = _lookup_with_trailing_exponent(t)
        if self.peek() in ("^", "**"):
            self.take()
            e = self.take() if self.peek() is not None else ""
            if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", e):
                raise UnitError(f"bad exponent {e!r}")
            u = u ** float(e)
        return u


def parse_unit(text: str) -> Unit:
    """Parse a unit string into a :class:`Unit`. Empty string → dimensionless 1."""
    s = _normalize_unit_string(text)
    if s in ("", "1"):
        return Unit(1.0, _DLESS, 0.0, "")
    # Standalone temperature scales keep their offset (absolute temperatures).
    if s in ("degC", "°C"):
        return Unit(1.0, _d(K=1), 273.15, "degC")
    if s in ("degF", "°F"):
        return Unit(5.0 / 9.0, _d(K=1), 459.67, "degF")
    return _Parser(s).parse()


def compatible(a: Unit, b: Unit) -> bool:
    return all(abs(x - y) < 1e-9 for x, y in zip(a.dim, b.dim))


def to_si(value: float, unit: Unit) -> float:
    return (value + unit.offset) * unit.factor


def from_si(value_si: float, unit: Unit) -> float:
    return value_si / unit.factor - unit.offset


def convert(value: float, from_unit: str | Unit, to_unit: str | Unit) -> float:
    """Convert ``value`` between units; raises :class:`DimensionError` if incompatible."""
    fu = parse_unit(from_unit) if isinstance(from_unit, str) else from_unit
    tu = parse_unit(to_unit) if isinstance(to_unit, str) else to_unit
    if not compatible(fu, tu):
        raise DimensionError(f"cannot convert {fu.symbol or '1'} {fu.dim} to {tu.symbol or '1'} {tu.dim}")
    return from_si(to_si(value, fu), tu)


def dimension_str(u: Unit) -> str:
    parts = [f"{n}^{d:g}" if d != 1 else n for n, d in zip(_DIMS, u.dim) if abs(d) > 1e-12]
    return "·".join(parts) or "1"
