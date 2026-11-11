"""Small, self-contained physical-unit parser for the evaluation numeric grader.

This is the eval-side unit table. It is intentionally not shared with the
training grader. A unit-conversion bug in one implementation then shows up as
a train/eval disagreement instead of silently inflating both.

Model
-----
A unit expression parses to ``(factor, dim)``. ``factor`` converts one unit
to SI. ``dim`` is a 7-tuple of integer exponents over
``(length, mass, time, temperature, current, amount, luminosity)``.
Angles, percent and ppm are dimensionless with a scale factor.

Supported syntax
----------------
``km/s``, ``m s^-1``, ``m·s⁻¹``, ``kg/(m s^2)``, ``g/cm³``, ``\\mathrm{km}\\,\\mathrm{s}^{-1}``,
``N*m``, ``dyn·cm``, ``mGal``, ``Myr``, ``°``, ``%``.
Temperatures with an offset (``°C``, ``°F``) are supported only as standalone
units, because offsets do not compose multiplicatively.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

Dim = tuple[int, int, int, int, int, int, int]
_ZERO: Dim = (0, 0, 0, 0, 0, 0, 0)


def _d(L=0, M=0, T=0, K=0, I=0, N=0, J=0) -> Dim:  # noqa: E741 - physics symbols
    return (L, M, T, K, I, N, J)


@dataclass(frozen=True)
class UnitValue:
    factor: float
    dim: Dim

    def __mul__(self, other: "UnitValue") -> "UnitValue":
        dim = tuple(a + b for a, b in zip(self.dim, other.dim))
        return UnitValue(self.factor * other.factor, dim)  # type: ignore[arg-type]

    def __pow__(self, p: int) -> "UnitValue":
        return UnitValue(self.factor**p, tuple(a * p for a in self.dim))  # type: ignore[arg-type]

    @property
    def dimensionless(self) -> bool:
        return self.dim == _ZERO


DIMENSIONLESS = UnitValue(1.0, _ZERO)

# symbol -> (factor to SI, dimension, prefixable)
_BASE: dict[str, tuple[float, Dim, bool]] = {
    "m": (1.0, _d(L=1), True),
    "g": (1e-3, _d(M=1), True),
    "s": (1.0, _d(T=1), True),
    "sec": (1.0, _d(T=1), False),
    "K": (1.0, _d(K=1), False),
    "A": (1.0, _d(I=1), True),
    "mol": (1.0, _d(N=1), True),
    "cd": (1.0, _d(J=1), False),
    "Hz": (1.0, _d(T=-1), True),
    "N": (1.0, _d(L=1, M=1, T=-2), True),
    "Pa": (1.0, _d(L=-1, M=1, T=-2), True),
    "J": (1.0, _d(L=2, M=1, T=-2), True),
    "W": (1.0, _d(L=2, M=1, T=-3), True),
    "V": (1.0, _d(L=2, M=1, T=-3, I=-1), True),
    "Ω": (1.0, _d(L=2, M=1, T=-3, I=-2), True),
    "ohm": (1.0, _d(L=2, M=1, T=-3, I=-2), True),
    "S": (1.0, _d(L=-2, M=-1, T=3, I=2), True),  # siemens (conductivity in S/m)
    "Tesla": (1.0, _d(M=1, T=-2, I=-1), False),
    "T": (1.0, _d(M=1, T=-2, I=-1), True),  # tesla; nT common in geomagnetism
    "bar": (1e5, _d(L=-1, M=1, T=-2), True),
    "atm": (101325.0, _d(L=-1, M=1, T=-2), False),
    "L": (1e-3, _d(L=3), True),
    "l": (1e-3, _d(L=3), False),
    "min": (60.0, _d(T=1), False),
    "h": (3600.0, _d(T=1), False),
    "hr": (3600.0, _d(T=1), False),
    "day": (86400.0, _d(T=1), False),
    "d": (86400.0, _d(T=1), False),
    "yr": (3.15576e7, _d(T=1), True),  # Julian year
    "a": (3.15576e7, _d(T=1), True),  # annum: ka, Ma, Ga
    "eV": (1.602176634e-19, _d(L=2, M=1, T=-2), True),
    "cal": (4.184, _d(L=2, M=1, T=-2), True),
    "erg": (1e-7, _d(L=2, M=1, T=-2), False),
    "dyn": (1e-5, _d(L=1, M=1, T=-2), False),
    "Gal": (1e-2, _d(L=1, T=-2), True),  # galileo; mGal in gravimetry
    "tonne": (1e3, _d(M=1), False),
    "in": (0.0254, _d(L=1), False),
    "ft": (0.3048, _d(L=1), False),
    "mi": (1609.344, _d(L=1), False),
    "rad": (1.0, _ZERO, False),
    "sr": (1.0, _ZERO, False),
    "deg": (math.pi / 180.0, _ZERO, False),
    "°": (math.pi / 180.0, _ZERO, False),
    "%": (1e-2, _ZERO, False),
    "‰": (1e-3, _ZERO, False),
    "ppm": (1e-6, _ZERO, False),
    "ppb": (1e-9, _ZERO, False),
}

_PREFIX: dict[str, float] = {
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

# Offset temperature units: value_K = (value + offset) * scale
_OFFSET_TEMPS: dict[str, tuple[float, float]] = {
    "°C": (273.15, 1.0),
    "degC": (273.15, 1.0),
    "℃": (273.15, 1.0),
    "°F": (459.67, 5.0 / 9.0),
    "degF": (459.67, 5.0 / 9.0),
    "℉": (459.67, 5.0 / 9.0),
}

_SUPERSCRIPTS = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
    }
)


class UnitParseError(ValueError):
    pass


def normalize_unit_text(text: str) -> str:
    """Turn LaTeX / unicode unit spellings into the plain syntax the parser accepts."""
    t = text.strip()
    t = re.sub(r"\\(?:mathrm|text|mathit|operatorname|rm)\s*\{([^{}]*)\}", r"\1", t)
    t = t.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ").replace("~", " ")
    t = t.replace("\\cdot", "·").replace("\\times", "·").replace("\\mu", "µ").replace("\\Omega", "Ω")
    t = t.replace("\\%", "%").replace("^\\circ", "°").replace("\\circ", "°").replace("\\degree", "°")
    t = t.replace("$", "")
    t = re.sub(r"\b([Mm])_\{?([wLsbe])\}?(?![A-Za-z])", r"\1\2", t)   # magnitude labels: M_s → Ms, m_b → mb
    t = re.sub(r"\^\s*\{\s*([-+]?\d+)\s*\}", r"^\1", t)
    t = t.replace("**", "^")
    # Superscript runs → ^n  (m s⁻¹ → m s^-1)
    t = re.sub(r"([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda m: "^" + m.group(1).translate(_SUPERSCRIPTS), t)
    t = re.sub(r"\bper\b", "/", t)
    t = t.replace("⋅", "·").replace("*", "·")
    return re.sub(r"\s+", " ", t).strip()


# Spelled-out units the policy may write ("20 seconds", "3 kilometres", "30 degrees").
_WORD_ALIASES = {
    "second": "s", "seconds": "s", "secs": "s", "minute": "min", "minutes": "min", "mins": "min",
    "hour": "h", "hours": "h", "hrs": "h", "days": "day", "year": "yr", "years": "yr", "yrs": "yr",
    "meter": "m", "meters": "m", "metre": "m", "metres": "m", "kilometer": "km", "kilometers": "km",
    "kilometre": "km", "kilometres": "km", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm",
    "centimetres": "cm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "gram": "g", "grams": "g", "kilogram": "kg", "kilograms": "kg", "degree": "deg", "degrees": "deg",
    "radian": "rad", "radians": "rad", "hertz": "Hz", "newton": "N", "newtons": "N", "pascal": "Pa",
    "pascals": "Pa", "joule": "J", "joules": "J", "watt": "W", "watts": "W", "percent": "%",
}
# Earthquake magnitude scales are dimensionless labels ("6.5 Mw"); same list as the training grader
# (rewards.t1_num.dimensionless_labels) so train and eval agree on the data contract.
_MAGNITUDE_LABELS = {"Mw", "ML", "Ms", "mb", "Mb", "Me", "Md", "M"}


# "7.97 \\text{(dimensionless)}": a note that there is no unit. The value is graded; the word means 1.
# Counts and magnitudes named after the number ("800 aftershocks", "6.7 magnitude") are dimensionless too.
_NO_UNIT_WORDS = {"unitless", "dimensionless", "magnitude", "aftershock", "aftershocks", "event", "events",
                  "earthquake", "earthquakes"}


def _atom(symbol: str) -> UnitValue:
    if symbol in _MAGNITUDE_LABELS or symbol.lower() in _NO_UNIT_WORDS:
        return UnitValue(1.0, _ZERO)
    symbol = _WORD_ALIASES.get(symbol, symbol)
    if symbol in _BASE:
        f, d, _ = _BASE[symbol]
        return UnitValue(f, d)
    # Longest prefix first ("da" before "d").
    for p in sorted(_PREFIX, key=len, reverse=True):
        if symbol.startswith(p) and symbol[len(p) :] in _BASE:
            f, d, prefixable = _BASE[symbol[len(p) :]]
            if prefixable:
                return UnitValue(_PREFIX[p] * f, d)
    raise UnitParseError(f"unknown unit {symbol!r}")


_TOKEN_RE = re.compile(r"\s*(?:(?P<num>[-+]?\d+)|(?P<sym>°C|°F|[A-Za-zµμΩ°%‰℃℉]+)|(?P<op>[\^/()·]))")


def _tokenize(text: str) -> list[tuple[str, str]]:
    toks: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            if text[pos].isspace():
                pos += 1
                continue
            raise UnitParseError(f"unexpected character {text[pos]!r} in unit {text!r}")
        kind = m.lastgroup or ""
        toks.append((kind, m.group(kind)))
        pos = m.end()
    return toks


class _Parser:
    """expr := term ( ('/' term) | ('·'? term) )* ; term := (atom | '(' expr ')') ('^' int)?"""

    def __init__(self, toks: list[tuple[str, str]]):
        self.toks, self.i = toks, 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def expr(self) -> UnitValue:
        val = self.term()
        while (tok := self.peek()) is not None and tok != ("op", ")"):
            if tok == ("op", "/"):
                self.take()
                val = val * (self.term() ** -1)
            else:
                if tok == ("op", "·"):
                    self.take()
                val = val * self.term()
        return val

    def term(self) -> UnitValue:
        tok = self.peek()
        if tok is None:
            raise UnitParseError("dangling operator in unit")
        if tok == ("op", "("):
            self.take()
            val = self.expr()
            if self.peek() != ("op", ")"):
                raise UnitParseError("unbalanced parenthesis in unit")
            self.take()
        elif tok[0] == "sym":
            self.take()
            val = _atom(tok[1])
        elif tok[0] == "num" and tok[1] == "1":  # "1/s"
            self.take()
            val = DIMENSIONLESS
        else:
            raise UnitParseError(f"unexpected token {tok[1]!r} in unit")
        if self.peek() == ("op", "^"):
            self.take()
            kind, num = self.take()
            if kind != "num":
                raise UnitParseError("exponent must be an integer")
            val = val ** int(num)
        elif self.peek() is not None and self.peek()[0] == "num":  # "m2" style exponent
            val = val ** int(self.take()[1])
        return val


def parse_unit(text: str) -> UnitValue:
    """Parse a (possibly compound) unit; empty string → dimensionless factor 1."""
    t = normalize_unit_text(text)
    if not t:
        return DIMENSIONLESS
    if t in _OFFSET_TEMPS:
        return UnitValue(_OFFSET_TEMPS[t][1], _d(K=1))
    p = _Parser(_tokenize(t))
    val = p.expr()
    if p.peek() is not None:
        raise UnitParseError(f"trailing tokens in unit {text!r}")
    return val


def to_si(value: float, unit_text: str) -> tuple[float, Dim]:
    """Convert ``value`` in ``unit_text`` to SI, handling offset temperatures."""
    t = normalize_unit_text(unit_text)
    if t in _OFFSET_TEMPS:
        offset, scale = _OFFSET_TEMPS[t]
        return (value + offset) * scale, _d(K=1)
    u = parse_unit(t)
    return value * u.factor, u.dim
