"""Unit registry: conversions, dimension checks, temperature offsets, LaTeX unit strings."""

from __future__ import annotations

import math

import pytest

from dpt.rewards.units import DimensionError, UnitError, convert, parse_unit


@pytest.mark.parametrize(
    "value,src,dst,expected",
    [
        (3, "km", "m", 3000),
        (6, "km/s", "m/s", 6000),
        (2.7, "g/cm^3", "kg/m^3", 2700),
        (2.7, "g cm^-3", "kg m^-3", 2700),
        (2.7, "g·cm⁻³", "kg/m³", 2700),
        (30, "MPa", "Pa", 3e7),
        (1, "GPa", "MPa", 1000),
        (25, "mGal", "m/s^2", 2.5e-4),
        (65, "Ma", "yr", 6.5e7),
        (50, "nT", "T", 5e-8),
        (1013, "hPa", "Pa", 101300),
        (10, "m/s", "km/h", 36),
        (1, "km/s/s", "m/s^2", 1000),
        (1, "kg/(m·s)", "Pa s", 1),
        (2, r"\mathrm{km\,s^{-1}}", "m/s", 2000),
        (90, "deg", "rad", math.pi / 2),
        (90, r"^\circ", "rad", math.pi / 2),
        (50, "%", "", 0.5),
        (1, "cm3", "m^3", 1e-6),
        (20000, "ms", "s", 20),
        (1, "day", "h", 24),
        (3, "公里", "m", 3000),
    ],
)
def test_conversions(value, src, dst, expected):
    assert convert(value, src, dst) == pytest.approx(expected, rel=1e-9)


def test_standalone_temperature_uses_offset_but_gradient_does_not():
    assert convert(25, "degC", "K") == pytest.approx(298.15)
    assert convert(25, "°C", "K") == pytest.approx(298.15)
    assert convert(32, "degF", "degC") == pytest.approx(0.0, abs=1e-9)
    # geothermal gradient: a temperature *difference* per length
    assert convert(30, "°C/km", "K/m") == pytest.approx(0.03)


def test_exact_symbols_beat_prefix_splits():
    assert parse_unit("Pa").factor == 1.0  # pascal, not peta-annum
    assert parse_unit("min").factor == 60.0  # minute, not milli-inch
    assert parse_unit("h").factor == 3600.0  # hour
    assert parse_unit("ms").factor == pytest.approx(1e-3)  # millisecond


def test_dimension_mismatch_raises():
    with pytest.raises(DimensionError):
        convert(1, "s", "m")
    with pytest.raises(DimensionError):
        convert(1, "Pa", "J")


def test_unknown_unit_raises():
    with pytest.raises(UnitError):
        parse_unit("furlongs_per_fortnight")
    with pytest.raises(UnitError):
        parse_unit("km 3")
