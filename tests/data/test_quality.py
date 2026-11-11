"""Quality filter: the 4-gram repetition rule from the SFT loss-spike incident (report §阶段一踩到的坑)."""

from __future__ import annotations

from dpt.data.quality import QualityConfig, filter_by_quality, quality_issues

CFG = QualityConfig()
PROSE = (
    "Seismic waves slow down when they enter low-velocity sediments, and their amplitude grows to conserve "
    "energy flux. Resonance between the free surface and the sediment-bedrock interface further amplifies "
    "motion near the fundamental frequency, which depends on layer thickness and shear-wave velocity. "
    "Engineers estimate this frequency from ambient noise spectral ratios measured at the surface."
)


def test_normal_prose_passes():
    assert quality_issues(PROSE, CFG) == []


def test_ocr_loop_is_dropped():
    loop = " ".join(["The P arrival is marked at ten seconds on the record"] * 20)
    assert "repetition_4gram" in quality_issues(loop, CFG)


def test_short_and_garbage_texts():
    assert "too_short" in quality_issues("hi", CFG)
    assert "encoding_garbage" in quality_issues(PROSE[:100] + "�" * 5, CFG)


def test_threshold_is_configurable():
    loose = QualityConfig(max_ngram_repetition=0.99)
    loop = " ".join(["the same words again and again"] * 20)
    assert quality_issues(loop, loose) == []


def test_filter_reports_reasons():
    class Item:
        def __init__(self, uid, text):
            self.uid, self.text = uid, text

    res = filter_by_quality([Item("a", PROSE), Item("b", "x")], lambda i: i.text, CFG)
    assert [i.uid for i in res.kept] == ["a"] and res.reasons["too_short"] == 1
