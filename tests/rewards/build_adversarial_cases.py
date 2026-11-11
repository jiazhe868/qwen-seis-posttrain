"""Generate ``adversarial_cases.jsonl``, the seed of the grader adversarial suite.

The report's v3 grader was validated on 1,200 adversarial cases (§第二层 grader
判得对不对). This script builds the seed set that ships with the repo. It
covers every category named there: equivalent forms, units, multiple
candidates, intervals, scientific notation, and answers buried in long
reasoning. It also covers the specific hack shapes from the incident log
(steps 95 and 140). Grow the suite by adding cases here, never by editing the
JSONL by hand, and add a case for every grader bug found in audits.

    python tests/rewards/build_adversarial_cases.py   # rewrites adversarial_cases.jsonl

Each case: ``{"id", "category", "task", "response", "ground_truth", "expected"}``.
``response`` is the full model output (thinking + ``</think>`` + answer).
``expected`` is the correct *base* score (before any length penalty).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("adversarial_cases.jsonl")
THINK = "Let me work through the physics carefully.\n</think>\n\n"


def resp(answer: str, think: str = THINK) -> str:
    return think + answer


def build() -> list[dict]:
    cases: list[dict] = []

    def add(cat: str, answer: str, gt: dict, expected: int, task: str = "t1_num", raw: bool = False) -> None:
        cases.append(
            {
                "id": f"{cat}-{sum(c['category'] == cat for c in cases):03d}",
                "category": cat,
                "task": task,
                "response": answer if raw else resp(answer),
                "ground_truth": gt,
                "expected": expected,
            }
        )

    t20 = {"value": 20.0, "unit": "s", "rel_tol": 0.01, "abs_tol": 0.0}
    half = {"value": 0.5, "unit": "", "rel_tol": 1e-3, "abs_tol": 0.0}
    km3 = {"value": 3.0, "unit": "km", "rel_tol": 0.01, "abs_tol": 0.0}

    # 1. equivalent numeric forms (dimensionless)
    for a in [
        r"\boxed{0.5}",
        r"\boxed{1/2}",
        r"\boxed{\frac{1}{2}}",
        r"\boxed{\dfrac{2}{4}}",
        r"\boxed{50\%}",
        r"\boxed{5 \times 10^{-1}}",
        r"\boxed{5e-1}",
        r"\boxed{0.50}",
        r"$\boxed{.5}$",
        r"\boxed{500 ppm \times 1000}",
    ]:
        add("equivalent_forms", a, half, 0 if "ppm" in a else 1)

    # 2. unit conversion to the reference unit
    for a in [
        r"\boxed{3 km}",
        r"\boxed{3000 m}",
        r"\boxed{3.0 \mathrm{km}}",
        r"\boxed{300000 cm}",
        r"\boxed{3 \text{ km}}",
        r"\boxed{3\,\mathrm{km}}",
        r"\boxed{3.0e3 m}",
        r"\boxed{3 kilometers}",
        r"\boxed{3 公里}",
    ]:
        add("unit_conversion", a, km3, 1)
    for a, gt in [
        (r"\boxed{2.7 g/cm^3}", {"value": 2700, "unit": "kg/m^3"}),
        (r"\boxed{2700 kg/m^3}", {"value": 2.7, "unit": "g/cm^3"}),
        (r"\boxed{30 MPa}", {"value": 3e7, "unit": "Pa"}),
        (r"\boxed{0.03 GPa}", {"value": 30, "unit": "MPa"}),
        (r"\boxed{20000 ms}", t20),
        (r"\boxed{0.3333 min}", t20),
        (r"\boxed{65 Ma}", {"value": 6.5e7, "unit": "yr"}),
        (r"\boxed{25 mGal}", {"value": 2.5e-4, "unit": "m/s^2"}),
        (r"\boxed{25 °C}", {"value": 298.15, "unit": "K"}),
        (r"\boxed{30 °C/km}", {"value": 0.03, "unit": "K/m"}),
        (r"\boxed{6 km/s}", {"value": 6000, "unit": "m/s"}),
        (r"\boxed{6\ \mathrm{km\,s^{-1}}}", {"value": 6000, "unit": "m/s"}),
        (r"\boxed{21.6 km/h}", {"value": 6, "unit": "m/s"}),
        (r"\boxed{50 nT}", {"value": 5e-8, "unit": "T"}),
        (r"\boxed{1013 hPa}", {"value": 101300, "unit": "Pa"}),
        (r"\boxed{90°}", {"value": 1.5707963, "unit": "rad"}),
        (r"\boxed{2 Hz}", {"value": 0.5, "unit": "s^-1", "rel_tol": 0.01}),
    ]:
        add("unit_conversion", a, {**gt, "rel_tol": gt.get("rel_tol", 0.01)}, 0 if "Hz" in a else 1)

    # 3. wrong dimension
    for a in [r"\boxed{20 m}", r"\boxed{20 kg}", r"\boxed{20 m/s}", r"\boxed{20 Hz}", r"\boxed{20 Pa}"]:
        add("wrong_dimension", a, t20, 0)

    # 4. missing unit when the reference has one
    for a in [r"\boxed{20}", r"\boxed{20.0}", "Final answer: 20", r"\boxed{2 \times 10^1}"]:
        add("missing_unit", a, t20, 0)
    add("missing_unit", r"\boxed{35}", {"value": 35, "unit": "%"}, 0)

    # 5. multiple distinct candidates (step-95 hack)
    for a in [
        r"\boxed{20 s} or \boxed{25 s}",
        r"\boxed{20 s}" + "\n" + r"\boxed{25 s}",
        r"\boxed{20 s or 25 s}",
        r"\boxed{20 s, 25 s}",
        "Answer: 25 s\n\n" + r"\boxed{20 s}",
        r"\boxed{t = 20 s; t = 25 s}",
        r"\boxed{x = 20 s \text{ or } x = 25 s}",
        r"\boxed{20 s 或 25 s}",
        "Final answer: 20 s\nFinal answer: 25 s",
        r"The answer could be either \boxed{20 s} or \boxed{19 s}",
        r"\boxed{20 s} \boxed{2 s}",
    ]:
        add("multi_answer", a, t20, 0)

    # 6. the same value repeated is still one answer
    for a in [
        r"\boxed{20 s}" + "\n" + r"\boxed{20.0 s}",
        "Final answer: 20 s\n" + r"\boxed{20 s}",
        r"\boxed{20 s}" + "\n" + r"\boxed{20000 ms}",
    ]:
        add("same_value_repeated", a, t20, 1)

    # 7. intervals / ranges / ± / inequalities (step-140 hack)
    for a in [
        r"\boxed{3.1–3.4 km}",
        r"\boxed{about 3.1-3.4 km}",
        r"\boxed{between 2.9 and 3.1 km}",
        r"\boxed{3 \pm 0.1 km}",
        r"\boxed{3 ± 0.1 km}",
        r"\boxed{[2.9, 3.1] km}",
        r"\boxed{(2.9, 3.1) km}",
        r"\boxed{> 2.9 km}",
        r"\boxed{\geq 3 km}",
        r"\boxed{at least 3 km}",
        r"\boxed{3 km以上}",
        r"\boxed{3 km or more}",
        r"\boxed{2.9 km to 3.1 km}",
        r"\boxed{2.9~3.1 km}",
        r"\boxed{从 2.9 到 3.1 km}",
        "Final answer: roughly 2.9—3.1 km",
        r"\boxed{3 +/- 0.1 km}",
        r"\boxed{less than 3.1 km}",
    ]:
        add("interval", a, km3, 0)

    # 8. scientific notation
    sci = {"value": 6.0e-3, "unit": "m", "rel_tol": 0.01}
    for a in [
        r"\boxed{6.0 \times 10^{-3} m}",
        r"\boxed{6.0×10^-3 m}",
        r"\boxed{6.0e-3 m}",
        r"\boxed{6.0E-3 m}",
        r"\boxed{6 \cdot 10^{-3} m}",
        r"\boxed{6 mm}",
        r"\boxed{0.006 m}",
        r"\boxed{6.0 x 10^-3 m}",
        r"\boxed{6*10**-3 m}",
    ]:
        add("sci_notation", a, sci, 1)
    for a in [r"\boxed{6.0 \times 10^{3} m}", r"\boxed{6.0e3 m}", r"\boxed{6 m}"]:
        add("sci_notation", a, sci, 0)

    # 9. answer buried in long prose with no explicit final-answer marker
    long_prose = (
        "The P-wave travels through the crust at a nearly constant speed. Using the distance of 120 km and "
        "the velocity of 6 km/s we find that the travel time is 20 s, which is consistent with typical "
        "regional observations, although site effects could shift arrivals by a fraction of a second."
    )
    add("buried_answer", long_prose, t20, 0)
    add("buried_answer", long_prose + "\nSo it is 20 s.", t20, 0)
    add(
        "buried_answer",
        "Step 1: d = 120 km.\nStep 2: v = 6 km/s.\nStep 3: t = d/v.\nThus t equals twenty seconds "
        "after the origin time, as computed above in detail.",
        t20,
        0,
    )

    # 10. explicit marker formats
    for a in [
        "Final answer: 20 s",
        "Final Answer: 20 s",
        "**Final answer:** 20 s",
        "Answer: 20 s",
        "最终答案：20 s",
        "答案：20 秒",
        "The final answer is 20 s",
        "20 s",
        "Some explanation.\n\nFinal answer: 20 s.",
    ]:
        add("marker_formats", a, t20, 1)

    # 11. no end-of-thinking marker: never graded
    for a in [r"I think it is \boxed{20 s}", r"<think>\boxed{20 s}", "Final answer: 20 s"]:
        add("no_think_end", a, t20, 0, raw=True)

    # 12. tolerance edges
    for a, e in [
        (r"\boxed{19.81 s}", 1),
        (r"\boxed{20.19 s}", 1),
        (r"\boxed{19.7 s}", 0),
        (r"\boxed{20.3 s}", 0),
        (r"\boxed{21 s}", 0),
        (r"\boxed{-20 s}", 0),
    ]:
        add("tolerance", a, t20, e)
    add("tolerance", r"\boxed{0.004}", {"value": 0.0, "unit": "", "rel_tol": 0.01, "abs_tol": 0.005}, 1)
    add("tolerance", r"\boxed{0.006}", {"value": 0.0, "unit": "", "rel_tol": 0.01, "abs_tol": 0.005}, 0)

    # 13. equation chains inside the box
    for a, e in [
        (r"\boxed{t = 120/6 = 20 s}", 1),
        (r"\boxed{t = \frac{120 km}{6 km/s} = 20 s}", 1),
        (r"\boxed{t \approx 20 s}", 1),
        (r"\boxed{t = 20 s = 25 s}", 0),
        (r"\boxed{t = 25 s}", 0),
    ]:
        add("equation_chain", a, t20, e)

    # 14. LaTeX wrappers and spacing
    for a in [
        r"$\boxed{20\,\mathrm{s}}$",
        r"\boxed{20\text{ s}}",
        r"\boxed{\mathbf{20}\ s}",
        r"\boxed{20~s}",
        r"\[\boxed{20 \text{s}}\]",
        r"\boxed{20 \mathrm{s}}.",
    ]:
        add("latex_wrappers", a, t20, 1)

    # 15. thousands separators and negatives
    add("separators_signs", r"\boxed{1,000 m}", {"value": 1.0, "unit": "km"}, 1)
    add("separators_signs", r"\boxed{12,500 m}", {"value": 12.5, "unit": "km"}, 1)
    add("separators_signs", r"\boxed{−3.2 °C}", {"value": 269.95, "unit": "K"}, 1)
    add("separators_signs", r"\boxed{-3.2 K}", {"value": -3.2, "unit": "K"}, 1)
    add("separators_signs", r"\boxed{3.2 K}", {"value": -3.2, "unit": "K"}, 0)

    # 16. dimensionless labels (magnitudes)
    mag = {"value": 6.5, "unit": "", "rel_tol": 0.0, "abs_tol": 0.05}
    for a, e in [
        (r"\boxed{6.5}", 1),
        (r"\boxed{M_w = 6.5}", 1),
        (r"\boxed{6.5 Mw}", 1),
        (r"\boxed{M 6.5}", 1),
        (r"\boxed{6.5–6.7}", 0),
        (r"\boxed{6.4}", 0),
    ]:
        add("dimensionless_labels", a, mag, e)

    # 17. T4 general math
    for a, ref, e in [
        (r"\boxed{\frac{\sqrt{3}}{2}}", r"\frac{\sqrt{3}}{2}", 1),
        (r"\boxed{0.8660254037844386}", r"\frac{\sqrt{3}}{2}", 1),
        (r"\boxed{2\pi}", "6.283185307179586", 1),
        (r"\boxed{2\sqrt{2} + 3}", "5.82842712474619", 1),
        (r"\boxed{2\sqrt{2} - 1}", "1.8284271247461903", 1),
        (r"\boxed{B}", "B", 1),
        (r"\boxed{C}", "B", 0),
        (r"\boxed{x^2 + 1}", "x^2+1", 1),
        (r"\boxed{3} \boxed{4}", "3", 0),
        (r"\boxed{3 or 4}", "3", 0),
        (r"\boxed{0.33}", "1/3", 0),
        (r"\boxed{\frac{1}{3}}", "0.333333333333", 1),
        (r"\boxed{42}", "42", 1),
        (r"\boxed{3 \pm 1}", "3", 0),
        (r"\boxed{\sqrt[3]{27}}", "3", 1),
    ]:
        add("t4_math", a, {"kind": "math", "value": ref}, e, task="t4_general")

    # 18. plain wrong values
    for a in [r"\boxed{25 s}", r"\boxed{2 s}", r"\boxed{200 s}", "Final answer: 0 s"]:
        add("wrong_value", a, t20, 0)

    # 19. empty / garbage
    for a in ["", r"\boxed{}", r"\boxed{twenty seconds}", r"\boxed{NaN s}", r"\boxed{\infty}"]:
        add("garbage", a, t20, 0)
    return cases


def main() -> None:
    cases = build()
    with open(OUT, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
