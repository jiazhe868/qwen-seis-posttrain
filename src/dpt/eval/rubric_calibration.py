"""How well a rubric judge agrees with gold labels: false passes and false fails, per criterion and per answer.

The T3 reward is a judge (Gemma) scoring a report against a rubric. A reward is only as good as its agreement with
what an expert would say, and §6.5 of the README found the judge lenient. On 2026-11-15 a gold set was made for
D-Report: a reference plan and report per question, then gold labels for the 300 D-Report answers of base_medium,
sft and rl_seed1 (each rubric criterion met / partly / not met, an overall 0-100 score and an accept verdict), by
GPT-6.1 Sol through the OpenAI API. OpenAI's terms forbid training on its output, so the labels only measure and
select rubric and judge versions; they never enter training data or a reward prompt.

    python -m dpt.eval.rubric_calibration --labels labels_sol.jsonl --scores gemma_scores.jsonl \
        --d-report d_report.jsonl --out rubric_calibration.json

Agreement is counted on two levels, with "positive" meaning met (criterion) or pass (answer):
  criterion  the judge's score of one criterion (0-10, met at >= the criterion threshold; or a yes/no check) against
             the gold mark ("yes" positive, "no" negative, "partly" counted apart). Criteria the gold labeller flags
             as wrong or ambiguous are counted apart too: they are rubric faults, not judge faults.
  answer     the judge's weighted rubric score (pass at >= the pass mark) against the gold, in two ways: the gold
             score at >= 60 ("a sound report with fixable issues", a third of the 300 answers), and the strict gold
             verdict (accept or not, 8 of 300); and Spearman's rho between the judge's score and the gold score.
"Partly" marks (43% of the gold criterion marks) are reported both set apart and counted as not met.
The questions are split 60/40 by a hash of their id. Thresholds are chosen on the tune split (best balanced accuracy)
and reported on the test split, so the test numbers are not fitted to themselves.

Rubric v2 is computed from the judge's own per-criterion scores, so trying it needs no new judge run:
  +gate   the main-result criterion of each question (gate_index: the highest-weight criterion that states a number,
          95 of the 100 D-Report rubrics have one) caps the score when the judge marks it unmet. The audit of the gold
          labels found partial credit for coverage lifting reports whose main result is wrong; this gate stops a
          reward from doing the same. The cap is chosen on the tune split.
  +clean  criteria flagged as wrong or ambiguous (--drop-criteria) are left out and the weights renormalised.

Three-level judging (kind "levels", 2026-11-16): each criterion full / partial / none, stored as 1 / 0.5 / 0. A
criterion counts as met only when full, and the judge's levels are tabled against the gold's yes / partly / no. The
reward's partial credit is a choice, so +partial0.25 and +partial0 rescore the same verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from dpt.common.io import read_jsonl, write_json

CRITERION_THRESHOLDS = (5, 6, 7, 8, 9)               # a 0-10 criterion counts as met at >= this score
PASS_MARKS = tuple(range(40, 95, 5))                  # an answer passes at a weighted score >= this (0-100)


def split_of(item_id: str, tune_share: float = 0.6, salt: str = "rubric-calibration-2026-11-15") -> str:
    """'tune' or 'test', fixed by a hash of the question id, so every rubric version sees the same split."""
    h = int(hashlib.sha256(f"{salt}:{item_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "tune" if h < tune_share else "test"


def confusion(pairs: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    """Counts and rates for (judge positive, gold positive) pairs. FP rate = FP / gold negatives (false passes);
    FN rate = FN / gold positives (false fails)."""
    tp = fp = tn = fn = 0
    for judge_says, gold_says in pairs:
        if judge_says and gold_says:
            tp += 1
        elif judge_says:
            fp += 1
        elif gold_says:
            fn += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    fp_rate = fp / (fp + tn) if fp + tn else None
    fn_rate = fn / (fn + tp) if fn + tp else None
    balanced = None if fp_rate is None or fn_rate is None else 1 - (fp_rate + fn_rate) / 2
    agree = (tp + tn) / n if n else None
    # Cohen's kappa: agreement beyond what the two marginals give by chance
    chance = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n) if n else None
    kappa = None if not n or chance == 1 else (agree - chance) / (1 - chance)
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "fp_rate": fp_rate, "fn_rate": fn_rate,
            "balanced_accuracy": balanced, "agreement": agree, "kappa": kappa}


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return cov / var if var else None


def criterion_pairs(labels: dict[tuple[str, str], dict[str, Any]], scores: dict[tuple[str, str], dict[str, Any]],
                    threshold: float, keys: Iterable[tuple[str, str]], partly_as_no: bool = False
                    ) -> tuple[list[tuple[bool, bool]], dict[str, int]]:
    """(judge met, gold met) for every criterion of the given answers, plus the counts set apart. ``partly_as_no``
    counts a "partly" mark as not met instead of setting it apart."""
    pairs, apart = [], {"partly": 0, "criterion_issue": 0, "missing": 0}
    for key in keys:
        lab, sc = labels.get(key), scores.get(key)
        if not lab or not sc or lab.get("no_answer") or sc.get("no_answer") or "per_criterion" not in sc:
            continue
        flagged = {int(c.get("id", 0)) for c in lab.get("criterion_issues") or []}
        marks = {int(c.get("id", 0)): str(c.get("met", "")).lower() for c in lab.get("criteria") or []}
        unit = sc.get("kind") in ("checks", "levels")        # 1 = yes / full, rather than a 0-10 score
        ids = sc.get("kept_ids") or range(1, len(sc["per_criterion"]) + 1)
        for i, judge_value in zip(ids, sc["per_criterion"]):
            mark = marks.get(i)
            if mark is None:
                apart["missing"] += 1
            elif i in flagged:
                apart["criterion_issue"] += 1
            elif mark == "partly" and not partly_as_no:
                apart["partly"] += 1
            else:
                judge_met = judge_value >= 1 if unit else judge_value >= threshold
                pairs.append((judge_met, mark == "yes"))
    return pairs, apart


def answer_pairs(labels: dict[tuple[str, str], dict[str, Any]], scores: dict[tuple[str, str], dict[str, Any]],
                 pass_mark: float, keys: Iterable[tuple[str, str]], gold_cut: float | None = None
                 ) -> tuple[list[tuple[bool, bool]], list, list]:
    """(judge pass, gold pass) per answer, and the paired scores for Spearman's rho. The gold pass is the accept
    verdict, or a gold score >= ``gold_cut`` when given. An answer with no final text fails on both sides."""
    pairs, judge_scores, gold_scores = [], [], []
    for key in keys:
        lab, sc = labels.get(key), scores.get(key)
        if not lab or not sc or "score" not in sc or "acceptable" not in lab:
            continue
        judge_score = 100.0 * float(sc["score"])
        gold_pass = bool(lab["acceptable"]) if gold_cut is None else float(lab.get("score") or 0.0) >= gold_cut
        pairs.append((judge_score >= pass_mark, gold_pass))
        judge_scores.append(judge_score)
        gold_scores.append(float(lab.get("score") or 0.0))
    return pairs, judge_scores, gold_scores


GOLD_CUT = 60.0                                       # gold score of a sound report with fixable issues
GATE_CAPS = (0.3, 0.4, 0.5)                           # a failed main-result criterion caps the score at this
GATE_MET = 7                                          # a 0-10 gate criterion counts as met at >= this score


def gate_index(item: dict[str, Any]) -> int:
    """0-based index of the question's main-result criterion: the highest-weight criterion that states a number
    (the first of equals), else the highest-weight criterion."""
    crit = item["answer"]["rubric"]
    numeric = [i for i, c in enumerate(crit) if any(ch.isdigit() for ch in c["criterion"])]
    pool = numeric or list(range(len(crit)))
    return max(pool, key=lambda i: (float(crit[i]["weight"]), -i))


def derived_scores(scores: dict[tuple[str, str], dict[str, Any]], items: dict[str, dict[str, Any]], *,
                   gate_cap: float | None = None, drop: dict[str, set[int]] | None = None,
                   partial_credit: float | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """Rubric v2 from the judge's per-criterion scores: drop the listed criteria (1-based ids) and renormalise the
    weights, then cap the score at ``gate_cap`` when the main-result criterion is unmet. ``partial_credit``
    rescores a three-level judge's "partial" verdicts (stored as 0.5)."""
    out = {}
    for key, sc in scores.items():
        item = items[key[1]]
        if sc.get("no_answer") or "per_criterion" not in sc or not sc["per_criterion"]:
            out[key] = sc
            continue
        weights = [float(c["weight"]) for c in item["answer"]["rubric"]]
        dropped = (drop or {}).get(key[1], set())
        keep = [i for i in range(len(weights)) if i + 1 not in dropped and i < len(sc["per_criterion"])]
        unit = sc.get("kind") in ("checks", "levels")
        scale = 1.0 if unit else 10.0

        def credit(v: float) -> float:
            return partial_credit if partial_credit is not None and sc.get("kind") == "levels" and v == 0.5 else v
        total = sum(weights[i] for i in keep)
        credited = sum(credit(float(sc["per_criterion"][i])) * weights[i] for i in keep)
        score = credited / (scale * total) if total else 0.0
        g = gate_index(item)
        if gate_cap is not None and g in keep:
            met = sc["per_criterion"][g] >= (1 if unit else GATE_MET)
            score = score if met else min(score, gate_cap)
        out[key] = {**sc, "score": score,
                    "per_criterion": [sc["per_criterion"][i] for i in keep],
                    "kept_ids": [i + 1 for i in keep]}
    return out


def level_table(labels: dict, scores: dict, keys: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """For a three-level judge: how often each gold mark (yes / partly / no) got each level (full / partial /
    none). Criteria the gold labeller flags as faulty are left out."""
    table = {g: {"full": 0, "partial": 0, "none": 0} for g in ("yes", "partly", "no")}
    for key in keys:
        lab, sc = labels.get(key), scores.get(key)
        if not lab or not sc or sc.get("kind") != "levels" or "per_criterion" not in sc:
            continue
        flagged = {int(c.get("id", 0)) for c in lab.get("criterion_issues") or []}
        marks = {int(c.get("id", 0)): str(c.get("met", "")).lower() for c in lab.get("criteria") or []}
        ids = sc.get("kept_ids") or range(1, len(sc["per_criterion"]) + 1)
        for i, v in zip(ids, sc["per_criterion"]):
            if i not in flagged and marks.get(i) in table:
                table[marks[i]]["full" if v >= 1 else "partial" if v > 0 else "none"] += 1
    return table


def calibrate_variant(labels: dict, scores: dict, keys_by_split: dict[str, list]) -> dict[str, Any]:
    """Choose the criterion threshold and the pass marks on the tune split, report both splits at those values."""
    tune = keys_by_split["tune"]

    def best(candidates, pairs_of):
        return max(candidates, key=lambda c: confusion(pairs_of(c, tune))["balanced_accuracy"] or 0.0)

    threshold = best(CRITERION_THRESHOLDS, lambda t, k: criterion_pairs(labels, scores, t, k)[0])
    pass_60 = best(PASS_MARKS, lambda m, k: answer_pairs(labels, scores, m, k, GOLD_CUT)[0])
    pass_accept = best(PASS_MARKS, lambda m, k: answer_pairs(labels, scores, m, k)[0])
    out: dict[str, Any] = {"criterion_threshold": threshold, "pass_mark_vs_gold60": pass_60,
                           "pass_mark_vs_accept": pass_accept}
    for split, keys in keys_by_split.items():
        cp, apart = criterion_pairs(labels, scores, threshold, keys)
        cp_no = criterion_pairs(labels, scores, threshold, keys, partly_as_no=True)[0]
        ap60, js, gs = answer_pairs(labels, scores, pass_60, keys, GOLD_CUT)
        out[split] = {"criterion": {**confusion(cp), "set_apart": apart},
                      "criterion_partly_as_no": confusion(cp_no),
                      "answer_vs_gold60": confusion(ap60),
                      "answer_vs_accept": confusion(answer_pairs(labels, scores, pass_accept, keys)[0]),
                      "answer_at_70_vs_gold60": confusion(answer_pairs(labels, scores, 70, keys, GOLD_CUT)[0]),
                      "spearman": spearman(js, gs)}
        if any(sc.get("kind") == "levels" for sc in scores.values()):
            out[split]["levels"] = level_table(labels, scores, keys)
    return out


def rubric_validity(labels: dict, rubric_weights: dict[str, list[float]],
                    keys: Iterable[tuple[str, str]]) -> float | None:
    """Spearman's rho between the gold labeller's own rubric score (weights of the criteria it marks met, partly as
    half) and its overall score. A low value means that applying the rubric perfectly would still miss quality."""
    rubric_scores, overall = [], []
    for key in keys:
        lab = labels.get(key)
        if not lab or lab.get("no_answer") or not lab.get("criteria"):
            continue
        weights = rubric_weights[key[1]]
        marks = {int(c.get("id", 0)): str(c.get("met", "")).lower() for c in lab["criteria"]}
        credit = sum(w * {"yes": 1.0, "partly": 0.5}.get(marks.get(i, "no"), 0.0) for i, w in enumerate(weights, 1))
        rubric_scores.append(100.0 * credit / sum(weights))
        overall.append(float(lab.get("score") or 0.0))
    return spearman(rubric_scores, overall)


def summarize(label_rows: Sequence[dict[str, Any]], score_rows: Sequence[dict[str, Any]],
              items: Sequence[dict[str, Any]], drop: dict[str, set[int]] | None = None) -> dict[str, Any]:
    labels = {(r["row"], r["id"]): r for r in label_rows}
    item_by_id = {it["id"]: it for it in items}
    weights = {it["id"]: [float(c["weight"]) for c in it["answer"]["rubric"]] for it in items}
    by_variant: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for r in score_rows:
        if "error" not in r:
            by_variant[r["variant"]][(r["row"], r["id"])] = r
    keys = sorted(labels)
    keys_by_split = {"tune": [k for k in keys if split_of(k[1]) == "tune"],
                     "test": [k for k in keys if split_of(k[1]) == "test"]}

    def tune_balanced(sc: dict) -> float:
        best_mark = max(PASS_MARKS, key=lambda m: confusion(
            answer_pairs(labels, sc, m, keys_by_split["tune"], GOLD_CUT)[0])["balanced_accuracy"] or 0.0)
        return confusion(answer_pairs(labels, sc, best_mark, keys_by_split["tune"], GOLD_CUT)[0])[
            "balanced_accuracy"] or 0.0

    chosen_caps = {}
    for base in list(by_variant):
        if any(sc.get("kind") == "levels" for sc in by_variant[base].values()):
            for credit in (0.25, 0.0):
                by_variant[f"{base}+partial{credit:g}"] = derived_scores(by_variant[base], item_by_id,
                                                                         partial_credit=credit)
        variants = {"+gate": None, "+clean": drop or {}, "+clean+gate": drop or {}}
        for suffix, drop_set in variants.items():
            if "clean" in suffix and not drop:
                continue
            if "gate" in suffix:
                cap = max(GATE_CAPS, key=lambda c: tune_balanced(
                    derived_scores(by_variant[base], item_by_id, gate_cap=c, drop=drop_set)))
                chosen_caps[base + suffix] = cap
            else:
                cap = None
            by_variant[base + suffix] = derived_scores(by_variant[base], item_by_id, gate_cap=cap, drop=drop_set)
    return {"answers": len(keys), "questions": {s: len({k[1] for k in v}) for s, v in keys_by_split.items()},
            "gold_accept_rate": sum(bool(labels[k].get("acceptable")) for k in keys) / max(1, len(keys)),
            "rubric_validity": {s: rubric_validity(labels, weights, v) for s, v in keys_by_split.items()},
            "gate_caps": chosen_caps,
            "variants": {v: calibrate_variant(labels, sc, keys_by_split) for v, sc in sorted(by_variant.items())}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agreement of a rubric judge with gold labels (FP / FN)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--d-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-criteria", default=None, help="JSON {item id: [1-based criterion ids]} for +clean")
    a = ap.parse_args(argv)

    drop = ({k: set(v) for k, v in json.loads(Path(a.drop_criteria).read_text()).items()}
            if a.drop_criteria else None)
    res = summarize(list(read_jsonl(a.labels)), list(read_jsonl(a.scores)), list(read_jsonl(a.d_report)), drop)
    write_json(Path(a.out), res)

    def f(x):
        return "—" if x is None else f"{x:.2f}"
    for v, r in res["variants"].items():
        t = r["test"]
        print(f"{v}: criterion >= {r['criterion_threshold']}, pass mark {r['pass_mark_vs_gold60']} | test: criterion "
              f"FP {f(t['criterion']['fp_rate'])} FN {f(t['criterion']['fn_rate'])} kappa {f(t['criterion']['kappa'])}"
              f" | answer vs gold>=60 FP {f(t['answer_vs_gold60']['fp_rate'])} FN {f(t['answer_vs_gold60']['fn_rate'])}"
              f" | at 70: FP {f(t['answer_at_70_vs_gold60']['fp_rate'])} FN {f(t['answer_at_70_vs_gold60']['fn_rate'])}"
              f" | rho {f(t['spearman'])}")
    print(json.dumps({"rubric_validity": res["rubric_validity"], "gold_accept_rate": res["gold_accept_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
