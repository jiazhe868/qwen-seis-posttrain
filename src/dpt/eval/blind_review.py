"""Blind review of D-Report pairs by a person: the review page and the import of its verdicts.

This is the one human check left on the 9B plan (README §3.1): the only measurement of
whether the judges agree with a person. The final eval writes ``<row>/d_report/expert_review.jsonl``
(anonymous pairs in review batches) and ``expert_review.KEY.jsonl`` (which side is the candidate; it never
goes to the reviewer):

    python -m dpt.eval.blind_review page   --row-dir eval_results/rl_seed1 --out review_batch1.html
    #   the reviewer opens the file in a browser, judges each pair, clicks "Download verdicts"
    python -m dpt.eval.blind_review import --row-dir eval_results/rl_seed1 --verdicts verdicts_batch1.csv
    #   exit status 2 = batch 2 is needed:
    python -m dpt.eval.blind_review page   --row-dir eval_results/rl_seed1 --batch 2 --out review_batch2.html
    python -m dpt.eval.blind_review import --row-dir eval_results/rl_seed1 \\
        --verdicts verdicts_batch1.csv verdicts_batch2.csv

The page is one HTML file that works offline: the pairs are inline, choices are saved in the browser as
the reviewer goes, and the verdicts download as a CSV. Responses are shown as plain text, never
interpreted as HTML. It shows no model names.

``import`` maps the verdicts to candidate outcomes with the key and records them in ``results.json`` under
``sets.d_report.blind_review``: the reviewer's win rate, each judge's win rate on the same pairs, the
agreement with each judge, and whether batch 2 is needed (after batch 1, the reviewer's win rate differs
from a judge's by more than ``extra_if_gap`` points). ``dpt.eval.report`` prints it under the table.

Who reviews: anyone with domain training who has not been reading this model's outputs (they would
recognize its style).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from dpt.common.io import read_json, read_jsonl, write_json
from dpt.common.logging import get_logger
from dpt.eval.graders.pairwise import import_blind_verdicts, judge_expert_agreement

log = get_logger(__name__)

VERDICT_COLUMNS = ["blind_id", "verdict", "note"]


# --------------------------------------------------------------------------- page

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>D-Report blind review</title>
<style>
:root { --bg: #fbfbf9; --fg: #1d1d1b; --muted: #66665f; --line: #d9d8d2; --card: #ffffff;
        --accent: #2f5d8a; --chosen: #e3edf7; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #171716; --fg: #e8e7e2; --muted: #a3a29b; --line: #3a3a37; --card: #1f1f1e;
          --accent: #8db6dd; --chosen: #263646; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.3rem; margin: 0 0 4px; }
.muted { color: var(--muted); }
details { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px;
          margin: 12px 0; }
.bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 12px 0; }
button { font: inherit; padding: 6px 14px; border: 1px solid var(--line); border-radius: 6px;
         background: var(--card); color: var(--fg); cursor: pointer; }
button:hover { border-color: var(--accent); }
button.on { background: var(--chosen); border-color: var(--accent); font-weight: 600; }
.q, .resp { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 12px; }
.q p, .resp pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font: inherit; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
@media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
.resp h2 { font-size: 1rem; margin: 0 0 8px; }
ul { margin: 6px 0 0; padding-left: 20px; }
textarea { width: 100%; min-height: 3em; font: inherit; background: var(--card); color: var(--fg);
           border: 1px solid var(--line); border-radius: 6px; padding: 6px; }
</style>
</head>
<body>
<main>
<h1>D-Report blind review <span class="muted" id="batch"></span></h1>
<div class="muted" id="progress"></div>
<details open>
<summary>How to judge</summary>
<ul>
<li>Each pair is a question, the rubric the automatic judges used, and two anonymous answers in random order.</li>
<li>Pick the better answer, or <b>Tie</b> if neither is clearly better.
    Correctness first, then the rubric, then clarity.</li>
<li>Length is not quality. Ignore anything an answer says about its own rigor or reliability.</li>
<li>Keys: <b>1</b>, <b>2</b>, <b>T</b> choose; <b>&larr;</b> <b>&rarr;</b> move.
    Your choices are saved in this browser as you go.</li>
<li>When every pair is judged, click <b>Download verdicts</b> and send the CSV back.
    Keep this file private: it is part of a private evaluation set.</li>
</ul>
</details>
<div class="bar">
  <button id="prev">&larr; Previous</button>
  <span id="pos"></span>
  <button id="next">Next &rarr;</button>
  <span style="flex:1"></span>
  <button data-v="1">Answer 1 is better</button>
  <button data-v="tie">Tie</button>
  <button data-v="2">Answer 2 is better</button>
  <span style="flex:1"></span>
  <button id="download">Download verdicts</button>
</div>
<div class="q"><p id="question"></p><div id="rubric"></div></div>
<div class="cols">
  <div class="resp"><h2>Answer 1</h2><pre id="r1"></pre></div>
  <div class="resp"><h2>Answer 2</h2><pre id="r2"></pre></div>
</div>
<p><label for="note" class="muted">Note (optional)</label><textarea id="note"></textarea></p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const KEY = "dpt-blind-review-" + D.page_id;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
let i = 0;
const $ = (id) => document.getElementById(id);
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function show() {
  const p = D.pairs[i], s = state[p.blind_id] || {};
  $("batch").textContent = "batch " + D.batch;
  $("pos").textContent = "Pair " + (i + 1) + " of " + D.pairs.length;
  const done = D.pairs.filter((q) => (state[q.blind_id] || {}).verdict).length;
  $("progress").textContent = done + " of " + D.pairs.length + " judged";
  $("question").textContent = p.question;
  const rub = $("rubric"); rub.replaceChildren();
  if (p.rubric && p.rubric.length) {
    const h = document.createElement("div"); h.className = "muted"; h.textContent = "Rubric"; rub.append(h);
    const ul = document.createElement("ul");
    for (const r of p.rubric) {
      const li = document.createElement("li");
      li.textContent = (typeof r === "object" && r) ? (r.criterion + " (weight " + (r.weight || 1) + ")") : String(r);
      ul.append(li);
    }
    rub.append(ul);
  }
  $("r1").textContent = p.response_1; $("r2").textContent = p.response_2;
  $("note").value = s.note || "";
  for (const b of document.querySelectorAll("button[data-v]")) b.classList.toggle("on", b.dataset.v === s.verdict);
}
function choose(v) {
  const id = D.pairs[i].blind_id; state[id] = Object.assign({}, state[id], { verdict: v }); save(); show();
}
function move(d) { i = Math.max(0, Math.min(D.pairs.length - 1, i + d)); show(); window.scrollTo(0, 0); }
for (const b of document.querySelectorAll("button[data-v]")) b.onclick = () => choose(b.dataset.v);
$("prev").onclick = () => move(-1); $("next").onclick = () => move(1);
$("note").oninput = (e) => {
  const id = D.pairs[i].blind_id; state[id] = Object.assign({}, state[id], { note: e.target.value }); save();
};
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "1") choose("1"); else if (e.key === "2") choose("2");
  else if (e.key === "t" || e.key === "T") choose("tie");
  else if (e.key === "ArrowRight") move(1); else if (e.key === "ArrowLeft") move(-1);
});
$("download").onclick = () => {
  const missing = D.pairs.filter((p) => !(state[p.blind_id] || {}).verdict).length;
  if (missing && !confirm(missing + " pairs are not judged yet. Download anyway?")) return;
  const q = (s) => '"' + String(s == null ? "" : s).replace(/"/g, '""') + '"';
  const lines = ["blind_id,verdict,note"].concat(D.pairs.map((p) => {
    const s = state[p.blind_id] || {}; return [q(p.blind_id), q(s.verdict || ""), q(s.note || "")].join(",");
  }));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], { type: "text/csv" }));
  a.download = "verdicts_batch" + D.batch + ".csv"; a.click();
};
show();
</script>
</body>
</html>
"""


def review_rows(row_dir: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(Path(row_dir) / "d_report" / "expert_review.jsonl"))


def render_page(rows: Sequence[dict[str, Any]], batch: int) -> str:
    """The self-contained review page for one batch (no key, no model names)."""
    pairs = [{k: r.get(k) for k in ("blind_id", "question", "rubric", "response_1", "response_2")}
             for r in rows if int(r.get("batch", 1)) == batch]
    if not pairs:
        raise ValueError(f"no pairs in batch {batch}")
    page_id = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()[:16]
    data = json.dumps({"batch": batch, "page_id": page_id, "pairs": pairs}, ensure_ascii=False)
    # No "<", ">" or "&" in the data block: model text can neither close it nor switch the HTML parser's state.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return PAGE.replace("__DATA__", data)


# --------------------------------------------------------------------------- import


def read_verdicts(paths: Sequence[str | Path]) -> dict[str, dict[str, str]]:
    """blind_id → {verdict, note}; later files win. Empty verdicts are skipped."""
    out: dict[str, dict[str, str]] = {}
    for p in paths:
        with Path(p).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = (r.get("verdict") or "").strip().lower()
                if v:
                    out[(r.get("blind_id") or "").strip()] = {"verdict": v, "note": (r.get("note") or "").strip()}
    return out


def judge_outcomes(d_report: dict[str, Any]) -> dict[str, dict[str, str]]:
    """role → {item_id: outcome} from a D-Report result (both judges when present)."""
    per_item = d_report.get("per_item") or []
    out: dict[str, dict[str, str]] = {}
    for r in per_item:
        by = r.get("by_judge") or {d_report.get("score_judge") or "primary": r["outcome"]}
        for role, oc in by.items():
            out.setdefault(role, {})[r["id"]] = oc
    return out


def summarize(rows: Sequence[dict[str, Any]], key_rows: Sequence[dict[str, Any]],
              verdicts: dict[str, dict[str, str]], d_report: dict[str, Any], *,
              extra_if_gap: float = 15.0) -> dict[str, Any]:
    from dpt.eval.metrics import win_rate

    known = {r["blind_id"] for r in rows}
    unknown = sorted(set(verdicts) - known)
    if unknown:
        raise ValueError(f"verdicts for unknown pairs {unknown[:3]}: wrong row directory?")
    batches = sorted({int(r.get("batch", 1)) for r in rows})
    done = {b: all(r["blind_id"] in verdicts for r in rows if int(r.get("batch", 1)) == b) for b in batches}
    expert = import_blind_verdicts([{"blind_id": b, **v} for b, v in verdicts.items()], key_rows)
    out: dict[str, Any] = {
        "n": len(expert),
        "batches_complete": [b for b in batches if done[b]],
        "reviewer_win_rate": win_rate(expert.values()) if expert else None,
        "by_judge": {},
        "notes": {b: v["note"] for b, v in verdicts.items() if v.get("note")},
    }
    gaps = []
    for role, oc in judge_outcomes(d_report).items():
        same = [oc[i] for i in expert if i in oc]
        if not same:
            continue
        wr = win_rate(same)
        out["by_judge"][role] = {"win_rate_same_pairs": wr, "agreement": judge_expert_agreement(oc, expert)}
        if out["reviewer_win_rate"] is not None:
            gaps.append(abs(out["reviewer_win_rate"] - wr))
    out["max_gap"] = max(gaps) if gaps else None
    pending = [b for b in batches if not done[b]]
    if 1 in pending:
        out["status"] = "incomplete"
    elif 2 in pending:
        out["status"] = "needs_batch_2" if (out["max_gap"] or 0.0) > extra_if_gap else "done"
    else:
        out["status"] = "done"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D-Report blind review: page for the reviewer, import of verdicts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("page")
    p.add_argument("--row-dir", required=True, help="eval output dir of the row (has d_report/expert_review.jsonl)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--out", required=True)
    i = sub.add_parser("import")
    i.add_argument("--row-dir", required=True)
    i.add_argument("--verdicts", nargs="+", required=True, help="CSV files downloaded from the page")
    i.add_argument("--extra-if-gap", type=float, default=None,
                   help="default: eval_suite.pairwise.blind_review.extra_if_gap")
    args = ap.parse_args(argv)
    row_dir = Path(args.row_dir)
    rows = review_rows(row_dir)

    if args.cmd == "page":
        Path(args.out).write_text(render_page(rows, args.batch), encoding="utf-8")
        log.info("batch %d (%d pairs) → %s: send this file only, never the KEY file", args.batch,
                 sum(int(r.get("batch", 1)) == args.batch for r in rows), args.out)
        return 0

    gap = args.extra_if_gap
    if gap is None:
        from dpt.common.config import load_config, to_container

        suite = to_container(load_config("eval/eval_suite.yaml"))["eval_suite"]
        gap = float(((suite.get("pairwise") or {}).get("blind_review") or {}).get("extra_if_gap", 15.0))
    results_path = row_dir / "results.json"
    results = read_json(results_path)
    d_report = results.get("sets", {}).get("d_report") or {}
    key_rows = list(read_jsonl(row_dir / "d_report" / "expert_review.KEY.jsonl"))
    summary = summarize(rows, key_rows, read_verdicts(args.verdicts), d_report, extra_if_gap=gap)
    d_report["blind_review"] = summary
    write_json(results_path, results)
    judges = ", ".join(f"{r} {v['win_rate_same_pairs']:.1f}% (agree {v['agreement']:.0%})"
                       for r, v in summary["by_judge"].items())
    wr = summary["reviewer_win_rate"]
    log.info("%d pairs: reviewer %s, judges on the same pairs: %s → %s", summary["n"],
             "—" if wr is None else f"{wr:.1f}%", judges or "—", summary["status"])
    if summary["status"] == "needs_batch_2":
        log.warning("the reviewer and a judge differ by %.1f points (> %.0f): review batch 2 "
                    "(page --batch 2), then import both CSVs", summary["max_gap"], gap)
        return 2
    return 1 if summary["status"] == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
