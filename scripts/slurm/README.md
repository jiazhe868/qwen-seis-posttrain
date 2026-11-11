# Running on the cluster

The plan and the reasoning are in `README`; this page is the commands. Every job script
sources `env.sh`, which sets `DPT_HW_PROFILE=slurm_3xa100_40g` and `DPT_MODEL_LINE=qwen35_9b`, so the
repo's own scripts (`scripts/0*.sh`) run unchanged inside the jobs.

Replace `PROJ` with your Slurm account.
Submit every job **from the repo root**; logs land there as `dpt-<job>-<id>.out`.

| step | where | command | time | SU |
|---|---|---|---|---|
| 1 | login | `git clone … $WORK/nanogpt_seis_qwen && cd $_ && bash scripts/slurm/setup_login.sh conda` | 5 min | 0 |
| 2 | compute | `sbatch -A PROJ scripts/slurm/build_env.slurm` | 1–2 h | ≤ 5 |
| 3 | login | `bash scripts/slurm/setup_login.sh models` and `… sandbox` (the interpreter env; the image comes in step 4) | ~1 h | 0 |
| 4 | compute | `sbatch -A PROJ scripts/slurm/check_node.slurm` → must print `SANDBOX ISOLATED` | 10 min | ≤ 2 |
| 5 | compute | `sbatch -A PROJ scripts/slurm/parity.slurm` | 1–2 h | ≤ 12 |
| 6 | compute | `sbatch -A PROJ scripts/slurm/smoke.slurm` | ~2 h | ≤ 18 |
| 7 | compute | serve the judge: `sbatch -A PROJ -p gpu-a100-small scripts/slurm/serve.slurm judge RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic 1` (FP8 weights, one A100) | ≤ 48 h | 1.5/h |
| 7b | login | draft D-Report and D-Fresh with GPT-6 Luna (§4b): `python -m dpt.eval.drafts topics …`, then `report …` and `fresh …` | ~1 h | 0 ($) |
| 7c | compute | check the drafts and build the two sets (§4b): `sbatch -A PROJ scripts/slurm/eval_sets.slurm` (needs the judge from step 7; `normal` queue) | ~1–2 h | ~2 |
| 7d | login | public eval sets and D-Num (§4c): `import_general fetch` + `build`; `t1 build`; `drafts wording`; `t1 finalize-dnum` | ~1 h | 0 ($) |
| 8 | compute | serve `teacher` (Qwen3.8-27B, H100 node, TP2), `judge` and `base`; then `sbatch -A PROJ scripts/slurm/data.slurm gen`, `… bank`, `… sft` (§4c) | ~2 days | ~450 |
| 9 | compute | `sbatch -A PROJ scripts/slurm/eval.slurm base` | ~1 day | 60–100 |
| 10 | compute | `sbatch -A PROJ scripts/slurm/sft.slurm` then `eval.slurm sft` (+ `bash scripts/slurm/second_judge.sh sft` on a login node if compute nodes cannot reach the API) | 3 h + 1 h | ~120 |
| 11 | compute | `sbatch -A PROJ scripts/slurm/rl.slurm` (resubmits itself); on a login node, `bash scripts/slurm/judge_drift.sh` at least daily (§5) | ~3 days | 1,350–2,000 |
| 11b | compute (reservation) | RL v2 (README §8.3), a reservation window each: `scripts/slurm/submit_window.sh data <res>` (`data_window.slurm`: teacher FP8 TP2 + Gemma + the SFT model ×3 on 2 nodes, builds bank v2), then `scripts/slurm/submit_window.sh rl <res>` (judge ×3 on 1 node, `rl.slurm --config rl/rl_9b_v2.yaml` on 8, then `final_eval_window.slurm` on the RL nodes). The time limits come from the reservation's end. Without a reservation: `submit_window.sh data-queue -t 48:00:00`, then `submit_window.sh rl-queue` (`rl_fallback.slurm`: judge, RL and final eval in one heterogeneous job of 8 gpu-a100 nodes + 1 judge node; the cluster allows at most 8 gpu-a100 nodes per job) | 24 h + 24 h | ≤ 144 + ≤ 648 |
| 12 | login + reviewer | blind review of the final row (§4b): `python -m dpt.eval.blind_review page …` → reviewer → `… import …` | ~1.5 h of the reviewer | 0 |

## 1. Login node: Miniforge in `$WORK`

`$HOME` has a 10 GB quota, so conda, pip caches and models live in `$WORK` (1 TB). `setup_login.sh conda`
installs Miniforge at `$WORK/miniforge3` (override with `DPT_CONDA`). Login nodes are for light work only
(site policy): no compiling, no GPU jobs.

## 2. The training env on a compute node

`build_env.slurm` runs `scripts/env/build_train_local.sh` (the script that built the workstation's env) on
a 1-GPU node. It:
* reads the driver's CUDA version and installs CUDA 12.9 forward compatibility only if the driver is older
  (the workstation's 560 needed it; on a newer driver the compat library would break `cuInit`);
* compiles Transformer Engine's PyTorch extension for SM80 with the newest `cuda/12.8+` module
  (or `CUDA_TOOLKIT=/path/to/cuda`);
* ends with import checks and `pip check` (one known, harmless conflict: see the script).

**No network on compute nodes?** (`check_node.slurm` reports it.) On a login node:
`pip download -d $WORK/wheels -r requirements/train.txt` plus the vLLM wheel URL from the build script,
then rerun the build with `PIP_FIND_LINKS=$WORK/wheels PIP_NO_INDEX=1`.

## 3. Models and the sandbox

`setup_login.sh models` downloads into `$HF_HOME = $WORK/hf`: Qwen3.5-0.8B (smoke), Qwen3.5-9B (base),
Qwen3.8-27B (teacher), Gemma 4 26B-A4B (judge and second verification family). Add others with
`DPT_MODELS="org/model …"`.
Jobs run with `HF_HUB_OFFLINE=1`.

### Sandbox

RL executes tens of thousands of programs written by the policy. On one shared account they would run next
to your data and checkpoints. The T2 sandbox therefore runs each program with Apptainer `--containall`:
the program sees its own temporary directory and a read-only interpreter (`dpt-sandbox`: Python 3.12,
numpy, scipy, pandas), nothing else; `--net --network none` removes the network where Apptainer allows it.

`setup_login.sh sandbox` creates `dpt-sandbox`. `check_node.slurm` pulls a minimal image to
`$WORK/dpt/sandbox/sandbox.sif` (login nodes have no working Apptainer), then runs `python -m dpt.rewards.sandbox_check`, which tries to read your home, the
repo and the eval sets, write to `$WORK`, list `$SCRATCH` and open a network connection from inside the
sandbox. **Do not start RL unless it prints `SANDBOX ISOLATED`.** If only the network probe fails
(Apptainer here may not allow `--network none` for users), set `rewards.t2_code.sandbox.container_network=host`
for the run: the filesystem isolation is the essential part, and network imports are still flagged.
The RL launcher refuses a cluster run with no container image at all.

## 4. Endpoints between jobs

`serve.slurm <name> <model> [tp] [max_len]` serves one model on one node (TP × as many data-parallel
replicas as fit) and writes `$WORK/dpt/endpoints/<name>.url` and `<name>.model`; the files disappear when
the job ends. `rl.slurm` and `eval.slurm` read `judge`; the data jobs will read `teacher`, `base` and the
verification family. An MoE model (the Gemma judge) must not use vLLM data parallelism, which shards its
experts across the ranks: `DPT_SERVE_DP=1`, and `DPT_SERVE_REPLICAS=K` runs K independent servers behind
the least-busy proxy `scripts/rr_proxy.py` at the same address. A serve job lasts at most 48 h: resubmit it before a long RL chain outlives it
(`rl.slurm` refuses to start without a live judge, because T3 rewards would silently be 0).

| endpoint | model | where |
|---|---|---|
| `judge` | RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic (Gemma 4 26B-A4B, FP8 weights): T3 reward, primary D-Report judge, second verification family (plan §5-2) | 1 `gpu-a100-small` node (one A100, TP1), or a 2-hour `gpu-a100-dev` window with one replica per GPU: `DPT_SERVE_DP=1 DPT_SERVE_REPLICAS=3 sbatch -p gpu-a100-dev -t 02:00:00 serve.slurm judge <model> 1` |
| `teacher` | Qwen/Qwen3.8-27B | `-p gpu-h100` (2 × 80 GB, fastest) and/or `gpu-a100` at TP2 |
| `base` | Qwen/Qwen3.5-9B | `gpu-a100`, TP1 × DP3 |

### The second D-Report judge (GPT-6 Luna, OpenAI API)

D-Report is judged by the Gemma endpoint and by GPT-6 Luna at xhigh effort; the gate reads the lower win
rate (plan §5-2). Put your API key in a file only you can read; `env.sh` exports it:

    mkdir -p ~/.config/dpt && (umask 077; printf '%s\n' 'sk-...' > ~/.config/dpt/openai_key)

* If `check_node.slurm` lists `https://api.openai.com` as reachable, `eval.slurm` uses both judges itself.
* If it is BLOCKED, the eval job records the Gemma verdicts and leaves the second win rate pending (the
  gate fails closed). Then, on a login node: `bash scripts/slurm/second_judge.sh sft`. It re-grades
  D-Report from the cached generations (Gemma verdicts from `<row>/d_report/judgments.primary.jsonl`),
  replaces only D-Report in `results.json`, and rebuilds the report and gates. It makes only HTTP calls.
* After the first run, pin the snapshot the API reported (`by_judge.second.served_models` in
  `results.json`): `export DPT_SECOND_JUDGE_PIN=<it>` in `env.sh`. A silent model change then fails loudly.
* Luna's output is for eval only: OpenAI's terms forbid training on it, and no training code imports its
  client.

## 4b. Eval sets without experts (plan §6)

GPT-6 Luna drafts D-Report and D-Fresh on a login node; Gemma checks every item on the cluster. Keep the
drafts in `$WORK/dpt/drafts` (they become eval items: not in the repo, not shared).

    D=$WORK/dpt/drafts
    python -m dpt.eval.drafts topics --scope "<scope, plan §5-4>" --n 40 --out $D/topics.csv
    python -m dpt.eval.drafts report --topics $D/topics.csv --per-topic 4 --out $D/d_report_drafts.csv
    python -m dpt.eval.drafts fresh  --topics $D/topics.csv --per-topic 3 --out $D/fresh_drafts.jsonl
    sbatch -A PROJ scripts/slurm/eval_sets.slurm          # judge endpoint up (step 7); CPU node

`eval_sets.slurm` runs `dpt.eval.autocheck`, with Gemma thinking up to ~28K tokens per reply:

* D-Fresh: the solver code runs in the sandbox container and must reproduce the stated answer. Gemma then
  solves each problem 4 times. One match keeps it, unless another answer gets at least 2 votes and at
  least as many as the stated one, or nothing matches: then Gemma compares the two worked solutions,
  labelled A/B, in both orders, and the problem stays only if both times it upholds the stated answer and
  names the wrong step in the other.
* D-Report: 3 votes, each one Gemma writing its own answer first and then ruling on the question, every
  rubric criterion and every common error; the majority decides.

It writes `d_fresh.jsonl` and `d_report.jsonl` with manifests into the secure eval dir, and
`d_fresh_check.csv` / `d_report_check.csv` next to the drafts: one row per draft with its status, the
reason, the vote counts and the adjudication outcome and note. Read both once: a pattern in the drops (a
whole topic, one kind of unit) means redraft that part. `adjudicated_refuted` means Gemma found the
drafted answer wrong; `adjudicated_ambiguous`, a question that admits two readings. The job warns if more
than 5% of replies were cut off at the token budget. With fewer than 40 / 80 passing items nothing is
written; draft more (`--per-topic`) and rerun. An existing set is never overwritten: it may be frozen
(Oct 20). Before the freeze, delete it or set `DPT_EVAL_SETS_FORCE=1`.

**Blind review** (week 4, the one human step). After the final eval of a row, on a login node:

    R=$WORK/dpt/eval_results/rl_seed1
    python -m dpt.eval.blind_review page --row-dir $R --out review_batch1.html
    # the reviewer opens the file in a browser (offline), judges 20 pairs, downloads verdicts_batch1.csv
    python -m dpt.eval.blind_review import --row-dir $R --verdicts verdicts_batch1.csv
    # exit status 2: the reviewer and a judge differ by > 15 points → batch 2 (10 pairs), then import both
    python -m dpt.eval.blind_review page --row-dir $R --batch 2 --out review_batch2.html
    bash scripts/07_final_eval.sh --report-only           # the report shows the review under the table

Send only the HTML file, never `expert_review.KEY.jsonl`. The reviewer: anyone with domain training who has
not been reading this model's outputs.

## 4c. Training data and the generated eval sets

Light tools on a login node run from a small env without torch: `P=$WORK/miniforge3/envs/dpt-lite/bin/python`
(`conda create -n dpt-lite -c conda-forge python=3.12 numpy pyyaml httpx omegaconf pyarrow pandas
huggingface_hub`). `S=$WORK/dpt/eval_secure` (mode 700), `RAW=$WORK/dpt/raw`, `G=$WORK/dpt/data/gen`.

    # public eval sets + benchmark prompts for decontamination (GPQA needs `hf auth login` after accepting its terms)
    $P -m dpt.eval.import_general fetch --raw-dir $RAW/general --lcb-cutoff 2025-01-01
    $P -m dpt.eval.import_general build --raw-dir $RAW/general --out-root $S --decontam-out $S/public_benchmarks \
        --lcb-cutoff 2025-01-01
    # T1 training candidates + the D-Num draft (held-out families), then Luna's wording check, then D-Num
    $P -m dpt.data.gen.t1 build --out $G/t1_candidates.jsonl --heldout-out $S/d_num_draft.jsonl --seed 1
    $P -m dpt.eval.drafts wording --items $S/d_num_draft.jsonl --out $WORK/dpt/drafts/d_num_wording.csv
    $P -m dpt.data.gen.t1 finalize-dnum --draft $S/d_num_draft.jsonl --wording $WORK/dpt/drafts/d_num_wording.csv \
        --out $S/d_num.jsonl --perturbed-out $S/d_num_perturbed.jsonl
    # T4 sources
    $P -m dpt.data.gen.t4 fetch --raw-dir $RAW/t4

`data.slurm` (CPU node; models are the `teacher`, `judge` and `base` endpoints from `serve.slurm`):

* `gen`: T1 (skipped if `$G/.done/t1` exists), T2 code tasks + D-Code (references and mutants run in the sandbox),
  T4, T3 report prompts (teacher), T1 rewording (teacher, every number kept).
* `bank`: `dpt.data.pipeline` ingest → … → publish. Dedup treats generated problems by instance (same solver and
  inputs), not by text. Verification audits 20 instances per template with the teacher (medium effort) and Gemma
  and keeps or drops each template whole (`$G/../v1/reports/verify_groups.json`). Decontamination embeds
  in-process (Qwen3-Embedding-0.6B) and protects every eval set and `$S/public_benchmarks/`. The difficulty
  pass grades T3 only if `DPT_DIFFICULTY_SKIP` leaves `t3_report` in; by default it needs no judge, so it can
  run between judge windows. It keeps 32 problems (256 replies) in flight on one base server, with a 4 h
  request timeout: bank replies average ~16K tokens, so a request waits ~50 min there, and at 64 problems with
  the old 30 min timeout most requests were re-sent (2026-10-25). `DPT_DIFFICULTY_CONCURRENCY` sets the problems
  in flight when `base` is a proxy over several servers. If the bank is still unpublished when `campaign.sh`
  starts, the campaign runs the pass itself on 24 one-GPU replicas (~2 h): it writes `campaign/bank_takeover`,
  cancels the one-GPU base job named in `campaign/base_job`, and waits for the login-node runner to answer
  with `campaign/bank_released` (or for its heartbeat `campaign/bank_runner.alive` to go stale).
* `sft`: teacher rejection sampling (`<subset>.report.json` holds the 0-of-k share per family: the number that
  decides whether a stronger teacher is worth it for the hard slice), base self-distillation, then the mix.
  Each subset is resumable. Report problems that the judge could not grade, because a judge window closed,
  stay unfinished: after 8 of them the run stops with exit code 1 and the next run picks them up. Once all
  four subsets are marked done, the mix needs only the teacher, which judges paraphrases during decontamination.

Spot-check `$G/t3_spot_check.csv` (50 rubrics) once `gen` is done.

**The hard bank and D-Num-hard (2026-10-27).** The base 9B scores 91.6 on D-Num, so `dpt.data.gen.seismology_hard`
adds harder families (bank `hard`): 12 for training, 8 held out. They build like the 2026-10 ones, into their own
files; D-Num-hard (160 items, 20 per family) is an exception to the Oct 23 freeze, recorded in `$S/FROZEN.json`:

    $P -m dpt.data.gen.t1 build --bank hard --set-name d_num_hard --out $G/t1_hard_candidates.jsonl \
        --heldout-out $S/d_num_hard_draft.jsonl --per-family 600 --n-heldout 240 --seed 1
    $P -m dpt.eval.drafts --max-concurrency 6 wording --items $S/d_num_hard_draft.jsonl --set-name d_num_hard \
        --out $WORK/dpt/drafts/d_num_hard_wording.csv
    $P -m dpt.data.gen.t1 finalize-dnum --set-name d_num_hard --draft $S/d_num_hard_draft.jsonl \
        --wording $WORK/dpt/drafts/d_num_hard_wording.csv --out $S/d_num_hard.jsonl \
        --perturbed-out $S/d_num_hard_perturbed.jsonl --n-items 160 --n-perturbed 80

The bank stages run through `data.slurm bank` with the `DPT_BANK_*` variables in its header: their own output
directory (`$WORK/dpt/data/v1_hard`) and markers. Verification uses a second Gemma server named `judge_v`, so the
RL judge keeps `judge`. The difficulty audit uses the SFT model served as `sft_audit`.

## 5. RL: chained 48-hour jobs

**RL-entry gate.** `campaign.sh` runs it after the SFT eval. On the 9B line a failed gate holds the job up to
8 h (the table is in `$WORK/dpt/campaign/logs/<job>/entry_gate.log`). Answer with one word, written after the
hold began: `echo go > $WORK/dpt/campaign/GATE_DECISION` starts RL anyway, `stop` ends the job
(`state/GATE_FAILED`), and so does no answer in time. The gate polls once a minute.

`rl.slurm` runs one ≤ 48 h slice of the run:
1. Exits at once if the run is finished, and **refuses** if it is `PAUSED.json` (an alert stopped it; follow
   `docs/runbook.md`, fix, delete `PAUSED.json`, resubmit).
2. Queues its successor (`afterany`), starts Ray on all 8 nodes, exports `DPT_DEADLINE`, and launches.
   verl resumes from the latest checkpoint together with the curriculum, sampler, RNG and alert state.
3. The trainer saves and stops at the last step boundary that still leaves time for a checkpoint
   (`WALLTIME.json`); the successor continues. Any other ending (crash, OOM, pause, done) cancels the
   successor, so a broken run cannot burn SUs in a loop. `DPT_CHAIN_LEFT` (default 6) caps the chain.

Watch it: `tail -f dpt-rl-<id>.out`, `metrics.jsonl` in the run dir (`$SCRATCH/dpt/checkpoints/rl/rl9b_seed1`),
`squeue -u $USER`.

**Judge drift**, at least once a day on a login node: `bash scripts/slurm/judge_drift.sh`. verl dumps each
dev validation (every 20 steps) to `<run>/val_generations/`; GPT-6 Luna re-scores at least 40 of the T3
responses that the Gemma reward judge scored, with the same rubric prompt. That is the first sample of 40 dev
prompts, or every sample of the 9B's 12 (48 responses). It prints both judges' scores per step and
exits with status 3 when Gemma's scores pull away from Luna's (the gap grows by > 8 points, or Gemma rises
while Luna stays flat for 3 checks). Treat that like a `proxy_gold` alert (`docs/runbook.md` §3): read the
T3 samples, then decide whether to `scancel` and roll back. The trainer never reads this output. The 8-node jobs wait in a shared queue: ask the site for a reservation for the RL week
(plan §8).

## 6. Storage rules

`$SCRATCH` deletes files not accessed for 10 days, and the site forbids touching files to dodge that. After SFT,
copy `…/sft/sft9b_seed1/hf_export` to `$WORK`; after RL, copy the selected step's export. Everything else
the plan needs (data, eval sets, results, endpoints, envs, models) is already on `$WORK`.

## 7. Troubleshooting

| symptom | check |
|---|---|
| `ray_up` times out | ssh between your job's nodes must work (`ssh <other node> hostname` inside the job); `$RAY_TMPDIR` must be a short local path |
| NCCL hangs at start | `NCCL_DEBUG=INFO` in the job; InfiniBand should be picked automatically |
| `cuInit` error 803 | the env was built with cuda-compat but the driver is newer (or the reverse): rebuild with `CUDA_COMPAT=no` / `yes` |
| `import transformer_engine` fails | `conda activate dpt-train` first (the activate hook sets the library path); check `python -m dpt.launch.check_env` |
| T2 rewards all 0 | `sandbox_check`; the `t2_code` reason counts in `metrics.jsonl` (`infra_error` = sandbox unavailable, fail closed) |
| RL job ends with "without a clean stop" | the trainer crashed: the log above it; the successor was cancelled on purpose |
