# RL ablations (report §消融实验)

Each file is an *overlay*: `python -m dpt.launch.rl --config <base> --ablation <name>` merges it
on top of the base config. The `ablation:` block records the hypothesis, the scale it runs at,
and the decision taken, so the result table in the README can always be regenerated and checked.

Run order matters. Screen on the 9B proxy first (`rl/rl_9b_proxy.yaml`, one node). Then confirm
the conclusions that matter on short 27B runs (`rl/rl_27b.yaml`, 80–250 steps). Directions
from 9B usually transfer; **numbers do not** (A8: the 9B learning rate fails completely at 27B).

| id | file | base | steps | question | report result → decision |
|---|---|---|---|---|---|
| A1 | `A01_kl_0p001` | 27B | 100 | KL 0.001 vs 0 | +3.0 vs +4.1, similar entropy decay → **KL 0** |
| A2 | `A02_std_norm` | 9B | 200 | divide by group std? | no-std +0.9 D-Num, +2.3 hard pass@8 → **no std** |
| A3 | `A03_clip_high_0p2` | 9B | 200 | clip-high 0.2 vs 0.28 | 0.28 keeps 62% entropy (vs 35%), +1.4 → **0.28** |
| A4 | `A04_no_tis` | 27B | 250 | TIS on vs off | off: collapse at step 180 (Incident 1) → **on** |
| A5 | `A05_effort_xhigh` | 27B | 100 | rollout effort xhigh vs medium | +1.2 but 2.6× tokens/step → **medium** |
| A6 | `A06_group16` | 9B | 200 | 16 vs 8 samples/prompt | +0.6 at 2× generation → **8** |
| A7 | `A07_t1_partial_credit` | 9B | 200 | partial credit for T1 | train reward up, D-Num −1.8 → **binary** |
| A8 | `A08_lr_2e-6` | 27B | 80 | LR 2e-6 vs 1e-6 | 3× entropy oscillation, stalls after 60 → **1e-6** |
| A9 | `A09_lora_r64` | 27B | 200 | LoRA r64 vs full | 82% of full gain, sync 10 s → 1 s → **full (LoRA = fast lane)** |
| A10 | `A10_no_filter_tokenmean` | 27B | 150 | zero-adv filtering + fixed constant vs original | curves overlap, −4.4 min/step → **filter** |
| A11 | `A11_no_mtp_spec` | 27B | 150 | MTP speculative decoding on/off | −32% gen time, no mismatch change → **on** |
| A12 | `A12_separate_async` | 27B | 60 | separate-node async vs colocated | 31 vs 27 min/step → **colocated** |

How 9B conclusions transferred to 27B: A2, A3 and A6 held in direction and size; A8 did not
transfer at all; A7 held in direction, but the gap shrank to 0.9.
