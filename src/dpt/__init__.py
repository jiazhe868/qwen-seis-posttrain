"""dpt — Domain Post-Training for Qwen3.8-27B.

Package map (see README §4 and docs/dev/conventions.md for the layering rule):

    dpt.common      config loading, hashing/manifests, JSONL I/O, n-gram utils, seeding
    dpt.chat        chat-template rendering, reasoning-effort control, thinking split, loss masks
    dpt.data        RL problem-bank pipeline (dedup → decontam → answer verify → difficulty) + SFT mix
    dpt.rewards     training-time graders for T1–T4 (fail-closed), hack-pattern scanners
    dpt.rl          framework-free RL algorithm pieces (advantages, loss, TIS, curriculum, state)
    dpt.verl_ext    the only package that imports verl (adapter layer: trainer, reward fn, datasets)
    dpt.monitoring  alert rules, per-task panels, proxy/gold divergence, hack-rate tracking
    dpt.eval        evaluation harness with *independent* graders, statistics, ship gates
    dpt.launch      YAML → verl/Hydra overrides, pre-flight checks, run manifests (no verl import)
    dpt.parity      vLLM ↔ Megatron logprob parity and DeltaNet kernel checks (GPU)
    dpt.planning    memory / throughput / FLOPs calculator that reproduces the README tables
"""

__version__ = "0.1.0"
