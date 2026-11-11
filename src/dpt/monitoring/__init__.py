"""Training monitoring: metric names, per-task panels, alert rules, hack/divergence detectors, audits.

The report's top lesson is that monitoring and evaluation infrastructure,
more than the choice of algorithm, decides whether an RL project succeeds.
All five long-run incidents were numerical or systems problems. The earliest
signal, the train↔rollout log-ppl gap, moved 60 steps before reward did.

Modules
-------
``keys``        canonical metric names (the trainer emits them; rules consume them)
``panels``      per-task aggregation of per-sample arrays + gradient-share decomposition
``alerts``      rule engine for every row of the default monitoring panel (warn / pause / block_rollout)
``divergence``  proxy-vs-gold divergence detector (reward hacking signal #1)
``hack_rate``   per-category hack-pattern rate tracker with growth detection
``audit``       periodic human-audit and grader spot-check exports
``sinks``       JSONL / wandb metric sinks keyed by step
"""
