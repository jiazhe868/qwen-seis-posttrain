"""Evaluation: fixed harness, independent graders, statistics, ship gates and report tables.

The evaluation system accepts only three things (report §评测体系):

1. **A fixed harness** (:mod:`dpt.eval.harness`). Every model is re-run in
   the same harness, base included. Template, reasoning effort, max length
   and sampling parameters are fixed, and no public number is ever copied.
2. **A quantified noise floor** (:mod:`dpt.eval.stats`). A difference counts
   as an improvement only if it exceeds the 95% CI. Checkpoints are selected
   on dev, never on test.
3. **Independently verified answers and graders** (:mod:`dpt.eval.graders`).
   This is a *separate implementation* from the training graders in
   :mod:`dpt.rewards`, so RL cannot overfit the evaluation grader.

Modules
-------
``graders/``      numeric (sympy + own unit table), code (hidden tests), ifeval, multiple choice,
                  tool call, pairwise judge (both orders), refusal detection
``datasets``      eval-set loaders (secure storage, manifests) and the general-benchmark registry
``harness``       OpenAI-compatible generation client with cache/resume, grading, results.json
``metrics``       avg@k, unbiased pass@k, pass@k curves, win rate with ties, token/truncation stats
``stats``         bootstrap CIs, paired bootstrap, noise floor, Wilson intervals, significance
``consistency``   reasoning↔answer consistency, perturbation gap, pass@64 floor check
``gates``         ship gates + RL-entry gate from ``configs/project.yaml``
``report``        markdown tables in the format of the report's final results table
"""
