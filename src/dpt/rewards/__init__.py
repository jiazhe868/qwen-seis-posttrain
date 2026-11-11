"""Training-time rewards for the four RL tasks (report §各任务的 reward).

Module map:

    base.py           RewardResult + the fail-closed decorator every grader uses
    answer_extract.py final-answer extraction, multi-answer / interval detection
    units.py          small self-contained unit registry (SI + geoscience units)
    numeric.py        T1 numeric grader and T4 math grader (binary, no partial credit)
    code_sandbox.py   T2: static tamper checks, local subprocess sandbox, remote sandbox client
    sandbox_server.py T2: stdlib HTTP worker for the CPU sandbox cluster
    citations.py      T3: reference parsing + Crossref / offline verification
    judge.py          T3: rubric judge over an OpenAI-compatible endpoint + calibration check
    instruction.py    T4: instruction-following rule checks + content-relevance check
    length.py         overlong penalty (20K→24K, up to −0.5) and truncation flag
    hack_patterns.py  regex/AST scanners for known reward-hack categories (also used by monitoring)
    router.py         verl entry points: compute_score / compute_score_batch
    validate.py       grader validation CLI (adversarial suite + human-label FP/FN rates)

These graders are deliberately a *different implementation* from the evaluation
graders in ``dpt.eval.graders`` (report §防范原则: training and evaluation graders
are implemented separately so the policy cannot overfit the eval grader).
"""
