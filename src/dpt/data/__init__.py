"""Data pipeline: RL problem bank and SFT mixture (report §数据, §处理流水线, §污染检查).

Stage order for the RL problem bank (each stage writes a manifested JSONL, so a
run can resume from any stage and every number in the funnel is auditable)::

    ingest      parse + license allow-list + schema validation       → 91k candidates
    quality     4-gram repetition / OCR-garbage / length filters
    dedup       MinHash LSH, char 5-gram, Jaccard ≥ 0.8 (within + across banks)  → 82k
    decontam    13-gram, MinHash ≥ 0.5, embedding+LLM paraphrase, answer-level  → 79.5k (−3.1%)
    verify      solver + 2 model families × 8 must agree with the label         → 74.1k (−6.8%)
    difficulty  base @ medium, 8 samples: drop 8/8, route 0/8 to the hard pool  → 60k (7k hard)
    publish     verl parquet/JSONL (main + hard pool) + manifests + contamination report

The SFT mixture (``sft_build``) reuses the same quality and decontamination code
and adds teacher-recitation checks, the 25% self-distillation replay share, chat
rendering with explicit reasoning effort, and length bucketing.

Module map::

    schema        Problem / SFTSample records and validation
    prompts       task system prompts + final-answer formats (shared by RL export and eval)
    ingest        parser protocol (layout-aware OCR output), LaTeX normalization, license filter
    quality       repetition / garbage / length checks
    dedup         self-contained numpy MinHash + LSH
    decontam      the four contamination layers, canaries, memorization probes, report
    solvers       programmatic solvers (registry) used by verification and perturbation checks
    answer_verify three-way agreement, expert queue, perturbation consistency
    difficulty    pass@8 calibration and hard pool
    backends      OpenAI-compatible HTTP clients (vLLM) + deterministic offline fakes
    sft_build     SFT mixture, render check, buckets, padding stats
    rl_build      Problem → verl row conversion and export
    pipeline      CLI orchestrating the stages
"""
