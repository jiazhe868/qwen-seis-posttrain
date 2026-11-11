"""Generators for the training data and the generated eval sets (README §3.1). Scope: seismology.

Every generator writes **candidate records**: one :class:`dpt.data.schema.Problem` dict per line, the input
format of ``dpt.data.pipeline`` (``data_pipeline.inputs.rl_candidates``). The pipeline then runs quality,
dedup, decontamination, verification and difficulty on them. Generators never filter for difficulty
themselves.

    common      candidate records, the license tag, deterministic sampling, JSONL output
    families    T1 numeric problem families (seismology): parameters, solver, question templates
    t1          T1 instances from the families (training) and D-Num items (held-out families)

Split rule: a family or task marked ``split="heldout"`` is used only for the eval sets (D-Num, D-Code).
Nothing from a held-out family may enter training data; ``dpt.data.decontam`` checks the result anyway.
The held-out definitions are not published: they live in the ``*_heldout.py`` modules, kept out of git,
and are imported when present (``common.import_private_module_if_present``; README, "Data & licensing").

Model output: the teacher (Qwen3.8-27B) and the judge (Gemma) may write training data. The hosted eval
model (``dpt.eval.hosted``) may not, and nothing here imports it (``tests/eval/test_independence.py``).
"""
