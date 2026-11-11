"""The pure compare step of the logprob parity tool (GPU phases are exercised on hardware)."""

from __future__ import annotations

import numpy as np

from dpt.parity.logprob_parity import compare


def _rows(n_tok, drift_after=None, noise=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(4):
        ref = rng.normal(-1.0, 0.5, n_tok)
        roll = ref + rng.normal(0, noise, n_tok)
        if drift_after is not None:
            roll[drift_after:] += 0.05            # a late-position-only gap (Incident 1 signature)
        rows.append({"response_ids": list(range(n_tok)), "rollout_logp": roll.tolist(), "ref_logp": ref.tolist()})
    return rows


def test_matched_engines_pass():
    assert compare(_rows(14000))["pass"]


def test_late_position_drift_fails():
    res = compare(_rows(14000, drift_after=12288))
    assert not res["pass"]
    assert res["late_early_ratio"] > 3


def test_chunked_logprobs_match_full_softmax():
    import torch

    from dpt.parity.logprob_parity import chunked_token_logprobs

    torch.manual_seed(0)
    head = torch.nn.Linear(16, 50, bias=False)
    h = torch.randn(37, 16)
    tgt = torch.randint(0, 50, (37,))
    full = torch.log_softmax(head(h), -1).gather(-1, tgt[:, None])[:, 0]
    assert torch.allclose(chunked_token_logprobs(h, head, tgt, chunk=8), full, atol=1e-6)
