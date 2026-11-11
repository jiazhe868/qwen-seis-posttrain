"""Tests for the partial-rollout buffer (report §单步时间预算: suspend > 16K, resume next step)."""

from __future__ import annotations

import pytest

from dpt.rl.partial_rollout import PartialRolloutBuffer, PendingGroup, SampleState


def _group(buf, gid, lengths_eos, version=0, step=0):
    samples = []
    for i, (n, eos) in enumerate(lengths_eos):
        s = SampleState(sample_id=f"{gid}-{i}")
        s.append(list(range(n)), [-0.1] * n, version)
        buf.finalize_chunk(s, hit_eos=eos)
        samples.append(s)
    return PendingGroup(group_id=gid, uid=f"u-{gid}", task="t2_code", prompt_ids=[1, 2, 3],
                        samples=samples, first_step=step)


def test_finished_group_is_ready_immediately():
    buf = PartialRolloutBuffer(chunk_tokens=16, max_response=24)
    buf.admit(_group(buf, "g", [(5, True), (7, True)]))
    ready = buf.pop_ready()
    assert [g.group_id for g in ready] == ["g"] and not buf.pending


def test_suspended_member_holds_back_the_whole_group_then_resumes():
    buf = PartialRolloutBuffer(chunk_tokens=16, max_response=24)
    buf.admit(_group(buf, "g", [(5, True), (16, False)], version=3))
    assert buf.pop_ready() == []
    reqs = buf.resume_requests()
    assert len(reqs) == 1 and reqs[0].max_new_tokens == 8                # min(chunk, 24 - 16)
    assert reqs[0].input_ids == [1, 2, 3] + list(range(16))
    buf.record_continuation("g", reqs[0].sample_id, [9, 9, 9], [-0.2] * 3, version=4, hit_eos=True)
    (g,) = buf.pop_ready()
    s = g.samples[1]
    assert s.segments == [[3, 16], [4, 3]] and s.token_versions()[-1] == 4
    assert PartialRolloutBuffer.off_policy_token_fraction([g], current_version=4) == pytest.approx(21 / 24)


def test_hitting_the_hard_cap_marks_truncated():
    buf = PartialRolloutBuffer(chunk_tokens=16, max_response=24)
    buf.admit(_group(buf, "g", [(16, False)]))
    req = buf.resume_requests()[0]
    buf.record_continuation("g", req.sample_id, list(range(8)), [-0.1] * 8, version=1, hit_eos=False)
    (g,) = buf.pop_ready()
    assert g.samples[0].finished and g.samples[0].truncated


def test_stale_groups_are_evicted():
    buf = PartialRolloutBuffer(chunk_tokens=16, max_response=48, max_staleness=1)
    buf.admit(_group(buf, "g", [(16, False)], version=0))
    assert buf.evict_stale(current_version=1) == []
    ev = buf.evict_stale(current_version=2)
    assert [g.group_id for g in ev] == ["g"] and buf.stats()["partial/evicted_total"] == 1


def test_state_round_trip():
    buf = PartialRolloutBuffer(chunk_tokens=16, max_response=24)
    buf.admit(_group(buf, "a", [(16, False), (3, True)], version=2))
    buf.admit(_group(buf, "b", [(4, True)], version=2))
    st = buf.state_dict()
    buf2 = PartialRolloutBuffer()
    buf2.load_state_dict(st)
    assert buf2.state_dict() == st
    assert buf2.resume_requests() == buf.resume_requests()


def test_duplicate_group_rejected_and_bad_append():
    buf = PartialRolloutBuffer(chunk_tokens=16, max_response=24)
    g = _group(buf, "g", [(16, False)])
    buf.admit(g)
    with pytest.raises(ValueError):
        buf.admit(g)
    with pytest.raises(ValueError):
        SampleState("x").append([1, 2], [0.0], 0)
