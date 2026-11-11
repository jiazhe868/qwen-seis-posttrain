"""End-to-end offline run of the toy pipeline (plumbing, manifests, funnel)."""

from __future__ import annotations

from dpt.common.io import read_json, read_jsonl
from dpt.common.manifest import verify_manifest
from dpt.data.pipeline import main


def test_toy_pipeline_offline(tmp_path):
    out = tmp_path / "toy"
    rc = main(["--offline", "--stages", "ingest,quality,dedup,decontam,verify,difficulty,publish,sft,probe",
               f"data_pipeline.output_dir={out}"])
    assert rc == 0
    funnel = {r["stage"]: r for r in read_json(out / "funnel.json")}
    # every planted defect in examples/toy_data is caught by the stage meant to catch it
    assert funnel["ingest"]["reasons"] == {"license": 1}
    assert funnel["quality"]["removed"] == 1
    assert funnel["dedup"]["reasons"]["near_duplicate"] == 1
    assert funnel["decontam"]["reasons"] == {"decontam:ngram_minhash": 2, "decontam:semantic": 1,
                                             "decontam:answer_level": 1}
    assert funnel["verify"]["reasons"] == {"verify:suspect_label": 1, "verify:no_solver": 1}
    for i in range(1, 6):
        prev, cur = list(funnel.values())[i - 1], list(funnel.values())[i]
        assert prev["out"] == cur["in"]
    # published files are manifested with the full funnel and verify cleanly
    m = verify_manifest(out / "rl_main.jsonl")
    assert [s["stage"] for s in m.stage_stats][:6] == ["ingest", "quality", "dedup", "decontam", "verify",
                                                       "difficulty"]
    rows = list(read_jsonl(out / "rl_main.jsonl")) + list(read_jsonl(out / "rl_hard.jsonl"))
    assert len(rows) == funnel["difficulty"]["out"]
    queue = list(read_jsonl(out / "expert_queue.jsonl"))
    assert queue[0]["priority"] == "high" and queue[0]["reason"] == "suspect_label"
    rep = read_json(out / "reports" / "contamination_report.json")
    assert set(rep) == {"rl", "sft", "memorization_probes"} and rep["rl"]["canary"]["hits"] == 0
    assert rep["sft"]["teacher_verbatim"] == 1
    # no eval text leaks into the training-side outputs
    removed = (out / "removed" / "decontam.jsonl").read_text()
    assert "hypocenter to a station 120 km away" not in removed
    sft = read_json(out / "sft" / "sft_report.json")
    assert abs(sft["mixture"]["self_distill_token_share"] - 0.25) < 0.05


def test_dev_candidates_are_written_after_verify_and_read_by_publish(tmp_path):
    import pytest

    from dpt.data.rl_build import dev_candidates
    from dpt.data.schema import Problem
    out = tmp_path / "toy"
    ov = [f"data_pipeline.output_dir={out}", "data_pipeline.publish.dev_fraction=0.25"]
    assert main(["--offline", "--stages", "ingest,quality,dedup,decontam,verify", *ov]) == 0
    assert main(["--offline", "--dev-candidates", *ov]) == 0
    verified = [Problem.from_dict(r) for r in read_jsonl(out / "stages" / "04_verify.jsonl")]
    cand = {r["uid"] for r in read_jsonl(out / "rl_dev_candidates.jsonl")}
    assert cand and cand == dev_candidates(verified, fraction=0.25, seed=1)
    # (the toy bank is too small for a dev split after difficulty; rl_build tests the selection itself)
    pub = ["--offline", "--stages", "difficulty,publish", *ov, "data_pipeline.publish.dev_candidates=true"]
    assert main(pub) == 0
    (out / "rl_dev_candidates.jsonl").unlink()
    with pytest.raises(FileNotFoundError):                    # publish reads them: never a split without them
        main(pub)


def test_resume_single_stage(tmp_path):
    out = tmp_path / "toy"
    main(["--offline", "--stages", "ingest,quality,dedup", f"data_pipeline.output_dir={out}"])
    first = (out / "stages" / "02_dedup.jsonl").read_text()
    main(["--offline", "--stages", "dedup", f"data_pipeline.output_dir={out}"])
    assert (out / "stages" / "02_dedup.jsonl").read_text() == first


def test_sft_samples_keep_only_verified_non_dev_problems(tmp_path):
    from dpt.common.io import write_jsonl
    from dpt.data.pipeline import filter_by_problem
    from dpt.data.schema import SFTSample

    def sample(i, problem):
        msgs = [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
        src = {"problem_uid": problem} if problem else {}
        return SFTSample.from_dict({"messages": msgs, "subset": "domain_reasoning", "source": src})

    samples = [sample(0, "p-kept"), sample(1, "p-dropped"), sample(2, "p-dev"), sample(3, None)]
    write_jsonl(tmp_path / "verify.jsonl", [{"uid": "p-kept"}, {"uid": "p-dev"}])
    write_jsonl(tmp_path / "rl_dev.jsonl", [{"extra_info": {"uid": "p-dev"}}])
    kept, stats = filter_by_problem(samples, str(tmp_path / "verify.jsonl"), [str(tmp_path / "rl_dev.jsonl")])
    assert [s.source.get("problem_uid") for s in kept] == ["p-kept", None]
    assert stats == {"kept": 2, "problem_not_verified": 1, "rl_dev_problem": 1}
    assert filter_by_problem(samples, None, [])[0] == samples
