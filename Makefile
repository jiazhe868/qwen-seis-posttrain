# Convenience targets. Every target is a thin wrapper around a documented command (see docs/design_27b_line.md §5).
PY ?= python
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: help test lint plan plan-local plan-cluster data-smoke dry-run-sft dry-run-rl check-env graders

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

test:            ## all CPU unit tests (1430, ~45 s)
	$(PY) -m pytest -q

lint:            ## flake8 (120 columns)
	$(PY) -m flake8 src tests

plan:            ## memory / FLOPs / communication budget for the cluster profile
	$(PY) -m dpt.planning.budget --profile cluster_2x8xa100_80g

plan-local:      ## same for the 4×A100-40GB workstation
	$(PY) -m dpt.planning.budget --profile local_4xa100_40g

plan-cluster:        ## the Qwen3.5-9B line on the cluster (8 nodes × 3 × A100-40GB)
	$(PY) -m dpt.planning.budget --profile slurm_3xa100_40g --model-line qwen35_9b

check-env:       ## environment report (GPUs, driver/CUDA, package versions)
	$(PY) -m dpt.launch.check_env

data-smoke:      ## toy data pipeline with offline fakes → data/smoke
	bash scripts/01_data.sh --smoke

graders:         ## adversarial grader suite (must be 100%)
	$(PY) -m dpt.rewards.validate

dry-run-sft:     ## render the 27B SFT command without running it
	DPT_HW_PROFILE=cluster_2x8xa100_80g $(PY) -m dpt.launch.sft --dry-run --allow-missing-manifest

dry-run-rl:      ## render the 27B RL command (and run all pre-flight checks) without running it
	DPT_HW_PROFILE=cluster_2x8xa100_80g $(PY) -m dpt.launch.rl --dry-run --allow-missing-manifest
