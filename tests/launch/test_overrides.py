"""Hydra override rendering (formatting bugs here only show up at launch time on the cluster)."""

from __future__ import annotations

from dpt.launch.overrides import render_value, to_overrides


def test_scalars():
    assert render_value(True) == "true"
    assert render_value(None) == "null"
    assert render_value(1e-6) == "1e-06"
    assert render_value(24576) == "24576"
    assert render_value("vllm") == "vllm"


def test_strings_needing_quotes():
    assert render_value("expandable_segments:True") == "'expandable_segments:True'"
    assert render_value("a b") == "'a b'"
    assert render_value("true") == "'true'"   # the *string* "true", not the boolean


def test_list_and_dict():
    assert render_value(["console", "wandb"]) == "[console,wandb]"
    assert render_value({"method": "mtp", "num_speculative_tokens": 1}) == "{method:mtp,num_speculative_tokens:1}"


def test_to_overrides_sorted_and_plus_prefix_kept():
    ov = to_overrides({"b.x": 1, "+a.y": "z", "c": [1, 2]})
    assert ov == ["+a.y=z", "b.x=1", "c=[1,2]"]


def test_unset_dropped_and_groups_first():
    ov = to_overrides({"optim.lr": 1e-6, "engine": "megatron", "engine.tp": "__unset__"})
    assert ov == ["engine=megatron", "optim.lr=1e-06"]


def test_numeric_looking_strings_stay_strings():
    """Smoke-run regression: Ray rejects runtime_env env_vars that Hydra parsed as int."""
    assert render_value("1") == "'1'"
    assert render_value(1) == "1"
    assert render_value("0.5") == "'0.5'"
    assert render_value("1e-3") == "'1e-3'"
