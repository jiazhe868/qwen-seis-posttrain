"""Parsing teacher replies: LaTeX written unescaped into JSON strings must not lose the reply (half of T3's first
drafting rounds were lost that way on the cluster)."""

from __future__ import annotations

import pytest

from dpt.data.gen.teacher import last_json


def test_unescaped_latex_is_repaired_without_eating_commands():
    reply = ('plan </think>\n{"items": [{"question": "How do \\sigma_n and \\Delta\\tau control slip? Use '
             '\\frac{a}{b}, \\beta and \\nabla.", "rubric": []}]}')
    q = last_json(reply, "items")[0]["question"]
    assert q == "How do \\sigma_n and \\Delta\\tau control slip? Use \\frac{a}{b}, \\beta and \\nabla."


def test_valid_json_is_read_exactly_as_before():
    reply = '</think>{"items": [{"question": "Tab\\tand \\"quote\\" and \\\\alpha \\u00e9\\n"}]}'
    assert last_json(reply, "items")[0]["question"] == 'Tab\tand "quote" and \\alpha é\n'


def test_no_json_still_raises():
    with pytest.raises(ValueError):
        last_json("</think>no json here", "items")
