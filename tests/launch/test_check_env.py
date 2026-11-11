"""dpt.launch.check_env: installed-but-broken packages are reported as such, not as missing."""

from __future__ import annotations

import importlib.metadata
import sys

from dpt.launch import check_env


def test_missing_package_is_none_and_not_an_import_error():
    check_env._IMPORT_ERRORS.clear()
    assert check_env._version("dpt_no_such_module_xyz") is None
    assert "dpt_no_such_module_xyz" not in check_env._IMPORT_ERRORS


def test_broken_import_keeps_version_and_records_error(tmp_path, monkeypatch):
    (tmp_path / "dpt_broken_mod.py").write_text("raise RuntimeError('cuDNN 9.12.0 or later required')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    real = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda d: "2.20.2+cu12" if d == "dpt_broken_mod" else real(d))
    check_env._IMPORT_ERRORS.clear()
    assert check_env._version("dpt_broken_mod") == "2.20.2"
    assert "cuDNN" in check_env._IMPORT_ERRORS["dpt_broken_mod"]
    sys.modules.pop("dpt_broken_mod", None)
