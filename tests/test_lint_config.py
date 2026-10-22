"""Configuración de pre-commit, finales de línea y editor."""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _precommit_hooks():
    cfg = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    return {h["id"]: h for repo in cfg["repos"] for h in repo["hooks"]}


def test_ruff_hooks_only_check():
    hooks = _precommit_hooks()
    assert "--no-fix" in hooks["ruff"]["args"]
    assert "--check" in hooks["ruff-format"]["args"]


def test_fixers_are_manual_stage():
    hooks = _precommit_hooks()
    for hook_id in ("end-of-file-fixer", "trailing-whitespace", "mixed-line-ending"):
        assert hooks[hook_id].get("stages") == ["manual"], hook_id


def test_gitattributes_enforces_lf_and_binaries():
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert re.search(r"^\*\s+text=auto\s+eol=lf", text, re.M)
    for ext in ("wav", "tflite", "png"):
        assert re.search(rf"^\*\.{ext}\s+binary", text, re.M)


def test_editorconfig_lf():
    text = (ROOT / ".editorconfig").read_text(encoding="utf-8")
    assert "end_of_line = lf" in text


def test_vscode_settings_lf_and_ruff():
    raw = (ROOT / ".vscode" / "settings.json").read_text(encoding="utf-8")
    raw = "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("//"))
    settings = json.loads(raw)
    assert settings["files.eol"] == "\n"
    assert settings["[python]"]["editor.defaultFormatter"] == "charliermarsh.ruff"
