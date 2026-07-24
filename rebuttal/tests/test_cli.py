from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "folder",
    [
        "01_sst5_diagnostics",
        "02_fixed_density",
        "03_kl_sensitivity",
        "04_elu_gap",
        "05_efficiency",
        "06_listops",
    ],
)
def test_cli_help(folder: str):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "rebuttal" / folder / "main.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

