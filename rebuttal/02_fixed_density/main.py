#!/usr/bin/env python
"""Paired fixed-versus-learned bivariate spectral-density experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rebuttal._shared.grid import add_common_arguments, run_sst_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    variants = [
        {
            "name": "fixed_bivariate",
            "model_name": "fska",
            "density_mode": "fixed_bivariate",
            "feature_map": "elu_plus_one",
            "qk_mode": "identity",
            "kl_weight": 0.0,
        },
        {
            "name": "learned_flow",
            "model_name": "fska",
            "density_mode": "learned_flow",
            "feature_map": "elu_plus_one",
            "qk_mode": "identity",
            "kl_weight": 1e-3,
        },
    ]
    return run_sst_grid(args, "fixed_density", variants)


if __name__ == "__main__":
    raise SystemExit(main())
