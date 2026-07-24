#!/usr/bin/env python
"""SST-5 sensitivity study for the FSKA KL coefficient."""

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
    # Extend above the paper default (1e-3) on a log scale to distinguish a
    # plateau from over-regularisation.
    weights = [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    variants = [
        {
            "name": f"lambda_{weight:g}",
            "model_name": "fska",
            "density_mode": "learned_flow",
            "feature_map": "elu_plus_one",
            "qk_mode": "identity",
            "kl_weight": weight,
        }
        for weight in weights
    ]
    return run_sst_grid(args, "kl_sensitivity", variants)


if __name__ == "__main__":
    raise SystemExit(main())
