#!/usr/bin/env python
"""SST-5 accuracy/macro-F1 diagnostic grid."""

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
        {"name": "transformer", "model_name": "transformer"},
        {"name": "performer", "model_name": "performer"},
        {"name": "rka", "model_name": "rka"},
        {
            "name": "fska",
            "model_name": "fska",
            "density_mode": "learned_flow",
            "feature_map": "elu_plus_one",
            "qk_mode": "identity",
            "kl_weight": 1e-3,
        },
    ]
    return run_sst_grid(args, "sst5_diagnostics", variants)


if __name__ == "__main__":
    raise SystemExit(main())
