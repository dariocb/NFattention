#!/usr/bin/env python
"""Reconstruct Table 7: 20NG KL sensitivity, including the PDF's exact grid."""
from pathlib import Path
import argparse, sys
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rebuttal._shared.zi import add_zi_arguments, run_zi_grid
p=argparse.ArgumentParser(description=__doc__); add_zi_arguments(p); p.add_argument("--variant-index", type=int, default=None); a=p.parse_args()
variants=[{"name":f"lambda_{x:g}","model_name":"fska","density_mode":"learned_flow","feature_map":"elu_plus_one","qk_mode":"identity","kl_weight":x} for x in (0.,1e-4,1e-3,1e-2,1e-1)]
if a.variant_index is not None:
 if not 0 <= a.variant_index < len(variants): p.error(f"--variant-index must be in [0, {len(variants)-1}]")
 variants=[variants[a.variant_index]]
raise SystemExit(run_zi_grid(a, "zi_kl_20ng", variants))
