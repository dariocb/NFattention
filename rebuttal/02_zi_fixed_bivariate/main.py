#!/usr/bin/env python
"""Reconstruct the PDF fixed-bivariate Gaussian ablation on an explicit dataset."""
from pathlib import Path
import argparse, sys
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rebuttal._shared.zi import add_zi_arguments, run_zi_grid
p=argparse.ArgumentParser(description=__doc__); add_zi_arguments(p); a=p.parse_args()
raise SystemExit(run_zi_grid(a, "zi_fixed_bivariate", [
 {"name":"fixed_bivariate_gaussian","model_name":"fska","density_mode":"fixed_bivariate","feature_map":"elu_plus_one","qk_mode":"identity","kl_weight":0.0},
 {"name":"full_fska","model_name":"fska","density_mode":"learned_flow","feature_map":"elu_plus_one","qk_mode":"identity","kl_weight":1e-3},]))
