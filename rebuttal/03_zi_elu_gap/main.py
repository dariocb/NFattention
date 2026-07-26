#!/usr/bin/env python
"""Reconstruct the PDF signed-RFF versus ELU+1 end-to-end ablation."""
from pathlib import Path
import argparse, sys
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rebuttal._shared.zi import add_zi_arguments, run_zi_grid
p=argparse.ArgumentParser(description=__doc__); add_zi_arguments(p); a=p.parse_args()
raise SystemExit(run_zi_grid(a, "zi_elu_gap", [
 {"name":"signed_raw_rff","model_name":"fska","density_mode":"learned_flow","feature_map":"raw_rff","qk_mode":"identity","kl_weight":1e-3},
 {"name":"elu_plus_one","model_name":"fska","density_mode":"learned_flow","feature_map":"elu_plus_one","qk_mode":"identity","kl_weight":1e-3},]))
