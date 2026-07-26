#!/usr/bin/env python
"""Reconstruct the PDF 20NG spectral-capacity / Q-K removal grid."""
from pathlib import Path
import argparse, sys
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rebuttal._shared.zi import add_zi_arguments, run_zi_grid
p=argparse.ArgumentParser(description=__doc__); add_zi_arguments(p); a=p.parse_args()
variants=[]
for m in (32,64,96,128):
 for qk in ("learned", "identity"):
  variants.append({"name":f"m{m}_{qk}_qk","model_name":"fska","num_spectral_pairs":m,"num_flows":2,"density_mode":"learned_flow","feature_map":"elu_plus_one","qk_mode":qk,"kl_weight":1e-3})
for flows in (2,3,5,7,9):
 for qk in ("learned", "identity"):
  variants.append({"name":f"flows{flows}_{qk}_qk","model_name":"fska","num_spectral_pairs":32,"num_flows":flows,"density_mode":"learned_flow","feature_map":"elu_plus_one","qk_mode":qk,"kl_weight":1e-3})
raise SystemExit(run_zi_grid(a, "zi_capacity_qk", variants))
