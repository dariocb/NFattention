#!/usr/bin/env python
"""Run the PDF's original 256--4096 efficiency grid in an isolated output."""
from pathlib import Path
import runpy, sys
if "--lengths" not in sys.argv:
    sys.argv += ["--lengths", "256,512,1024,2048,4096"]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "05_efficiency" / "main.py"), run_name="__main__")
