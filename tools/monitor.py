"""Root forwarder for real-time benchmark monitor."""
import os
import runpy
import sys

_TARGET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "v1_drawing_vla", "handvsai", "tools", "monitor.py")
)
if not os.path.exists(_TARGET):
    sys.exit(f"[error] Cannot find monitor.py at {_TARGET}")

sys.argv[0] = _TARGET
runpy.run_path(_TARGET, run_name="__main__")
