from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    scripts = [
        "1_timelapse_temporal_independent.py",
        "2_timelapse_temporal_smoothness.py",
        "3_timelapse_temporal_active_time_constraint.py",
        "4_timelapse_temporal_baseline_reference.py",
    ]
    passthrough = sys.argv[1:]
    for script_name in scripts:
        command = [sys.executable, str(here / script_name), *passthrough]
        print("\n==>", " ".join(command))
        code = subprocess.call(command)
        if code != 0:
            print(f"Stopped on {script_name} with code {code}")
            return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
