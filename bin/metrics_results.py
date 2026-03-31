"""兼容转发入口，实际实现位于 analysis/metrics.py。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from analysis.metrics import parse_args_and_run

    parse_args_and_run()
else:
    from analysis.metrics import *  # noqa: F401,F403
