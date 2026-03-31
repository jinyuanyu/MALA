"""兼容转发入口，实际实现位于 legacy/MAE_LaMa.py。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy.MAE_LaMa import *  # noqa: F401,F403
