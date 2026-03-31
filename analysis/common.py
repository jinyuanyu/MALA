"""
分析公共常量与绘图设置
======================
"""

from __future__ import annotations

import matplotlib.pyplot as plt


ALGORITHM_NAMES = {
    "DINEOF": "数据插值（DINEOF）",
    "Proposed": "掩码自编码（MaskAE）",
    "Lama": "LaMA",
    "Spline": "样条插值（Spline）",
    "Nearest_Neighbor": "最近邻插值（NN）",
    "EMAE": "方法一（EMAE）",
    "MALA": "方法二（MALA）",
}


def configure_matplotlib_chinese() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def get_algorithm_display_name(algorithm_name: str) -> str:
    return ALGORITHM_NAMES.get(algorithm_name, algorithm_name)
