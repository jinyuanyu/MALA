"""
MALA项目数据加载模块
====================

导出所有数据集类供外部调用。

作者: MALA Team
日期: 2024
"""

from .dataset import Datasets, Datasets_inference

__all__ = [
    'Datasets',
    'Datasets_inference'
]
