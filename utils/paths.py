"""
MALA路径工具
============

集中处理 `E:/...` 风格路径和环境变量映射。
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_data_path(path: str | None, env_var: str = "MALA_DATA_ROOT") -> str | None:
    if path is None:
        return None

    normalized = path.replace("\\", "/")
    direct = Path(path)
    if direct.exists():
        return str(direct)

    repo_relative = Path(__file__).resolve().parents[1] / normalized
    if repo_relative.exists():
        return str(repo_relative)

    mapped_root = os.environ.get(env_var)
    if mapped_root and normalized.startswith("E:/"):
        mapped = Path(mapped_root) / normalized[3:]
        if mapped.exists():
            return str(mapped)

    return path


def env_or_default(env_var: str, default: str | None = None) -> str | None:
    value = os.environ.get(env_var)
    return value if value else default


def normalize_path_text(path: str | None) -> str:
    return "" if path is None else path.replace("\\", "/")
