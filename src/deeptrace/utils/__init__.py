"""
DeepTrace 工具模块

提供通用的工具函数和类，包括：
- 多进程并行处理工具
- 路径工具函数
"""

from .parallel import ParallelProcessor, parallel_process
from .path_utils import get_project_root, get_models_dir, get_asset_dir, get_result_dir

__all__ = [
    "ParallelProcessor",
    "parallel_process",
    "get_project_root",
    "get_models_dir",
    "get_asset_dir",
    "get_result_dir",
]

