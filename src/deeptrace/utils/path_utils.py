"""路径工具函数。

提供项目路径相关的工具函数。
"""

from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录。
    
    Returns:
        项目根目录的 Path 对象
    """
    # 当前文件路径: src/deeptrace/utils/path_utils.py
    # 向上查找，直到找到包含 models 目录的项目根目录
    current = Path(__file__).resolve().parent  # utils 目录
    
    # 向上遍历最多 10 层
    for _ in range(10):
        # 检查是否包含项目标识文件/目录
        if (current / "models").exists() and (current / "src").exists():
            return current
        if (current / "pyproject.toml").exists() and (current / "models").exists():
            return current
        if (current / ".git").exists() and (current / "models").exists():
            return current
        
        parent = current.parent
        if parent == current:  # 到达文件系统根目录
            break
        current = parent
    
    # 如果找不到，使用固定的相对路径
    # (utils -> deeptrace -> src -> project_root)
    return Path(__file__).resolve().parent.parent.parent.parent


def get_models_dir() -> Path:
    """获取模型目录。
    
    Returns:
        模型目录的 Path 对象
    """
    return get_project_root() / "models"


def get_asset_dir() -> Path:
    """获取资源目录。
    
    Returns:
        资源目录的 Path 对象
    """
    return get_project_root() / "asset"


def get_result_dir() -> Path:
    """获取结果目录。
    
    Returns:
        结果目录的 Path 对象
    """
    return get_project_root() / "result"

