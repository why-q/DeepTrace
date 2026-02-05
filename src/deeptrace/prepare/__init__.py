"""
数据准备工具 - 将 DeepTrace 数据集转换为 VCSL 格式

主要功能:
- 将 DeepTrace 数据集转换为 VCSL 输入格式
- 生成正负样本对（1:4比例）
- 生成片段对齐标注
"""

from .convert_to_vcsl import convert_deeptrace_to_vcsl

__all__ = ["convert_deeptrace_to_vcsl"]

