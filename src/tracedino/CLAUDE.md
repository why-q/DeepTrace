# TraceDINO

## 1. 概述

本子项目 TraceDINO（特征增强的深度伪造溯源）是一个基于对比学习的框架，通过微调 DINOv3 特征用于片段级深度伪造视频溯源。

## 2. 项目结构

```
src/tracedino/
├── config.py                  # 配置管理
├── preprocess.py              # 数据预处理脚本
├── train.py                   # 训练脚本
├── evaluate.py                # 评估脚本
├── models/                    # 模型实现
│   ├── backbone.py            # DINOv3主干网络
│   ├── adapter.py             # 特征适配模块
│   └── tracedino.py           # 完整TraceDINO模型
├── losses/                    # 损失函数
│   ├── supcon.py              # 监督对比损失
│   ├── koleo.py               # KoLeo熵正则化
│   └── combined.py            # 组合损失
├── dataset/                   # 数据管道
│   ├── metadata.py            # CSV元数据解析
│   ├── frame_extractor.py     # 视频帧提取
│   ├── augmentations.py       # 数据增强
│   ├── dataset.py             # Dataset类
│   ├── preprocessed_dataset.py # 预处理数据集类
│   └── datamodule.py          # DataModule
├── utils/                     # 工具函数
└── README.md                  # 完整文档（包含方法概述、预处理说明等）
```

## 3. 核心命令

- **数据预处理**：`python -m src.tracedino.preprocess --split train --batch-size 32 --io-workers 8`
- **单 GPU 训练**：`python -m src.tracedino.train`
- **多 GPU 训练**：`torchrun --nproc_per_node=4 --standalone -m src.tracedino.train`
- **验证**：`python -m src.tracedino.evaluate --checkpoint checkpoints/tracedino/best_model.pth --split test`
- 如果该子项目进行了更新，请同步更新本文档的「项目结构」部分

## 4. 渐进式披露

- 如果需要全局快速了解本项目的方法细节和使用方法，你需要阅读 `src/tracedino/README.md`
  - 包含：环境安装、快速开始、数据预处理、训练评估、方法概述、数据集格式等完整信息