# TraceDINO

TraceDINO（特征增强的深度伪造溯源）是一个基于对比学习的框架，通过微调DINOv3特征用于片段级深度伪造视频溯源。

## 环境安装

使用 `uv` 环境管理工具：

```bash
# 核心依赖
uv add torch torchvision
uv add transformers safetensors

# 检测模型
uv add ultralytics  # YOLOv8
uv add insightface  # RetinaFace

# 数据处理
uv add opencv-python pandas numpy tqdm scikit-learn

# 可选
uv add tensorboard
```

## 快速开始

### 1. 配置

默认配置在 [config.py](config.py) 中定义，主要参数：

- `backbone_path`: DINOv3模型路径（默认 `pretrained/dinov3/dinov3-vitb16/`）
- `train_csv/valid_csv/test_csv`: 数据集CSV路径
- `query_video_dir`: Query视频目录（默认 `/datadrive2/pychen/deeptrace/query_v/`）
- `source_frame_dir`: 源视频帧目录（默认 `/datadrive2/pychen/deeptrace/vorpus_f/`）
- `batch_size`: 批量大小（默认256）
- `num_epochs`: 训练轮数（默认50）

### 2. 训练

```bash
cd /home/qid/pychen/DeepTrace
python -m src.tracedino.train
```

训练过程中会：
- 每个epoch在验证集上评估
- 自动保存最佳模型到 `checkpoints/tracedino/best_model.pth`
- 定期保存检查点

### 3. 评估

```bash
python -m src.tracedino.evaluate --checkpoint checkpoints/tracedino/best_model.pth --split test
```

评估指标包括：
- **Triplet Accuracy**: 锚点-正样本距离 < 锚点-困难负样本距离的比例
- **Recall@K**: Top-K检索召回率（K=1,5,10）
- **Separation Margin**: 正负样本相似度分离度
- **AUC-ROC**: 分类能力

## 方法概述

### 模型架构

1. **DINOv3主干（冻结）**: 提取多层Patch Tokens（层3、6、9、12）
2. **特征适配模块（可训练）**:
   - 多层特征拼接: [B, N, 4×768]
   - 线性融合: [B, N, 768]
   - GeM池化: [B, 768]
   - MLP投影: [B, 512]

### 对比学习

每个样本包含：
- **1个锚点**: 深度伪造视频帧
- **3个正样本**:
  1. 原始源帧
  2. 裁剪变体（人体检测+随机裁剪）
  3. 人脸模糊变体（人脸检测+高斯模糊）
- **3个困难负样本**: 同一源视频不同时间段帧（15秒安全半径外）
- **批内负样本**: 同一batch中的其他样本

### 损失函数

L_total = L_SupCon + λ * L_KoLeo

- **L_SupCon**: 监督对比损失（τ=0.05）
- **L_KoLeo**: KoLeo熵正则化（λ=30）

## 数据集

### 格式要求

**CSV文件** (`train.csv`, `valid.csv`, `test.csv`):
- `id`: Query视频UUID
- `origin_id`: 源视频ID（4位数字）
- `gt_start_f/gt_end_f`: 源视频帧范围
- `fps`: 帧率
- `frames`: 总帧数

**数据目录**:
- Query视频: `{query_video_dir}/{id}.mp4`
- 源视频帧: `{source_frame_dir}/{origin_id}/{frame_no:04d}.jpg`

## 训练参数优化

推荐配置：
- **学习率**: 3e-4（余弦退火至1e-5）
- **批量大小**: 256（多GPU分布式训练）
- **温度参数**: 0.05
- **KoLeo权重**: 30
- **训练轮数**: 50

## 引用

如果使用本代码，请引用相关论文。

## 许可证

本项目遵循MIT许可证。
