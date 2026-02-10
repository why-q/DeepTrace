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
- `train_csv/valid_csv/test_csv`: 数据集CSV路径（默认 `asset/dataset/new/`）
- `query_video_dir`: Query视频目录（默认 `/datadrive2/pychen/deeptrace/query_v/`）
- `source_frame_dir`: 源视频帧目录（默认 `/datadrive2/pychen/deeptrace/new_vorpus/`）
- `batch_size`: 批量大小（默认256）
- `num_epochs`: 训练轮数（默认50）

### 2. 数据预处理（推荐）

**为什么需要预处理？** 训练时检测模型（InsightFace、YOLOv8）在每个 DataLoader worker 中重复初始化会导致死锁。预处理可以预先提取帧和检测结果，提高训练效率。

```bash
cd ~/pychen/DeepTrace && source src/.venv/bin/activate

# 优化模式（默认，3-5倍速度提升）
python -m src.tracedino.preprocess --split all

# 自定义参数
python -m src.tracedino.preprocess --split train --batch-size 32 --io-workers 8

# 旧版串行模式（用于对比）
python -m src.tracedino.preprocess --split all --legacy
```

**性能对比**:
- **优化模式**: ~0.5-0.8s/视频（批量GPU推理 + 多线程I/O）
- **旧版模式**: ~2.5s/视频（串行处理）

**预处理输出** (~66GB):
```
preprocessed/{split}/{video_uuid}/
├── anchor_0.jpg, anchor_1.jpg           # 从query视频提取的锚点帧
├── negative_00.jpg ~ negative_19.jpg    # 预采样的困难负样本（20张）
└── meta.json                            # 人脸/人体检测框 + 元数据
```

**meta.json 示例**:
```json
{
  "video_id": "...",
  "origin_id": "204697cea05b43cc99a357041d80e246",
  "anchors": [{"frame_idx": 20, "source_frame_no": 130}],
  "detections": {
    "130": {"human_bbox": [x1,y1,x2,y2], "face_bbox": [x1,y1,x2,y2]}
  }
}
```

### 3. 训练

```bash
cd ~/pychen/DeepTrace && source src/.venv/bin/activate
python -m src.tracedino.train
```

训练过程中会：
- 每个epoch在验证集上评估
- 自动保存最佳模型到 `/datadrive2/pychen/deeptrace/checkpoints/tracedino/best_model.pth`
- 定期保存检查点

**训练时数据流**:
1. **Anchor**: 从 `preprocessed/` 读取预处理的锚点帧
2. **Positives**: 从 `new_vorpus/` 读取源帧，根据检测框实时裁剪/模糊
3. **Negatives**: 从 `preprocessed/` 的20张中随机选3张
4. **JPEG压缩**: 训练时实时应用数据增强

### 4. 评估

```bash
cd ~/pychen/DeepTrace && source src/.venv/bin/activate
python -m src.tracedino.evaluate --checkpoint /datadrive2/pychen/deeptrace/checkpoints/tracedino/best_model.pth --split test
```

评估指标包括：

**样本级指标**（在6个候选中评估）：
- **Triplet Accuracy**: 锚点-正样本距离 < 锚点-困难负样本距离的比例
- **Recall@K**: Top-K检索召回率（K=1,5,10）
- **Separation Margin**: 正负样本相似度分离度
- **AUC-ROC**: 分类能力

**批次级指标**（跨视频区分能力）：
- **Batch Triplet Accuracy**: 正样本相似度 > 批次内其他视频的比例
- **Batch Recall@K**: 在完整候选集（3正样本 + 3困难负样本 + B-1个其他anchor）中的Top-K召回率
- **Hard vs Batch Neg Gap**: 困难负样本与批次负样本的相似度差距（正值表示困难负样本确实更"困难"）

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

### CSV格式

**文件路径**: `asset/dataset/new/{train,valid,test}.csv`

**主要字段**:
- `id`: Query视频UUID（32字符哈希）
- `origin_id`: 源视频ID（32字符哈希）
- `gt_start_f/gt_end_f`: 源视频原始帧范围
- `gt_start_img/gt_end_img`: 源视频抽帧后的图像索引（1-based，直接使用）
- `fps`: 原始帧率
- `frames`: Query视频总帧数
- `frame_count`: 源视频抽帧后的总帧数
- `output_fps`: 抽帧频率（1.0 FPS）
- `category`: 伪造类别
- `celebrity`: 人物名称
- `v_no/scene_no/scene_sub_no`: 场景标识
- `face_no`: 人脸编号（可选）
- `method`: 伪造方法（可选）

### 数据目录结构

- **Query视频**: `{query_video_dir}/{id}.mp4`
- **源视频帧**: `{source_frame_dir}/{origin_id}/{frame_no:04d}.jpg`
  - 帧文件名为4位数字（0001.jpg ~ 0300.jpg）
  - 帧以1 FPS抽取，使用 `gt_start_img`/`gt_end_img` 索引

**帧号映射**（新数据集）:
```python
# 直接使用预计算的图像索引
source_img_idx = meta.gt_start_img  # 如 130

# 旧方式已废弃：
# vorpus_frame_no = original_frame_no // fps
```

## 训练参数优化

推荐配置：
- **学习率**: 3e-4（余弦退火至1e-5）
- **批量大小**: 256（多GPU分布式训练）
- **温度参数**: 0.05
- **KoLeo权重**: 30
- **训练轮数**: 50

## 相关文件

| 文件 | 说明 |
|------|------|
| `preprocess.py` | 数据预处理脚本 |
| `dataset/preprocessed_dataset.py` | 预处理数据集类 |
| `dataset/dataset.py` | 原始数据集类（不使用预处理） |
| `config.py` | 配置管理（`use_preprocessed`, `preprocessed_dir`） |
| `dataset/datamodule.py` | PyTorch Lightning DataModule |

## 引用

如果使用本代码，请引用相关论文。

## 许可证

本项目遵循MIT许可证。
