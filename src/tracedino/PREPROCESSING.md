# TraceDINO 数据预处理

## 背景

训练时 InsightFace 和 YOLOv8 检测模型在每个 DataLoader worker 中重复初始化，导致死锁。解决方案：预处理数据集，预先提取帧和检测结果。

## 数据结构

### 输入
- `query_v/{uuid}.mp4` - Query 视频 (30fps)
- `vorpus_f/{origin_id}/{frame:04d}.jpg` - Source 帧 (1 FPS 抽取)
- `gt_start_f/gt_end_f` - 原始帧号，需除以 fps 映射到 vorpus_f

### 输出 (~66GB)
```
preprocessed/{split}/{video_uuid}/
├── anchor_0.jpg, anchor_1.jpg    # 从 query video 提取
├── negative_00.jpg ~ negative_19.jpg  # 困难负样本 (20张)
└── meta.json                     # 检测框 + 元数据
```

### meta.json 格式
```json
{
  "video_id": "...",
  "origin_id": "1379",
  "anchors": [{"frame_idx": 20, "source_frame_no": 175}],
  "detections": {
    "175": {"human_bbox": [x1,y1,x2,y2], "face_bbox": [x1,y1,x2,y2]}
  }
}
```

## 关键代码

| 文件 | 说明 |
|------|------|
| `preprocess.py` | 预处理脚本 |
| `dataset/preprocessed_dataset.py` | 预处理数据集类 |
| `config.py` | 新增 `preprocessed_dir`, `use_preprocessed`, `n_presampled_negatives` |
| `dataset/datamodule.py` | 支持 `use_preprocessed` 参数 |

## 使用方法

```bash
# 1. 预处理 (约 2.5s/视频)
python -m src.tracedino.preprocess --split train
python -m src.tracedino.preprocess --split valid
python -m src.tracedino.preprocess --split test

# 2. 训练 (自动使用预处理数据)
python -m src.tracedino.train
```

## 帧号映射

vorpus_f 以 1 FPS 抽取，原始帧号需转换：
```python
vorpus_frame_no = original_frame_no // fps  # 如 5257 // 30 = 175
```

## 训练时数据流

1. Anchor: 从 `preprocessed/` 读取
2. Positives: 从 `vorpus_f/` 读取 source frame，根据 bbox 实时裁剪/模糊
3. Negatives: 从 `preprocessed/` 的 20 张中随机选 3 张
4. JPEG 压缩: 训练时实时应用
