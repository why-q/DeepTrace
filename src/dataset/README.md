# Vorpus Dataset Processor

将 Vorpus 数据集的长视频根据 Ground Truth (GT) 帧范围截取为最长 5 分钟的片段，以 1 FPS 频率抽帧保存为图片序列，并更新 CSV 元数据。

## 快速开始

```bash
# 1. 进入项目目录并激活虚拟环境
cd ~/pychen/DeepTrace
source src/.venv/bin/activate

# 2. 运行处理脚本（从 src 目录）
cd src

# 处理所有视频（GT + non-GT），打包为 tar 文件节省磁盘空间
python -m dataset.vorpus_processor all \
    --input-dir /Volumes/AIGO/vorpus \
    --output-dir /Volumes/AIGO/vorpus_frames \
    --csv-dir ../asset/dataset \
    --output-csv-dir ../asset/dataset/new \
    --max-duration 300 \
    --output-fps 1 \
    --workers 8 \
    --pack-tar
```

## 子命令

### `all` - 处理所有视频（推荐）

处理 GT 视频和 non-GT 视频，生成统一的 `vorpus.csv` 索引文件。

```bash
python -m dataset.vorpus_processor all \
    --input-dir /Volumes/AIGO/vorpus \
    --output-dir /Volumes/AIGO/vorpus_frames \
    --csv-dir ../asset/dataset \
    --output-csv-dir ../asset/dataset/new \
    --max-duration 300 \
    --output-fps 1 \
    --workers 2 \
    --gt-limit 10 \
    --non-gt-limit 20 \
    --pack-tar
```

### `gt` - 仅处理 GT 视频

只处理有 Ground Truth 标注的视频。

```bash
python -m dataset.vorpus_processor gt \
    --input-dir /Volumes/AIGO/vorpus \
    --output-dir /Volumes/AIGO/vorpus_frames \
    --csv-dir ../asset/dataset \
    --output-csv-dir ../asset/dataset/new \
    --max-duration 300 \
    --output-fps 1 \
    --workers 2 \
    --limit 10 \
    --pack-tar
```

### `non-gt` - 增量处理非 GT 视频（新增）

只处理非 GT 视频，用于增加数据集片段数量。支持智能采样策略和断点续传。

```bash
# 默认过滤 10 分钟以下的视频，目标 5000 个片段
python -m dataset.vorpus_processor non-gt \
    --input-dir /Volumes/AIGO/vorpus \
    --output-dir /Volumes/AIGO/vorpus_frames \
    --csv-dir ../asset/dataset \
    --output-csv-dir ../asset/dataset/new \
    --target-total 5000 \
    --max-duration 300 \
    --output-fps 1 \
    --workers 8 \
    --pack-tar

# 自定义最小时长（15 分钟）
python -m dataset.vorpus_processor non-gt \
    --input-dir /Volumes/AIGO/vorpus \
    --output-dir /Volumes/AIGO/vorpus_frames \
    --csv-dir ../asset/dataset \
    --output-csv-dir ../asset/dataset/new \
    --target-total 5000 \
    --min-duration 900 \
    --pack-tar
```

**特点：**
- ✅ **智能过滤**：只处理长视频（默认 ≥ 10 分钟）
- ✅ **随机采样**：从视频中随机选择不重叠的片段，增加多样性
- ✅ **按时长排序**：优先处理最长的视频，充分利用长视频资源
- ✅ **智能分配**：根据目标数量自动计算每个视频的采样次数
- ✅ **断点续传**：支持中断后继续处理，不会重复已完成的视频

## 参数说明

### 通用参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-dir` | 是 | - | 原始视频目录 (0000.mp4 ~ 1573.mp4) |
| `--output-dir` | 是 | - | 输出帧序列目录 |
| `--csv-dir` | 是 | - | 原始 CSV 目录 (包含 all.csv, train.csv 等) |
| `--output-csv-dir` | 是 | - | 输出 CSV 目录 |
| `--max-duration` | 否 | 300 | 最大片段时长（秒） |
| `--output-fps` | 否 | 1.0 | 抽帧频率（每秒抽取帧数） |
| `--workers` | 否 | 8 | 并发线程数 |
| `--pack-tar` | 否 | False | 将帧序列打包为 tar 文件（节省磁盘空间） |
| `--no-resume` | 否 | False | 禁用断点续传，重新处理所有视频 |

### `all` 命令专用参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--gt-limit` | 否 | None | 限制处理的 GT origin 数量（用于测试） |
| `--non-gt-limit` | 否 | None | 限制处理的 non-GT 视频数量（用于测试） |

### `gt` 命令专用参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--limit` | 否 | None | 限制处理的 origin 数量（用于测试） |

### `non-gt` 命令专用参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--target-total` | 否 | 5000 | 目标片段总数 |
| `--min-duration` | 否 | 600 | 最小视频时长（秒），过滤短视频（默认 10 分钟） |

## 输入输出

**输入：**
- 视频：`/Volumes/AIGO/vorpus/0000.mp4` ~ `1573.mp4`（共 1574 个视频）
- CSV：`asset/dataset/{all,train,valid,test}.csv`

**输出（不使用 --pack-tar）：**
- 帧序列：`/Volumes/AIGO/vorpus_frames/{uuid}/0001.jpg, 0002.jpg, ...`

**输出（使用 --pack-tar）：**
- tar 文件：`/Volumes/AIGO/vorpus_frames/{uuid}.tar`（内含 0001.jpg, 0002.jpg, ...）

**CSV 文件：**
- `asset/dataset/new/all.csv` - GT 视频的详细元数据
- `asset/dataset/new/train.csv` - 训练集
- `asset/dataset/new/valid.csv` - 验证集
- `asset/dataset/new/test.csv` - 测试集
- `asset/dataset/new/vorpus.csv` - 所有片段的统一索引（GT + non-GT）

## 输出目录结构

**不使用 --pack-tar：**
```
output_dir/
├── {uuid_1}/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── {uuid_2}/
│   └── ...
└── ...
```

**使用 --pack-tar：**
```
output_dir/
├── {uuid_1}.tar
├── {uuid_2}.tar
└── ...
```

tar 文件内部结构：
```
{uuid}.tar
├── 0001.jpg
├── 0002.jpg
└── ...
```

## CSV 格式

### all.csv（GT 详细记录）

| 字段 | 说明 |
|------|------|
| `id` | 记录唯一 ID |
| `origin_id` | 帧序列目录名（UUID） |
| `gt_start_f` | GT 起始帧号（相对于裁剪片段） |
| `gt_end_f` | GT 结束帧号（相对于裁剪片段） |
| `gt_start_img` | GT 起始图片序号（从 1 开始） |
| `gt_end_img` | GT 结束图片序号（从 1 开始） |
| `frame_count` | 该片段的总帧数 |
| `output_fps` | 抽帧频率 |
| ... | 其他原始字段 |

### vorpus.csv（统一索引）

| 字段 | 说明 |
|------|------|
| `origin_id` | 帧序列目录名（UUID） |
| `frame_count` | 该片段的总帧数 |
| `output_fps` | 抽帧频率 |
| `label` | `gt` 或 `non_gt` |

## 核心算法

### GT 视频处理

1. 按 `origin_id` 分组所有记录
2. 对每个 origin，按 `gt_start_f` 排序后贪心分组（总跨度 ≤ 5 分钟则合并）
3. 对每个分组使用 FFmpeg 抽帧，生成 UUID 命名的目录
4. 更新该组所有记录的 `origin_id`、帧号和图片序号

### non-GT 视频处理（智能采样策略）

#### 传统方式（`all` 命令）
1. 找出不在 GT CSV 中的视频
2. 随机选择起始点，裁剪 5 分钟片段
3. 使用 FFmpeg 抽帧

#### 增量方式（`non-gt` 命令）
1. **过滤短视频**：只保留时长 ≥ `min-duration` 的视频（默认 10 分钟）
2. **按时长排序**：将长视频按时长降序排序（最长的在前）
3. **智能分配采样次数**：
   - 计算需要的片段数：`needed = target_total - current_count`
   - 计算基础采样次数：`base_clips = needed // 长视频数量`
   - 计算额外采样数：`extra_clips = needed % 长视频数量`
   - 所有长视频都采样 `base_clips` 次
   - 最长的 `extra_clips` 个视频额外多采样 1 次
4. **随机采样**：从每个视频中随机选择不重叠的片段起始位置
5. **并行处理**：使用多线程并行处理所有视频
6. **断点续传**：保存进度到 `.progress_non_gt/` 目录，支持中断后继续

**示例**：
- 当前有 3066 个片段，目标 5000 个，需要 1934 个
- 过滤后有 800 个长视频（≥ 10 分钟）
- 1934 ÷ 800 = 2 余 334
- 所有 800 个视频都采样 2 次 → 1600 个片段
- 最长的 334 个视频额外采样 1 次 → 334 个片段
- 总计：1934 个片段

### 抽帧策略

抽取每秒中间的帧，而不是第一帧：
- 1 FPS：抽取 0.5s, 1.5s, 2.5s, ... 时刻的帧
- 2 FPS：抽取 0.25s, 0.75s, 1.25s, 1.75s, ... 时刻的帧

## 断点续传

所有命令都支持断点续传功能：

- **GT 处理进度**：保存在 `output-csv-dir/.progress/`
- **非 GT 处理进度**：保存在 `output-csv-dir/.progress_non_gt/`

进度文件包括：
- `progress_origins.jsonl`：已完成的 origin/视频列表
- `progress_records.jsonl`：已处理的记录
- `.progress.lock`：文件锁（确保多线程安全）

中断后重新运行相同命令，会自动跳过已完成的部分。如需重新处理，使用 `--no-resume` 参数。

## 依赖

- polars
- opencv-python
- ffmpeg (系统安装)
- loguru
- tqdm
