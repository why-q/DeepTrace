# Vorpus 数据集视频截取与重组计划

## Context

当前 vorpus 数据集中的视频时长过长，不利于后续处理。需要根据元数据中的 Ground Truth 帧范围，将长视频截取为最长 5 分钟的片段，同时更新元数据中的帧号信息。

**数据分析结果：**
- 共 536 个唯一的 origin 视频，1574 个视频文件
- 其中 76 个 origin 需要拆分（GT 间隔超过 5 分钟）
- 最多的一个 origin 需要拆分成 15 个片段
- 同一 origin 可能有多条记录共享相同的 GT 范围

**运行环境：**
- CPU: Intel N150 (低功耗处理器)
- 内存: 32 GB
- 存储: E 盘为机械硬盘（HDD）

## Git 分支

在开始实现前，创建新分支：
```bash
git checkout -b dataset
```

所有代码修改在 `dataset` 分支上进行。

## 输入/输出

| 项目 | 路径 |
|------|------|
| 原始视频 | `E:\dataset\videos\vorpus` (0000.mp4 ~ 1573.mp4) |
| 输出视频 | `E:\dataset\videos\vorpus_new` |
| 原始 CSV | `asset/dataset/{all,train,valid,test}.csv` |
| 输出 CSV | `asset/dataset/new/{all,train,valid,test}.csv` |

## 实现方案

### 1. 核心算法：GT 分组与视频截取

```
对于每个原视频 origin_id:
  1. 收集所有引用该视频的查询记录
  2. 按 gt_start_f 排序
  3. 贪心分组：如果相邻 GT 的 end 到下一个 start 间隔 <= max_duration_frames，合并到同一组
  4. 对每个组：
     a. 计算包含所有 GT 的最小帧范围 [min_start, max_end]
     b. 计算需要的总帧数 = max(max_end - min_start, max_duration_frames)
     c. 如果原视频足够长，随机选择一个起始点使得所有 GT 都在截取范围内
     d. 截取视频，生成新的 UUID 哈希命名
     e. 更新该组所有记录的 origin_id 和 gt_start_f/gt_end_f（相对于新视频起始帧）
```

### 2. 代码结构

```
src/dataset/
├── __init__.py
├── vorpus_processor.py    # 主处理逻辑
└── utils.py               # 工具函数（视频截取、哈希生成等）
```

### 3. 关键函数设计

**vorpus_processor.py:**

```python
def process_vorpus(
    input_dir: Path,
    output_dir: Path,
    csv_dir: Path,
    output_csv_dir: Path,
    max_duration_sec: int = 300
) -> None:
    """主入口函数"""

def group_records_by_origin(df: pl.DataFrame) -> dict[int, pl.DataFrame]:
    """按 origin_id 分组"""

def cluster_gt_ranges(
    records: pl.DataFrame,
    max_gap_frames: int
) -> list[list[int]]:
    """
    将记录按 GT 间隔分成簇
    返回：每个簇包含的记录索引列表
    """

def calculate_clip_range(
    gt_min: int,
    gt_max: int,
    max_frames: int,
    total_frames: int
) -> tuple[int, int]:
    """
    计算截取的帧范围 [start, end]
    确保 GT 范围在截取范围内，并随机选择起始点
    """
```

**utils.py:**

```python
def get_video_info(video_path: Path) -> dict:
    """获取视频的 fps、总帧数等信息"""

def clip_video(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float
) -> bool:
    """使用 FFmpeg 截取视频片段"""

def generate_video_hash() -> str:
    """生成 32 位 UUID 哈希（无连字符）"""
```

### 4. 使用的库

| 库 | 用途 |
|---|---|
| polars | CSV 读写和数据处理 |
| subprocess + ffmpeg | 视频截取（保留音频） |
| uuid | 生成随机哈希 |
| cv2 | 获取视频元信息（fps、帧数） |
| loguru | 日志记录 |
| concurrent.futures | 线程池并发处理 |
| tqdm | 进度条显示 |

### 5. FFmpeg 截取命令

```bash
ffmpeg -ss START_SEC -i input.mp4 -t DURATION_SEC -c:v libx264 -c:a aac -y output.mp4
```

注意：`-ss` 放在 `-i` 前面可以实现快速 seek。

### 6. 性能优化策略

考虑到运行环境（低功耗 CPU + 机械硬盘），采用以下优化策略：

**IO 优化（针对 HDD）：**
- 使用 FFmpeg 的 `-ss` 前置 seek，避免解码整个视频
- 使用 `-c:v copy -c:a copy` 直接复制流（当不需要截取时），避免重新编码
- 批量读取 CSV 后在内存中处理，减少磁盘 IO

**并发策略：**
- 使用 `concurrent.futures.ThreadPoolExecutor` 进行并发处理
- FFmpeg 是 IO 密集型任务，线程池比进程池更适合
- 默认 worker 数量 = 4（HDD 不宜过多并发，避免磁头频繁寻道）
- 提供 `--workers` 参数允许用户调整

**内存优化：**
- 使用 Polars 的惰性求值（lazy evaluation）处理大 CSV
- 分批处理视频，避免一次性加载所有任务到内存

**进度显示：**
- 使用 `tqdm` 显示处理进度
- 记录已处理的视频，支持断点续传

### 7. 数据流示意

```
原始 CSV (all.csv)
    │
    ▼
按 origin_id 分组
    │
    ▼
对每个 origin:
    ├─ 按 gt_start_f 排序
    ├─ 按间隔分簇
    └─ 对每个簇:
        ├─ 计算截取范围
        ├─ 生成新 origin_id (UUID)
        ├─ 截取视频
        └─ 更新记录的 origin_id, gt_start_f, gt_end_f
    │
    ▼
新 CSV (all.csv, train.csv, valid.csv, test.csv)
```

## 验证方案

1. **数量检查**：新 CSV 记录数 = 原 CSV 记录数 (30000)
2. **帧号检查**：所有 gt_start_f >= 0，gt_end_f <= 视频总帧数
3. **抽样验证**：随机选取 10 个视频，检查 GT 帧是否在截取范围内
4. **视频完整性**：检查所有输出视频是否可正常播放

## 命令行接口

```bash
cd ~/pychen/DeepTrace
source src/.venv/bin/activate
python -m dataset.vorpus_processor \
    --input-dir E:/dataset/videos/vorpus \
    --output-dir E:/dataset/videos/vorpus_new \
    --csv-dir asset/dataset \
    --output-csv-dir asset/dataset/new \
    --max-duration 300 \
    --workers 4
```

## 实现步骤

1. 创建 `dataset` 分支
2. 创建 `src/dataset/` 目录结构
3. 实现 `utils.py`（视频工具函数）
4. 实现 `vorpus_processor.py`（主处理逻辑）
5. 运行处理脚本
6. 验证输出结果
