# Vorpus Dataset Processor

将 Vorpus 数据集的长视频根据 Ground Truth (GT) 帧范围截取为最长 5 分钟的片段，并更新 CSV 元数据。

## 快速开始

```bash
# 1. 进入项目目录并激活虚拟环境
cd ~/pychen/DeepTrace
source src/.venv/bin/activate

# 2. 运行处理脚本（从 src 目录）
cd src
python -m dataset.vorpus_processor \
    --input-dir E:/dataset/videos/vorpus \
    --output-dir E:/dataset/videos/vorpus_new \
    --csv-dir ../asset/dataset \
    --output-csv-dir ../asset/dataset/new \
    --max-duration 300 \
    --workers 4
```

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-dir` | 是 | - | 原始视频目录 (0000.mp4 ~ 1573.mp4) |
| `--output-dir` | 是 | - | 输出视频目录 (UUID 命名) |
| `--csv-dir` | 是 | - | 原始 CSV 目录 (包含 all.csv, train.csv 等) |
| `--output-csv-dir` | 是 | - | 输出 CSV 目录 |
| `--max-duration` | 否 | 300 | 最大视频时长（秒） |
| `--workers` | 否 | 4 | 并发线程数 |
| `--limit` | 否 | None | 限制处理的 origin 数量（用于测试） |

## 测试运行

```bash
# 小批量测试（处理 5 个 origin）
python -m dataset.vorpus_processor \
    --input-dir E:/dataset/videos/vorpus \
    --output-dir E:/dataset/videos/vorpus_new \
    --csv-dir ../asset/dataset \
    --output-csv-dir ../asset/dataset/new \
    --limit 5
```

## 输入输出

**输入：**
- 视频：`E:/dataset/videos/vorpus/0000.mp4` ~ `1573.mp4`
- CSV：`asset/dataset/{all,train,valid,test}.csv`

**输出：**
- 视频：`E:/dataset/videos/vorpus_new/{uuid}.mp4`
- CSV：`asset/dataset/new/{all,train,valid,test}.csv`

## 核心算法

1. 按 `origin_id` 分组所有记录
2. 对每个 origin，按 `gt_start_f` 排序后贪心分组（相邻 GT 间隔 ≤ 5 分钟则合并）
3. 对每个分组截取视频片段，生成 UUID 命名的新文件
4. 更新该组所有记录的 `origin_id` 和帧号（`gt_start_f`, `gt_end_f`）

## 依赖

- polars
- opencv-python
- ffmpeg (系统安装)
- loguru
- tqdm
