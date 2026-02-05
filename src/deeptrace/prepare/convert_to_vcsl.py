#!/usr/bin/env python3
"""
将 DeepTrace 数据集格式转换为 VCSL 输入格式

VCSL 需要的格式:
1. frames_all.csv: uuid, path, frame_count
2. pair_file_{train,val,test}.csv: query_id, reference_id, label
3. label_file.json: 包含正样本对的片段对应关系

正负样本构建策略:
- 正样本: 每个视频(id) 与其原视频(origin_id) 组成正样本对 (1个)
- 负样本: 从 similar.csv 中选择最相似的4个其他视频组成负样本对 (4个)
- 正负比例: 1:4
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import polars as pl
import numpy as np
from loguru import logger


def load_deeptrace_dataset(
    train_path: Path,
    val_path: Path,
    test_path: Path
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """加载 DeepTrace 数据集"""
    logger.info("加载 DeepTrace 数据集...")
    
    train_df = pl.read_csv(train_path)
    val_df = pl.read_csv(val_path)
    test_df = pl.read_csv(test_path)
    
    logger.info(f"训练集: {len(train_df)} 样本")
    logger.info(f"验证集: {len(val_df)} 样本")
    logger.info(f"测试集: {len(test_df)} 样本")
    
    return train_df, val_df, test_df


def load_similar_mapping(similar_path: Path) -> Dict[str, List[str]]:
    """
    加载相似度映射
    
    Returns:
        Dict[origin_id, List[similar_ids]]
        所有 ID 都格式化为四位数字（前面补零）
    """
    logger.info(f"加载相似度映射: {similar_path}")
    
    similar_df = pl.read_csv(similar_path)
    similar_map = {}
    
    for row in similar_df.iter_rows(named=True):
        # 确保 origin_id 始终为四位数（前面补零）
        origin_id = str(row['id']).zfill(4)
        # sim 列是用 '-' 分隔的相似视频 id，也需要格式化
        similar_ids = [str(sid).zfill(4) for sid in row['sim'].split('-')]
        similar_map[origin_id] = similar_ids
    
    logger.info(f"✓ 加载了 {len(similar_map)} 个原视频的相似度映射")
    return similar_map


def load_fps_mappings(
    videos_fps_path: Path,
    origins_fps_path: Path
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    加载视频和源视频的 FPS 映射
    
    Args:
        videos_fps_path: 短视频 FPS 文件路径
        origins_fps_path: 源视频 FPS 文件路径
    
    Returns:
        (videos_fps_map, origins_fps_map)
        videos_fps_map: Dict[video_id, fps]
        origins_fps_map: Dict[origin_id, fps]
    """
    logger.info(f"加载 FPS 映射...")
    
    # 加载短视频 FPS
    videos_fps_df = pl.read_csv(videos_fps_path)
    videos_fps_map = {}
    for row in videos_fps_df.iter_rows(named=True):
        videos_fps_map[row['id']] = float(row['fps'])
    
    # 加载源视频 FPS
    origins_fps_df = pl.read_csv(origins_fps_path)
    origins_fps_map = {}
    for row in origins_fps_df.iter_rows(named=True):
        # 确保 origin_id 为四位数格式
        origin_id = str(row['id']).zfill(4)
        origins_fps_map[origin_id] = float(row['fps'])
    
    logger.info(f"✓ 加载短视频 FPS: {len(videos_fps_map)} 个")
    logger.info(f"✓ 加载源视频 FPS: {len(origins_fps_map)} 个")
    
    return videos_fps_map, origins_fps_map


def generate_frames_all_csv(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_path: Path
) -> None:
    """
    生成 frames_all.csv
    
    格式: uuid, path, frame_count
    """
    logger.info("生成 frames_all.csv...")
    
    # 合并所有数据集
    all_df = pl.concat([train_df, val_df, test_df])
    
    # 提取需要的列
    frames_df = all_df.select([
        pl.col('id').alias('uuid'),
        pl.col('id').alias('path'),  # path 设置为与 uuid 相同
        pl.col('frames').alias('frame_count')
    ])
    
    # 去重
    frames_df = frames_df.unique(subset=['uuid'])
    
    # 保存
    frames_df.write_csv(output_path)
    logger.info(f"✓ 保存 frames_all.csv: {len(frames_df)} 个视频")


def compute_segment_alignment(
    gt_start_f: int,
    gt_end_f: int,
    frames: int,
    query_fps: float,
    ref_fps: float,
    min_segment_length: int = 100
) -> List[List[int]]:
    """
    计算正样本的片段对齐
    
    Args:
        gt_start_f: 原视频起始帧
        gt_end_f: 原视频结束帧
        frames: 伪造视频总帧数
        query_fps: 伪造视频的 FPS
        ref_fps: 原视频的 FPS
        min_segment_length: 最小片段长度（默认100帧）
    
    Returns:
        List of [query_start, ref_start, query_end, ref_end]
        过滤掉长度小于 min_segment_length 的段落
        ref_end - ref_start 保持与 query_end - query_start 相同
        所有值都除以对应的 FPS（1秒1帧的采样率）
    """
    segment_length = gt_end_f - gt_start_f
    segments = []
    
    if frames <= segment_length:
        # 情况1: 伪造视频比原片段短或相等
        # 整个伪造视频对应到原视频的 gt_start_f 到 gt_start_f+frames
        query_length = frames
        if query_length >= min_segment_length:
            # 除以 FPS，转换为秒数（1秒1帧采样）
            query_start_sec = int(0 / query_fps)
            query_end_sec = int(frames / query_fps)
            ref_start_sec = int(gt_start_f / ref_fps)
            ref_end_sec = int((gt_start_f + frames) / ref_fps)
            
            segments.append([query_start_sec, ref_start_sec, query_end_sec, ref_end_sec])
    else:
        # 情况2: 伪造视频比原片段长，进行了重复播放
        # 计算重复次数
        num_repeats = int(np.ceil(frames / segment_length))
        
        for i in range(num_repeats):
            query_start = i * segment_length
            query_end = min((i + 1) * segment_length, frames)
            query_length = query_end - query_start
            
            # 过滤掉长度小于 min_segment_length 的段落
            if query_length < min_segment_length:
                continue
            
            # 每次重复都对应到原视频的同一片段
            # ref_end - ref_start 应该等于 query_end - query_start
            ref_end = gt_start_f + query_length
            
            # 除以 FPS，转换为秒数（1秒1帧采样）
            query_start_sec = int(query_start / query_fps)
            query_end_sec = int(query_end / query_fps)
            ref_start_sec = int(gt_start_f / ref_fps)
            ref_end_sec = int(ref_end / ref_fps)
            
            segments.append([
                query_start_sec,
                ref_start_sec,
                query_end_sec,
                ref_end_sec
            ])
            
            # 如果已经覆盖了所有帧，停止
            if query_end >= frames:
                break
    
    return segments


def generate_pair_file_and_labels(
    df: pl.DataFrame,
    similar_map: Dict[str, List[str]],
    videos_fps_map: Dict[str, float],
    origins_fps_map: Dict[str, float],
    output_pair_path: Path,
    output_label_path: Path,
    negative_ratio: int = 4,
    seed: int = 42
) -> None:
    """
    生成 pair_file CSV 和 label_file JSON
    
    Args:
        df: 数据集 DataFrame
        similar_map: 相似度映射
        videos_fps_map: 短视频 FPS 映射
        origins_fps_map: 源视频 FPS 映射
        output_pair_path: 输出配对文件路径
        output_label_path: 输出标签文件路径
        negative_ratio: 每个正样本对应的负样本数量
        seed: 随机种子
    """
    logger.info(f"生成 {output_pair_path.name}...")
    
    np.random.seed(seed)
    
    pairs = []  # (query_id, reference_id, label)
    labels = {}  # {(query_id, reference_id): segments}
    
    positive_count = 0
    negative_count = 0
    missing_similar_count = 0
    filtered_segments_count = 0  # 被过滤的视频数（没有有效段落）
    total_segments_count = 0  # 总段落数
    missing_video_fps_count = 0  # 缺失短视频 FPS
    missing_origin_fps_count = 0  # 缺失源视频 FPS（使用短视频 FPS 回退）
    
    for row in df.iter_rows(named=True):
        video_id = row['id']
        # 确保 origin_id 始终为四位数（前面补零）
        origin_id = str(row['origin_id']).zfill(4)
        gt_start_f = row['gt_start_f']
        gt_end_f = row['gt_end_f']
        frames = row['frames']
        
        # 获取短视频 FPS
        if video_id not in videos_fps_map:
            logger.warning(f"缺失短视频 FPS: {video_id}，跳过")
            missing_video_fps_count += 1
            continue
        query_fps = videos_fps_map[video_id]
        
        # 获取源视频 FPS，如果找不到或为0，使用短视频 FPS
        if origin_id not in origins_fps_map or origins_fps_map[origin_id] == 0:
            ref_fps = query_fps
            missing_origin_fps_count += 1
        else:
            ref_fps = origins_fps_map[origin_id]
        
        # 1. 添加正样本对: video_id <-> origin_id
        pairs.append((video_id, origin_id, 1))
        positive_count += 1
        
        # 计算片段对齐（会过滤掉长度小于100帧的段落，并除以FPS）
        segments = compute_segment_alignment(
            gt_start_f, gt_end_f, frames,
            query_fps, ref_fps
        )
        
        # 只有当有有效段落时才添加标签
        if segments:
            labels[f"{video_id}-{origin_id}"] = segments
            total_segments_count += len(segments)
        else:
            filtered_segments_count += 1
        
        # 2. 添加负样本对: video_id <-> similar_ids (除了origin_id自己)
        if origin_id not in similar_map:
            # 如果没有相似度信息，随机选择其他视频作为负样本
            # 获取其他 origin_id 并格式化为四位数
            other_origins_raw = df.filter(pl.col('origin_id') != int(origin_id))['origin_id'].unique().to_list()
            other_origins = [str(oid).zfill(4) for oid in other_origins_raw]
            
            if len(other_origins) >= negative_ratio:
                negative_ids = np.random.choice(other_origins, negative_ratio, replace=False).tolist()
            else:
                negative_ids = other_origins.copy()
                # 补足到 negative_ratio 个（如果不够就重复）
                while len(negative_ids) < negative_ratio:
                    negative_ids.append(np.random.choice(other_origins))
            missing_similar_count += 1
        else:
            similar_ids = similar_map[origin_id]
            
            # 过滤掉 origin_id 自己
            negative_candidates = [sid for sid in similar_ids if sid != origin_id]
            
            # 选择前 negative_ratio 个作为负样本
            if len(negative_candidates) >= negative_ratio:
                negative_ids = negative_candidates[:negative_ratio]
            else:
                # 如果不够，用其他随机视频补足
                negative_ids = negative_candidates.copy()
                other_origins_raw = df.filter(pl.col('origin_id') != int(origin_id))['origin_id'].unique().to_list()
                other_origins = [str(oid).zfill(4) for oid in other_origins_raw if str(oid).zfill(4) not in negative_ids]
                
                if len(other_origins) > 0:
                    needed = negative_ratio - len(negative_ids)
                    additional = np.random.choice(
                        other_origins,
                        min(needed, len(other_origins)),
                        replace=False
                    ).tolist()
                    negative_ids.extend(additional)
        
        # 添加负样本对
        for neg_id in negative_ids:
            pairs.append((video_id, neg_id, 0))
            negative_count += 1
    
    # 打乱顺序
    np.random.shuffle(pairs)
    
    # 保存配对文件
    pair_df = pl.DataFrame(pairs, schema=['query_id', 'reference_id', 'label'])
    pair_df.write_csv(output_pair_path)
    
    # 保存标签文件
    with open(output_label_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    logger.info(f"✓ 保存 {output_pair_path.name}: {len(pairs)} 对")
    logger.info(f"  正样本: {positive_count} ({positive_count/len(pairs)*100:.1f}%)")
    logger.info(f"  负样本: {negative_count} ({negative_count/len(pairs)*100:.1f}%)")
    logger.info(f"  缺失相似度映射: {missing_similar_count} 个视频")
    logger.info(f"✓ 保存 {output_label_path.name}: {len(labels)} 个正样本标注")
    logger.info(f"  总段落数: {total_segments_count}")
    logger.info(f"  被过滤的视频数（段落长度<100帧）: {filtered_segments_count}")
    if missing_video_fps_count > 0:
        logger.warning(f"  缺失短视频 FPS 的视频数: {missing_video_fps_count}")
    if missing_origin_fps_count > 0:
        logger.info(f"  缺失/为0 的源视频 FPS（使用短视频FPS回退）: {missing_origin_fps_count}")


def convert_deeptrace_to_vcsl(
    dataset_dir: Path,
    similar_path: Path,
    videos_fps_path: Path,
    origins_fps_path: Path,
    output_dir: Path,
    negative_ratio: int = 4,
    seed: int = 42
) -> None:
    """
    将 DeepTrace 数据集转换为 VCSL 格式
    
    Args:
        dataset_dir: DeepTrace 数据集目录
        similar_path: 相似度映射文件路径
        videos_fps_path: 短视频 FPS 文件路径
        origins_fps_path: 源视频 FPS 文件路径
        output_dir: 输出目录
        negative_ratio: 负样本比例 (默认 4，即 1:4)
        seed: 随机种子
    """
    dataset_dir = Path(dataset_dir)
    similar_path = Path(similar_path)
    videos_fps_path = Path(videos_fps_path)
    origins_fps_path = Path(origins_fps_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DeepTrace → VCSL 格式转换")
    logger.info("=" * 60)
    logger.info(f"正负样本比例: 1:{negative_ratio}")
    
    # 1. 加载数据集
    train_df, val_df, test_df = load_deeptrace_dataset(
        dataset_dir / "train.csv",
        dataset_dir / "valid.csv",
        dataset_dir / "test.csv"
    )
    
    # 2. 加载相似度映射
    logger.info("\n" + "-" * 60)
    similar_map = load_similar_mapping(similar_path)
    
    # 3. 加载 FPS 映射
    logger.info("\n" + "-" * 60)
    videos_fps_map, origins_fps_map = load_fps_mappings(
        videos_fps_path,
        origins_fps_path
    )
    
    # 4. 生成 frames_all.csv
    logger.info("\n" + "-" * 60)
    generate_frames_all_csv(
        train_df, val_df, test_df,
        output_dir / "frames_all.csv"
    )
    
    # 5. 生成 pair_file_train.csv 和 label_file_train.json
    logger.info("\n" + "-" * 60)
    generate_pair_file_and_labels(
        train_df,
        similar_map,
        videos_fps_map,
        origins_fps_map,
        output_dir / "pair_file_train.csv",
        output_dir / "label_file_train.json",
        negative_ratio=negative_ratio,
        seed=seed
    )
    
    # 6. 生成 pair_file_val.csv 和 label_file_val.json
    logger.info("\n" + "-" * 60)
    generate_pair_file_and_labels(
        val_df,
        similar_map,
        videos_fps_map,
        origins_fps_map,
        output_dir / "pair_file_val.csv",
        output_dir / "label_file_val.json",
        negative_ratio=negative_ratio,
        seed=seed + 1
    )
    
    # 7. 生成 pair_file_test.csv 和 label_file_test.json
    logger.info("\n" + "-" * 60)
    generate_pair_file_and_labels(
        test_df,
        similar_map,
        videos_fps_map,
        origins_fps_map,
        output_dir / "pair_file_test.csv",
        output_dir / "label_file_test.json",
        negative_ratio=negative_ratio,
        seed=seed + 2
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ 转换完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    # 打印文件列表
    logger.info("\n生成的文件:")
    for file in sorted(output_dir.glob("*")):
        logger.info(f"  - {file.name}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="将 DeepTrace 数据集转换为 VCSL 格式"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="asset/dataset/final",
        help="DeepTrace 数据集目录 (默认: asset/dataset/final)"
    )
    parser.add_argument(
        "--similar-file",
        type=str,
        default="asset/dataset/final/similar.csv",
        help="相似度映射文件 (默认: asset/dataset/final/similar.csv)"
    )
    parser.add_argument(
        "--videos-fps",
        type=str,
        default="asset/fps/videos_fps.csv",
        help="短视频 FPS 文件路径 (默认: asset/fps/videos_fps.csv)"
    )
    parser.add_argument(
        "--origins-fps",
        type=str,
        default="asset/fps/origins.csv",
        help="源视频 FPS 文件路径 (默认: asset/fps/origins.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="asset/vcsl_data",
        help="输出目录 (默认: asset/vcsl_data)"
    )
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=4,
        help="负样本比例 (默认: 4，即 1:4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    
    args = parser.parse_args()
    
    convert_deeptrace_to_vcsl(
        dataset_dir=Path(args.dataset_dir),
        similar_path=Path(args.similar_file),
        videos_fps_path=Path(args.videos_fps),
        origins_fps_path=Path(args.origins_fps),
        output_dir=Path(args.output_dir),
        negative_ratio=args.negative_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
