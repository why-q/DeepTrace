"""
视频增强模块

对视频进行社交媒体风格的数据增强，包括：
- 编码质量降低（模拟平台压缩）
- 颜色抖动（模拟不同设备显示差异）

CLI 使用示例:
    python -m deeptrace.preprocessing.video_augmenter --input-dir E:/Videos --output-dir E:/AugmentedVideos
"""

import json
import random
from pathlib import Path
from typing import Optional
import augly.video as avd
from deeptrace.utils.parallel import parallel_process


class VideoAugmenter:
    """
    视频增强器
    
    对视频应用社交媒体常见的增强变换，所有参数在合理范围内随机选择。
    """
    
    def __init__(
        self,
        encoding_quality_range: tuple[int, int] = (20, 40),
        brightness_range: tuple[float, float] = (0.1, 0.3),
        contrast_range: tuple[float, float] = (0.1, 0.3),
        saturation_range: tuple[float, float] = (0.1, 0.3),
        seed: Optional[int] = None
    ):
        """
        初始化视频增强器
        
        Args:
            encoding_quality_range: 编码质量范围 (min, max)，值越小质量越低
                默认 (20, 40) 覆盖中度到重度压缩
            brightness_range: 亮度调整因子范围 (min, max)
            contrast_range: 对比度调整因子范围 (min, max)
            saturation_range: 饱和度调整因子范围 (min, max)
            seed: 随机种子，用于可复现的增强
        """
        self.encoding_quality_range = encoding_quality_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        
        if seed is not None:
            random.seed(seed)
    
    def augment_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        apply_compression: bool = True,
        apply_color_jitter: bool = True
    ) -> dict:
        """
        对单个视频应用增强
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            apply_compression: 是否应用压缩
            apply_color_jitter: 是否应用颜色抖动
            
        Returns:
            dict: 应用的增强参数
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 记录应用的参数
        augmentation_params = {}
        
        # 临时文件路径
        temp_path = None
        current_input = str(input_path)
        
        try:
            # 1. 应用压缩
            if apply_compression:
                quality = random.randint(*self.encoding_quality_range)
                augmentation_params['encoding_quality'] = quality
                
                if apply_color_jitter:
                    # 如果还要应用颜色抖动，先输出到临时文件
                    temp_path = output_path.parent / f"temp_{output_path.name}"
                    avd.encoding_quality(
                        video_path=current_input,
                        output_path=str(temp_path),
                        quality=quality
                    )
                    current_input = str(temp_path)
                else:
                    # 直接输出到目标文件
                    avd.encoding_quality(
                        video_path=current_input,
                        output_path=str(output_path),
                        quality=quality
                    )
            
            # 2. 应用颜色抖动
            if apply_color_jitter:
                brightness_factor = random.uniform(*self.brightness_range)
                contrast_factor = random.uniform(*self.contrast_range)
                saturation_factor = random.uniform(*self.saturation_range)
                
                augmentation_params['color_jitter'] = {
                    'brightness_factor': brightness_factor,
                    'contrast_factor': contrast_factor,
                    'saturation_factor': saturation_factor
                }
                
                avd.color_jitter(
                    video_path=current_input,
                    output_path=str(output_path),
                    brightness_factor=brightness_factor,
                    contrast_factor=contrast_factor,
                    saturation_factor=saturation_factor
                )
            
            # 清理临时文件
            if temp_path and temp_path.exists():
                temp_path.unlink()
            
            return augmentation_params
            
        except Exception as e:
            # 清理临时文件
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise e
    
    def batch_augment(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.mp4",
        apply_compression: bool = True,
        apply_color_jitter: bool = True,
        keep_filename: bool = True,
        suffix: str = "_aug",
        save_params: bool = True,
        verbose: bool = True
    ) -> list[dict]:
        """
        批量增强文件夹中的所有视频
        
        Args:
            input_dir: 输入视频目录
            output_dir: 输出视频目录
            pattern: 文件匹配模式，默认 "*.mp4"
            apply_compression: 是否应用压缩
            apply_color_jitter: 是否应用颜色抖动
            keep_filename: 如果为 True，保持原文件名；如果为 False，添加后缀
            suffix: 输出文件名后缀，默认 "_aug"（仅当 keep_filename=False 时使用）
            save_params: 是否将增强参数保存到 JSON 文件
            verbose: 是否打印处理进度
            
        Returns:
            list[dict]: 每个视频的增强参数列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有匹配的视频文件
        video_files = sorted(input_dir.glob(pattern))
        
        if not video_files:
            print(f"警告: 在 {input_dir} 中没有找到匹配 '{pattern}' 的视频文件")
            return []
        
        if verbose:
            print(f"找到 {len(video_files)} 个视频文件")
            print(f"输入目录: {input_dir}")
            print(f"输出目录: {output_dir}")
            print()
        
        results = []
        params_dict = {}  # Store all parameters with video ID as key
        
        for i, video_path in enumerate(video_files, 1):
            if verbose:
                print(f"[{i}/{len(video_files)}] 处理: {video_path.name}")
            
            # 生成输出文件名
            if keep_filename:
                output_filename = video_path.name  # Keep original filename
            else:
                output_filename = f"{video_path.stem}{suffix}{video_path.suffix}"
            output_path = output_dir / output_filename
            
            # Use video stem (without extension) as ID
            video_id = video_path.stem
            
            # 检查输出文件是否已存在
            if output_path.exists():
                result = {
                    'video_id': video_id,
                    'input_file': str(video_path),
                    'output_file': str(output_path),
                    'augmentation_params': None,
                    'status': 'skipped',
                    'message': '输出文件已存在'
                }
                results.append(result)
                
                if verbose:
                    print(f"  ⊘ 跳过: 输出文件已存在")
                    print()
                
                continue
            
            try:
                # 应用增强
                params = self.augment_video(
                    input_path=video_path,
                    output_path=output_path,
                    apply_compression=apply_compression,
                    apply_color_jitter=apply_color_jitter
                )
                
                result = {
                    'video_id': video_id,
                    'input_file': str(video_path),
                    'output_file': str(output_path),
                    'augmentation_params': params,
                    'status': 'success'
                }
                
                # Store params with video ID as key
                params_dict[video_id] = params
                
                if verbose:
                    print(f"  ✓ 完成: {output_filename}")
                    if apply_compression:
                        print(f"    - 压缩质量: {params.get('encoding_quality', 'N/A')}")
                    if apply_color_jitter:
                        cj = params.get('color_jitter', {})
                        print(f"    - 颜色抖动: 亮度={cj.get('brightness_factor', 0):.3f}, "
                              f"对比度={cj.get('contrast_factor', 0):.3f}, "
                              f"饱和度={cj.get('saturation_factor', 0):.3f}")
                
            except Exception as e:
                result = {
                    'video_id': video_id,
                    'input_file': str(video_path),
                    'output_file': str(output_path),
                    'augmentation_params': None,
                    'status': 'failed',
                    'error': str(e)
                }
                
                if verbose:
                    print(f"  ✗ 失败: {str(e)}")
            
            results.append(result)
            
            if verbose:
                print()
        
        # Save parameters to JSON file
        if save_params and params_dict:
            params_file = output_dir / "augmentation_params.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(params_dict, f, indent=2, ensure_ascii=False)
            
            if verbose:
                print(f"Saved augmentation parameters to: {params_file}")
                print()
        
        # 打印总结
        if verbose:
            success_count = sum(1 for r in results if r['status'] == 'success')
            skipped_count = sum(1 for r in results if r['status'] == 'skipped')
            failed_count = sum(1 for r in results if r['status'] == 'failed')
            print("=" * 80)
            print(f"批量处理完成！")
            print(f"  成功: {success_count}")
            print(f"  跳过: {skipped_count}")
            print(f"  失败: {failed_count}")
            print(f"  总计: {len(results)}")
            print("=" * 80)
        
        return results
    
    def batch_augment_parallel(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.mp4",
        apply_compression: bool = True,
        apply_color_jitter: bool = True,
        keep_filename: bool = True,
        suffix: str = "_aug",
        save_params: bool = True,
        num_workers: Optional[int] = None,
        verbose: bool = True
    ) -> list[dict]:
        """
        使用多进程并行批量增强文件夹中的所有视频
        
        Args:
            input_dir: 输入视频目录
            output_dir: 输出视频目录
            pattern: 文件匹配模式，默认 "*.mp4"
            apply_compression: 是否应用压缩
            apply_color_jitter: 是否应用颜色抖动
            keep_filename: 如果为 True，保持原文件名；如果为 False，添加后缀
            suffix: 输出文件名后缀，默认 "_aug"（仅当 keep_filename=False 时使用）
            save_params: 是否将增强参数保存到 JSON 文件
            num_workers: 并行工作进程数，None 表示使用 CPU 核心数
            verbose: 是否打印处理进度
            
        Returns:
            list[dict]: 每个视频的增强参数列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有匹配的视频文件
        video_files = sorted(input_dir.glob(pattern))
        
        if not video_files:
            print(f"警告: 在 {input_dir} 中没有找到匹配 '{pattern}' 的视频文件")
            return []
        
        if verbose:
            print(f"找到 {len(video_files)} 个视频文件")
            print(f"输入目录: {input_dir}")
            print(f"输出目录: {output_dir}")
            print()
        
        # 准备任务参数
        tasks = []
        for video_path in video_files:
            # 生成输出文件名
            if keep_filename:
                output_filename = video_path.name
            else:
                output_filename = f"{video_path.stem}{suffix}{video_path.suffix}"
            output_path = output_dir / output_filename
            
            tasks.append({
                'video_path': video_path,
                'output_path': output_path,
                'video_id': video_path.stem,
                'apply_compression': apply_compression,
                'apply_color_jitter': apply_color_jitter
            })
        
        # 使用多进程并行处理
        results = parallel_process(
            func=_process_single_video,
            items=tasks,
            num_workers=num_workers,
            show_progress=True,
            verbose=verbose,
            augmenter_params={
                'encoding_quality_range': self.encoding_quality_range,
                'brightness_range': self.brightness_range,
                'contrast_range': self.contrast_range,
                'saturation_range': self.saturation_range
            }
        )
        
        # 收集成功的参数
        params_dict = {}
        for result in results:
            if result['status'] == 'success' and result['augmentation_params']:
                params_dict[result['video_id']] = result['augmentation_params']
        
        # 保存参数到 JSON 文件
        if save_params and params_dict:
            params_file = output_dir / "augmentation_params.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(params_dict, f, indent=2, ensure_ascii=False)
            
            if verbose:
                print(f"已保存增强参数到: {params_file}")
                print()
        
        # 打印总结
        if verbose:
            success_count = sum(1 for r in results if r['status'] == 'success')
            skipped_count = sum(1 for r in results if r['status'] == 'skipped')
            failed_count = sum(1 for r in results if r['status'] == 'failed')
            print("=" * 80)
            print(f"并行批量处理完成！")
            print(f"  成功: {success_count}")
            print(f"  跳过: {skipped_count}")
            print(f"  失败: {failed_count}")
            print(f"  总计: {len(results)}")
            print("=" * 80)
        
        return results


def _process_single_video(task: dict, augmenter_params: dict) -> dict:
    """
    处理单个视频的工作函数（用于多进程）
    
    Args:
        task: 包含视频路径和参数的字典
        augmenter_params: VideoAugmenter 的初始化参数
        
    Returns:
        dict: 处理结果
    """
    import random
    from pathlib import Path
    import augly.video as avd
    
    video_path = task['video_path']
    output_path = task['output_path']
    video_id = task['video_id']
    apply_compression = task['apply_compression']
    apply_color_jitter = task['apply_color_jitter']
    
    # 检查输出文件是否已存在
    if output_path.exists():
        return {
            'video_id': video_id,
            'input_file': str(video_path),
            'output_file': str(output_path),
            'augmentation_params': None,
            'status': 'skipped',
            'message': '输出文件已存在'
        }
    
    # 从参数中获取范围
    encoding_quality_range = augmenter_params['encoding_quality_range']
    brightness_range = augmenter_params['brightness_range']
    contrast_range = augmenter_params['contrast_range']
    saturation_range = augmenter_params['saturation_range']
    
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 记录应用的参数
        augmentation_params = {}
        
        # 临时文件路径
        temp_path = None
        current_input = str(video_path)
        
        # 1. 应用压缩
        if apply_compression:
            quality = random.randint(*encoding_quality_range)
            augmentation_params['encoding_quality'] = quality
            
            if apply_color_jitter:
                # 如果还要应用颜色抖动，先输出到临时文件
                temp_path = output_path.parent / f"temp_{output_path.name}"
                avd.encoding_quality(
                    video_path=current_input,
                    output_path=str(temp_path),
                    quality=quality
                )
                current_input = str(temp_path)
            else:
                # 直接输出到目标文件
                avd.encoding_quality(
                    video_path=current_input,
                    output_path=str(output_path),
                    quality=quality
                )
        
        # 2. 应用颜色抖动
        if apply_color_jitter:
            brightness_factor = random.uniform(*brightness_range)
            contrast_factor = random.uniform(*contrast_range)
            saturation_factor = random.uniform(*saturation_range)
            
            augmentation_params['color_jitter'] = {
                'brightness_factor': brightness_factor,
                'contrast_factor': contrast_factor,
                'saturation_factor': saturation_factor
            }
            
            avd.color_jitter(
                video_path=current_input,
                output_path=str(output_path),
                brightness_factor=brightness_factor,
                contrast_factor=contrast_factor,
                saturation_factor=saturation_factor
            )
        
        # 清理临时文件
        if temp_path and temp_path.exists():
            temp_path.unlink()
        
        return {
            'video_id': video_id,
            'input_file': str(video_path),
            'output_file': str(output_path),
            'augmentation_params': augmentation_params,
            'status': 'success'
        }
        
    except Exception as e:
        # 清理临时文件
        if temp_path and temp_path.exists():
            temp_path.unlink()
        
        return {
            'video_id': video_id,
            'input_file': str(video_path),
            'output_file': str(output_path),
            'augmentation_params': None,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="对视频进行社交媒体风格的数据增强",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个目录的所有 MP4 视频
  python -m deeptrace.preprocessing.video_augmenter --input-dir E:/Videos --output-dir E:/AugVideos
  
  # 使用多进程并行处理（自动使用所有 CPU 核心）
  python -m deeptrace.preprocessing.video_augmenter \\
      --input-dir E:/Videos \\
      --output-dir E:/AugVideos \\
      --parallel
  
  # 使用多进程并行处理，指定工作进程数
  python -m deeptrace.preprocessing.video_augmenter \\
      --input-dir E:/Videos \\
      --output-dir E:/AugVideos \\
      --parallel \\
      --num-workers 8
  
  # 自定义压缩质量范围
  python -m deeptrace.preprocessing.video_augmenter \\
      --input-dir E:/Videos \\
      --output-dir E:/AugVideos \\
      --quality-min 15 \\
      --quality-max 35
  
  # 只应用压缩，不应用颜色抖动
  python -m deeptrace.preprocessing.video_augmenter \\
      --input-dir E:/Videos \\
      --output-dir E:/AugVideos \\
      --no-color-jitter
  
  # 设置随机种子以获得可复现的结果
  python -m deeptrace.preprocessing.video_augmenter \\
      --input-dir E:/Videos \\
      --output-dir E:/AugVideos \\
      --seed 42
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入视频目录"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出视频目录"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="文件匹配模式 (默认: *.mp4)"
    )
    
    parser.add_argument(
        "--quality-min",
        type=int,
        default=20,
        help="编码质量最小值 (默认: 20，值越小质量越低)"
    )
    
    parser.add_argument(
        "--quality-max",
        type=int,
        default=40,
        help="编码质量最大值 (默认: 40，值越小质量越低)"
    )
    
    parser.add_argument(
        "--brightness-min",
        type=float,
        default=0.1,
        help="亮度调整因子最小值 (默认: 0.1)"
    )
    
    parser.add_argument(
        "--brightness-max",
        type=float,
        default=0.3,
        help="亮度调整因子最大值 (默认: 0.3)"
    )
    
    parser.add_argument(
        "--contrast-min",
        type=float,
        default=0.1,
        help="对比度调整因子最小值 (默认: 0.1)"
    )
    
    parser.add_argument(
        "--contrast-max",
        type=float,
        default=0.3,
        help="对比度调整因子最大值 (默认: 0.3)"
    )
    
    parser.add_argument(
        "--saturation-min",
        type=float,
        default=0.1,
        help="饱和度调整因子最小值 (默认: 0.1)"
    )
    
    parser.add_argument(
        "--saturation-max",
        type=float,
        default=0.3,
        help="饱和度调整因子最大值 (默认: 0.3)"
    )
    
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="不应用编码压缩"
    )
    
    parser.add_argument(
        "--no-color-jitter",
        action="store_true",
        help="不应用颜色抖动"
    )
    
    parser.add_argument(
        "--keep-filename",
        action="store_true",
        help="保持原文件名 (默认: 保持原名)"
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        default="_aug",
        help="输出文件名后缀 (默认: _aug, 仅当不保持原名时使用)"
    )
    
    parser.add_argument(
        "--no-save-params",
        action="store_true",
        help="不保存增强参数到 JSON 文件"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于可复现的增强"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式，不打印详细进度"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="使用多进程并行处理视频"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="并行工作进程数（仅在 --parallel 模式下有效），默认使用 CPU 核心数"
    )
    
    args = parser.parse_args()
    
    # 创建增强器
    augmenter = VideoAugmenter(
        encoding_quality_range=(args.quality_min, args.quality_max),
        brightness_range=(args.brightness_min, args.brightness_max),
        contrast_range=(args.contrast_min, args.contrast_max),
        saturation_range=(args.saturation_min, args.saturation_max),
        seed=args.seed
    )
    
    # 执行批量增强
    if args.parallel:
        # 使用多进程并行处理
        results = augmenter.batch_augment_parallel(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            apply_compression=not args.no_compression,
            apply_color_jitter=not args.no_color_jitter,
            keep_filename=args.keep_filename,
            suffix=args.suffix,
            save_params=not args.no_save_params,
            num_workers=args.num_workers,
            verbose=not args.quiet
        )
    else:
        # 使用单进程顺序处理
        results = augmenter.batch_augment(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            apply_compression=not args.no_compression,
            apply_color_jitter=not args.no_color_jitter,
            keep_filename=args.keep_filename,
            suffix=args.suffix,
            save_params=not args.no_save_params,
            verbose=not args.quiet
        )
    
    # 如果有失败的任务，返回非零退出码
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    if failed_count > 0:
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


