"""
Video Cropper based on Face Detection

Crops videos to specified scale while keeping detected faces centered.
Uses face detection JSON results to intelligently crop videos.

Note: This module supports both the old 'crop_scale' API (direct scale factor)
and the new 'crop_degree' API (fraction of area to remove). The crop_degree
API is recommended for clarity.
"""

import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from deeptrace.utils.parallel import parallel_process


class VideoCropper:
    """
    Intelligent video cropper based on face detection results.

    Crops videos using either:
    1. crop_scale (legacy): Direct scale factor (0.5 = 50% of dimensions, 25% of area)
    2. crop_degree (recommended): Fraction of area to remove (0.25 = remove 25%, keep 75%)

    Attributes:
        crop_scale (float): Target size as fraction of original (0-1)
        crop_degree (float): Fraction of area to remove (0-1), overrides crop_scale if provided
        margin (float): Additional margin around face (fraction)
        face_vertical_offset (float): Vertical position of face in cropped frame
                                     (0=top, 0.5=center, 1.0=bottom)

    Example:
        >>> # New API (recommended)
        >>> cropper = VideoCropper(crop_degree=0.25)  # Remove 25% area, keep 75%
        >>> # Old API (legacy)
        >>> cropper = VideoCropper(crop_scale=0.866)  # Same effect as above
        >>> cropper.crop_video(
        ...     "video.mp4",
        ...     {"x": 100, "y": 100, "width": 200, "height": 200},
        ...     "output.mp4"
        ... )
    """

    def __init__(
        self,
        crop_scale: Optional[float] = None,
        crop_degree: Optional[float] = None,
        margin: float = 0.1,
        face_vertical_offset: float = 0.3,
    ):
        """
        Initialize video cropper.

        Args:
            crop_scale: (Legacy) Target size as fraction of original (0-1)
                       e.g., 0.5 = crop to 50% of dimensions (25% area)
                       Ignored if crop_degree is provided.
            crop_degree: (Recommended) Fraction of area to remove (0-1)
                        e.g., 0.25 = remove 25% of area, keep 75%
                        Takes precedence over crop_scale.
            margin: Additional margin around face (0-1)
                   e.g., 0.1 = add 10% padding around detected face
            face_vertical_offset: Vertical position of face in cropped frame (0-1)
                                 e.g., 0.3 = face at 30% from top (upper-center)
                                       0.5 = face at center (default behavior)
        """
        # Determine which API to use
        if crop_degree is not None:
            # New API: crop_degree specifies fraction of area to remove
            if not 0 <= crop_degree < 1:
                raise ValueError("crop_degree must be in [0, 1)")
            # Convert crop_degree to scale factor
            # retained_area = 1 - crop_degree
            # scale = sqrt(retained_area)
            retained_area = 1.0 - crop_degree
            self.crop_scale = math.sqrt(retained_area)
            self.crop_degree = crop_degree
        elif crop_scale is not None:
            # Old API: crop_scale is direct scale factor
            if not 0 < crop_scale <= 1:
                raise ValueError("crop_scale must be in (0, 1]")
            self.crop_scale = crop_scale
            self.crop_degree = 1.0 - (crop_scale ** 2)  # Reverse calculation
        else:
            # Default: crop_degree = 0.5 (remove 50% of area)
            self.crop_degree = 0.5
            self.crop_scale = math.sqrt(0.5)
        
        if not 0 <= margin < 1:
            raise ValueError("margin must be between 0 and 1")
        if not 0 <= face_vertical_offset <= 1:
            raise ValueError("face_vertical_offset must be between 0 and 1")

        self.margin = margin
        self.face_vertical_offset = face_vertical_offset

    def calculate_crop_region(
        self,
        face_bbox: Optional[Dict[str, int]],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Calculate crop region to keep face at upper-center, or use center crop if no face.

        When face is detected, places it at specified vertical offset (default: 30% from top)
        to allow body to be included in the frame.

        Args:
            face_bbox: Face bounding box with keys: x, y, width, height
                      If None, uses center crop
            frame_width: Original frame width
            frame_height: Original frame height

        Returns:
            Tuple of (crop_x, crop_y, crop_width, crop_height)
        """
        # Calculate target crop dimensions
        target_width = int(frame_width * self.crop_scale)
        target_height = int(frame_height * self.crop_scale)

        if face_bbox is None:
            # No face detected - use center crop
            crop_x = (frame_width - target_width) // 2
            crop_y = (frame_height - target_height) // 2
        else:
            # Calculate face center
            face_center_x = face_bbox["x"] + face_bbox["width"] // 2
            face_center_y = face_bbox["y"] + face_bbox["height"] // 2

            # Calculate crop region with face at upper-center position
            # This allows the body to be included below the face
            crop_x = face_center_x - target_width // 2
            crop_y = face_center_y - int(target_height * self.face_vertical_offset)

        # Ensure crop region is within frame boundaries
        crop_x = max(0, min(crop_x, frame_width - target_width))
        crop_y = max(0, min(crop_y, frame_height - target_height))

        # Ensure dimensions don't exceed frame
        crop_width = min(target_width, frame_width - crop_x)
        crop_height = min(target_height, frame_height - crop_y)

        return (crop_x, crop_y, crop_width, crop_height)

    def crop_video_opencv(
        self,
        input_path: Path,
        crop_region: Tuple[int, int, int, int],
        output_path: Path,
        preserve_audio: bool = True,
    ) -> bool:
        """
        Crop video using OpenCV (frame by frame).

        Args:
            input_path: Input video path
            crop_region: (x, y, width, height) crop region
            output_path: Output video path
            preserve_audio: Whether to preserve audio (requires ffmpeg)

        Returns:
            True if successful, False otherwise
        """
        crop_x, crop_y, crop_width, crop_height = crop_region

        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Create output video writer
        out = cv2.VideoWriter(
            str(output_path), fourcc, fps, (crop_width, crop_height)
        )

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop frame
            cropped_frame = frame[
                crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
            ]

            # Write cropped frame
            out.write(cropped_frame)
            frame_count += 1

        cap.release()
        out.release()

        print(f"  Processed {frame_count} frames")

        # Preserve audio if requested
        if preserve_audio:
            self._add_audio_ffmpeg(input_path, output_path)

        return True

    def crop_video_ffmpeg(
        self,
        input_path: Path,
        crop_region: Tuple[int, int, int, int],
        output_path: Path,
    ) -> bool:
        """
        Crop video using ffmpeg (faster, preserves quality).

        Args:
            input_path: Input video path
            crop_region: (x, y, width, height) crop region
            output_path: Output video path

        Returns:
            True if successful, False otherwise
        """
        crop_x, crop_y, crop_width, crop_height = crop_region

        # FFmpeg crop filter: crop=width:height:x:y
        crop_filter = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vf",
            crop_filter,
            "-c:a",
            "copy",  # Copy audio without re-encoding
            "-y",  # Overwrite output file
            str(output_path),
        ]

        try:
            # Run ffmpeg
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            return False
        except FileNotFoundError:
            print("FFmpeg not found. Please install ffmpeg or use OpenCV method.")
            return False

    def _add_audio_ffmpeg(self, input_path: Path, output_path: Path) -> bool:
        """
        Add audio from input video to output video using ffmpeg.

        Args:
            input_path: Original video with audio
            output_path: Cropped video without audio

        Returns:
            True if successful, False otherwise
        """
        temp_path = output_path.parent / f"{output_path.stem}_temp{output_path.suffix}"

        cmd = [
            "ffmpeg",
            "-i",
            str(output_path),  # Video without audio
            "-i",
            str(input_path),  # Original video with audio
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",  # ? makes audio optional
            "-shortest",
            "-y",
            str(temp_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            # Replace original with temp
            temp_path.replace(output_path)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg fails, keep video without audio
            if temp_path.exists():
                temp_path.unlink()
            return False

    def crop_video(
        self,
        input_path: Path,
        face_bbox: Optional[Dict[str, int]],
        output_path: Path,
        method: str = "ffmpeg",
    ) -> bool:
        """
        Crop video based on face bounding box, or use center crop if no face.

        Args:
            input_path: Input video path
            face_bbox: Face bounding box dict with x, y, width, height
                      If None, uses center crop
            output_path: Output video path
            method: Cropping method ('ffmpeg' or 'opencv')

        Returns:
            True if successful, False otherwise
        """
        # Get video dimensions
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return False

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate crop region (with or without face)
        crop_region = self.calculate_crop_region(
            face_bbox, frame_width, frame_height
        )

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Crop video using selected method
        if method == "ffmpeg":
            return self.crop_video_ffmpeg(input_path, crop_region, output_path)
        elif method == "opencv":
            return self.crop_video_opencv(input_path, crop_region, output_path)
        else:
            raise ValueError(f"Unknown method: {method}")

    def crop_videos_from_json(
        self,
        json_path: Path,
        video_dir: Path,
        output_dir: Path,
        method: str = "ffmpeg",
        face_index: int = 0,
    ) -> Dict[str, bool]:
        """
        Batch crop videos based on face detection JSON.

        Args:
            json_path: Path to face detection JSON file
            video_dir: Directory containing input videos
            output_dir: Directory for cropped videos
            method: Cropping method ('ffmpeg' or 'opencv')
            face_index: Which face to use if multiple detected (default: 0)

        Returns:
            Dictionary mapping video names to success status
        """
        # Load detection results
        with open(json_path, "r", encoding="utf-8") as f:
            detections = json.load(f)

        results = {}
        total = len(detections)
        successful = 0
        failed = 0
        skipped = 0

        print(f"\n{'=' * 60}")
        print(f"Video Cropping Pipeline")
        print(f"{'=' * 60}")
        print(f"Crop scale: {self.crop_scale * 100}%")
        print(f"Face position: {self.face_vertical_offset * 100:.0f}% from top")
        print(f"Method: {method}")
        print(f"Total videos: {total}")
        print(f"{'=' * 60}\n")

        for idx, (video_name, detection_data) in enumerate(detections.items(), 1):
            print(f"[{idx}/{total}] Processing: {video_name}")

            # Construct paths
            input_path = video_dir / video_name
            output_path = output_dir / video_name

            # Check if output file already exists
            if output_path.exists():
                print(f"  ⊘ Skipped: Output file already exists")
                results[video_name] = True
                skipped += 1
                continue

            # Skip if error in detection
            if "error" in detection_data:
                print(f"  ⚠ Warning: Detection error - using center crop")
                face_bbox = None  # Use center crop
            else:
                # Get faces from detection
                faces = detection_data.get("faces", [])
                
                if not faces:
                    # No faces detected - use center crop
                    print(f"  ⚠ No faces detected - using center crop")
                    face_bbox = None
                else:
                    # Use first face (or specified face if available)
                    actual_face_index = min(face_index, len(faces) - 1)
                    face_bbox = faces[actual_face_index]
                    
                    if len(faces) > 1:
                        print(f"  ℹ Multiple faces ({len(faces)}) detected - using face {actual_face_index}")

            if not input_path.exists():
                print(f"  ✗ Error: Video not found: {input_path}")
                results[video_name] = False
                failed += 1
                continue

            # Crop video (with face or center crop)
            try:
                success = self.crop_video(input_path, face_bbox, output_path, method)
                results[video_name] = success

                if success:
                    successful += 1
                    # Get output size
                    cap = cv2.VideoCapture(str(output_path))
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        crop_type = "face-centered" if face_bbox else "center"
                        print(f"  ✓ Cropped to {w}x{h} ({crop_type})")
                    else:
                        print(f"  ✓ Done")
                else:
                    failed += 1
                    print(f"  ✗ Failed to crop")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                results[video_name] = False
                failed += 1

        # Summary
        print(f"\n{'=' * 60}")
        print(f"Cropping complete!")
        print(f"Successful: {successful} | Skipped: {skipped} | Failed: {failed}")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 60}")

        return results

    def crop_videos_from_json_parallel(
        self,
        json_path: Path,
        video_dir: Path,
        output_dir: Path,
        method: str = "ffmpeg",
        face_index: int = 0,
        num_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, bool]:
        """
        使用多进程并行批量裁剪视频（基于人脸检测 JSON）

        Args:
            json_path: 人脸检测 JSON 文件路径
            video_dir: 输入视频目录
            output_dir: 裁剪后视频输出目录
            method: 裁剪方法 ('ffmpeg' 或 'opencv')
            face_index: 使用哪个人脸（如果检测到多个）(默认: 0)
            num_workers: 并行工作进程数，None 表示使用 CPU 核心数
            verbose: 是否打印详细进度

        Returns:
            字典，映射视频名称到成功状态
        """
        # 加载检测结果
        with open(json_path, "r", encoding="utf-8") as f:
            detections = json.load(f)

        if not detections:
            print("警告: JSON 文件中没有检测结果")
            return {}

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"并行视频裁剪")
            print(f"{'=' * 60}")
            print(f"裁剪比例: {self.crop_scale * 100}%")
            print(f"人脸位置: {self.face_vertical_offset * 100:.0f}% (从顶部)")
            print(f"裁剪方法: {method}")
            print(f"总视频数: {len(detections)}")
            print(f"{'=' * 60}\n")

        # 准备任务参数
        tasks = []
        for video_name, detection_data in detections.items():
            input_path = video_dir / video_name
            output_path = output_dir / video_name

            # 提取人脸边界框
            if "error" in detection_data:
                face_bbox = None  # 使用中心裁剪
            else:
                faces = detection_data.get("faces", [])
                if not faces:
                    face_bbox = None
                else:
                    actual_face_index = min(face_index, len(faces) - 1)
                    face_bbox = faces[actual_face_index]

            tasks.append({
                'video_name': video_name,
                'input_path': input_path,
                'output_path': output_path,
                'face_bbox': face_bbox,
                'method': method,
            })

        # 使用多进程并行处理
        results_list = parallel_process(
            func=_process_single_video_crop,
            items=tasks,
            num_workers=num_workers,
            show_progress=True,
            verbose=verbose,
            cropper_params={
                'crop_scale': self.crop_scale,
                'margin': self.margin,
                'face_vertical_offset': self.face_vertical_offset,
            }
        )

        # 将结果转换为字典
        results = {r['video_name']: r['success'] for r in results_list}

        # 统计结果
        if verbose:
            successful = sum(1 for r in results_list if r['status'] == 'success')
            skipped = sum(1 for r in results_list if r['status'] == 'skipped')
            failed = sum(1 for r in results_list if r['status'] == 'failed')
            
            print(f"\n{'=' * 60}")
            print(f"并行裁剪完成！")
            print(f"成功: {successful} | 跳过: {skipped} | 失败: {failed}")
            print(f"输出目录: {output_dir}")
            print(f"{'=' * 60}")

        return results


def _process_single_video_crop(task: dict, cropper_params: dict) -> dict:
    """
    处理单个视频的工作函数（用于多进程）

    Args:
        task: 包含视频路径和参数的字典
        cropper_params: VideoCropper 的初始化参数

    Returns:
        dict: 处理结果
    """
    import cv2
    import subprocess
    from pathlib import Path
    
    video_name = task['video_name']
    input_path = task['input_path']
    output_path = task['output_path']
    face_bbox = task['face_bbox']
    method = task['method']

    # 检查输出文件是否已存在
    if output_path.exists():
        return {
            'video_name': video_name,
            'success': True,
            'status': 'skipped',
            'message': '输出文件已存在'
        }

    # 检查输入文件是否存在
    if not input_path.exists():
        return {
            'video_name': video_name,
            'success': False,
            'status': 'failed',
            'error': f'输入视频不存在: {input_path}'
        }

    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 获取视频尺寸
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return {
                'video_name': video_name,
                'success': False,
                'status': 'failed',
                'error': '无法打开视频文件'
            }

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 计算裁剪区域
        crop_scale = cropper_params['crop_scale']
        face_vertical_offset = cropper_params['face_vertical_offset']
        
        target_width = int(frame_width * crop_scale)
        target_height = int(frame_height * crop_scale)

        if face_bbox is None:
            # 无人脸 - 使用中心裁剪
            crop_x = (frame_width - target_width) // 2
            crop_y = (frame_height - target_height) // 2
        else:
            # 有人脸 - 以人脸为中心裁剪
            face_center_x = face_bbox["x"] + face_bbox["width"] // 2
            face_center_y = face_bbox["y"] + face_bbox["height"] // 2
            crop_x = face_center_x - target_width // 2
            crop_y = face_center_y - int(target_height * face_vertical_offset)

        # 确保裁剪区域在视频边界内
        crop_x = max(0, min(crop_x, frame_width - target_width))
        crop_y = max(0, min(crop_y, frame_height - target_height))
        crop_width = min(target_width, frame_width - crop_x)
        crop_height = min(target_height, frame_height - crop_y)

        # 执行裁剪
        if method == "ffmpeg":
            crop_filter = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-vf", crop_filter,
                "-c:a", "copy",
                "-y",
                str(output_path),
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            
            return {
                'video_name': video_name,
                'success': True,
                'status': 'success',
                'crop_size': f"{crop_width}x{crop_height}"
            }
        else:
            return {
                'video_name': video_name,
                'success': False,
                'status': 'failed',
                'error': f'并行模式下仅支持 ffmpeg 方法，不支持 {method}'
            }

    except subprocess.CalledProcessError as e:
        return {
            'video_name': video_name,
            'success': False,
            'status': 'failed',
            'error': f'FFmpeg 错误: {e.stderr}'
        }
    except Exception as e:
        return {
            'video_name': video_name,
            'success': False,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """CLI entry point for video cropping."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Crop videos based on face detection results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop to 50% of original size
  python -m deeptrace.preprocessing.video_cropper \\
      --json results.json \\
      --video-dir videos/ \\
      --output-dir cropped/ \\
      --scale 0.5

  # Crop to 70% using OpenCV (slower but doesn't require ffmpeg)
  python -m deeptrace.preprocessing.video_cropper \\
      --json results.json \\
      --video-dir videos/ \\
      --output-dir cropped/ \\
      --scale 0.7 \\
      --method opencv
        """,
    )

    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to face detection JSON file",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing input videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for cropped videos",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Crop scale (0-1, e.g., 0.5 = 50%%) (default: 0.5)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Additional margin around face (0-1) (default: 0.1)",
    )
    parser.add_argument(
        "--face-offset",
        type=float,
        default=0.3,
        help="Vertical position of face in cropped frame (0-1, 0=top, 0.5=center) (default: 0.3)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ffmpeg", "opencv"],
        default="ffmpeg",
        help="Cropping method (default: ffmpeg)",
    )
    parser.add_argument(
        "--face-index",
        type=int,
        default=0,
        help="Which face to use if multiple detected (default: 0)",
    )

    args = parser.parse_args()

    # Create cropper
    cropper = VideoCropper(
        crop_scale=args.scale,
        margin=args.margin,
        face_vertical_offset=args.face_offset,
    )

    # Crop videos
    cropper.crop_videos_from_json(
        json_path=Path(args.json),
        video_dir=Path(args.video_dir),
        output_dir=Path(args.output_dir),
        method=args.method,
        face_index=args.face_index,
    )


if __name__ == "__main__":
    main()

