"""
多进程并行处理工具

提供基于 multiprocessing 的并行处理功能，支持进度显示和错误处理。
"""

import multiprocessing as mp
from typing import Callable, Any, Optional
from functools import partial
from pathlib import Path


class ParallelProcessor:
    """
    多进程并行处理器
    
    封装了 multiprocessing 的常用功能，提供简洁的 API 用于并行处理任务列表。
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        verbose: bool = True
    ):
        """
        初始化并行处理器
        
        Args:
            num_workers: 工作进程数量，None 表示使用 CPU 核心数
            verbose: 是否打印详细日志
        """
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = min(num_workers, mp.cpu_count())
        
        self.verbose = verbose
    
    def process(
        self,
        func: Callable,
        items: list[Any],
        **kwargs
    ) -> list[Any]:
        """
        并行处理任务列表
        
        Args:
            func: 处理函数，接收单个 item 和额外的关键字参数
            items: 待处理的项目列表
            **kwargs: 传递给 func 的额外关键字参数
            
        Returns:
            list[Any]: 处理结果列表，顺序与输入一致
        """
        if not items:
            if self.verbose:
                print("警告: 没有待处理的项目")
            return []
        
        if self.verbose:
            print(f"使用 {self.num_workers} 个 CPU 核心并行处理 {len(items)} 个任务...")
        
        # 使用 partial 固定额外的关键字参数
        if kwargs:
            process_func = partial(func, **kwargs)
        else:
            process_func = func
        
        # 使用进程池并行处理
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(process_func, items)
        
        if self.verbose:
            print(f"所有任务处理完成！")
        
        return results
    
    def process_with_progress(
        self,
        func: Callable,
        items: list[Any],
        **kwargs
    ) -> list[Any]:
        """
        并行处理任务列表，带进度显示
        
        Args:
            func: 处理函数，接收单个 item 和额外的关键字参数
            items: 待处理的项目列表
            **kwargs: 传递给 func 的额外关键字参数
            
        Returns:
            list[Any]: 处理结果列表，顺序与输入一致
        """
        if not items:
            if self.verbose:
                print("警告: 没有待处理的项目")
            return []
        
        if self.verbose:
            print(f"使用 {self.num_workers} 个 CPU 核心并行处理 {len(items)} 个任务...")
        
        # 使用 partial 固定额外的关键字参数
        if kwargs:
            process_func = partial(func, **kwargs)
        else:
            process_func = func
        
        # 使用进程池和 imap 来实现进度显示
        results = []
        with mp.Pool(processes=self.num_workers) as pool:
            for i, result in enumerate(pool.imap(process_func, items), 1):
                results.append(result)
                if self.verbose:
                    print(f"进度: {i}/{len(items)} ({100*i/len(items):.1f}%)", end='\r')
        
        if self.verbose:
            print()  # 换行
            print(f"所有任务处理完成！")
        
        return results


def parallel_process(
    func: Callable,
    items: list[Any],
    num_workers: Optional[int] = None,
    show_progress: bool = True,
    verbose: bool = True,
    **kwargs
) -> list[Any]:
    """
    快捷函数：并行处理任务列表
    
    Args:
        func: 处理函数，接收单个 item 和额外的关键字参数
        items: 待处理的项目列表
        num_workers: 工作进程数量，None 表示使用 CPU 核心数
        show_progress: 是否显示进度条
        verbose: 是否打印详细日志
        **kwargs: 传递给 func 的额外关键字参数
        
    Returns:
        list[Any]: 处理结果列表，顺序与输入一致
    
    示例:
        >>> def process_item(item, multiplier=1):
        ...     return item * multiplier
        >>> 
        >>> items = [1, 2, 3, 4, 5]
        >>> results = parallel_process(process_item, items, multiplier=2)
        >>> print(results)
        [2, 4, 6, 8, 10]
    """
    processor = ParallelProcessor(num_workers=num_workers, verbose=verbose)
    
    if show_progress:
        return processor.process_with_progress(func, items, **kwargs)
    else:
        return processor.process(func, items, **kwargs)

