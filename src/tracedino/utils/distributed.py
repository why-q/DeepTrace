"""Distributed training utilities for TraceDINO."""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed training from environment variables.

    Supports torchrun launcher which sets:
        - RANK: global rank
        - LOCAL_RANK: local rank on this node
        - WORLD_SIZE: total number of processes

    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )
            dist.barrier()
        return rank, local_rank, world_size
    else:
        # Single GPU fallback
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    """Get the number of processes in the distributed group."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all processes with gradient support.

    This custom autograd function allows gradients to flow back through
    the all_gather operation, which is necessary for contrastive learning
    where we need to compute gradients w.r.t. features from all GPUs.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(input)
        if not dist.is_initialized():
            return (input,)

        world_size = dist.get_world_size()
        output = [torch.zeros_like(input) for _ in range(world_size)]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        if not dist.is_initialized():
            return grads[0]

        rank = dist.get_rank()
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[rank]
        return grad_out


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes while preserving gradients.

    Args:
        tensor: Input tensor [B, D]

    Returns:
        Concatenated tensor from all processes [B*world_size, D]
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor

    gathered = GatherLayer.apply(tensor)
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def gather_without_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes without gradient tracking.

    Args:
        tensor: Input tensor [B, ...]

    Returns:
        Concatenated tensor from all processes [B*world_size, ...]
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor

    world_size = dist.get_world_size()
    output = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor)
    return torch.cat(output, dim=0)


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor across all processes by averaging.

    Args:
        tensor: Input tensor

    Returns:
        Averaged tensor
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor
