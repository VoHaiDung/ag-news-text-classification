"""
Distributed training utilities for AG News Text Classification Framework.

Provides utilities for distributed training across multiple GPUs and nodes
following PyTorch distributed training best practices.

References:
    - Li, S., et al. (2020). "PyTorch Distributed: Experiences on Accelerating 
      Data Parallel Training". Proceedings of the VLDB Endowment.
    - Sergeev, A., & Del Balso, M. (2018). "Horovod: fast and easy distributed 
      deep learning in TensorFlow". arXiv preprint arXiv:1802.05799.

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
"""

import os
import logging
import socket
import subprocess
from typing import Optional, Any, List, Dict, Union, Tuple
from datetime import timedelta
from contextlib import contextmanager
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import ReduceOp
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


class DistributedManager:
    """
    Manager for distributed training operations.
    
    Handles initialization, communication, and synchronization across
    distributed processes following PyTorch best practices.
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        timeout: int = 30,
    ):
        """
        Initialize distributed manager.
        
        Args:
            backend: Backend to use (nccl, gloo, mpi)
            init_method: URL specifying how to initialize the process group
            world_size: Number of processes participating
            rank: Rank of current process
            timeout: Timeout for operations in minutes
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.timeout = timedelta(minutes=timeout)
        self.initialized = False
        
        # Store original environment variables
        self._original_env = {}
    
    def setup(self, local_rank: Optional[int] = None):
        """
        Setup distributed training environment.
        
        Args:
            local_rank: Local rank for GPU assignment
        """
        if self.initialized:
            logger.warning("Distributed already initialized")
            return
        
        # Get configuration from environment if not provided
        if self.world_size is None:
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if self.rank is None:
            self.rank = int(os.environ.get("RANK", 0))
        
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Single process, no need for distributed
        if self.world_size == 1:
            logger.info("Single process training, distributed not initialized")
            return
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            logger.info(f"Process {self.rank} using GPU {local_rank}")
        
        # Initialize process group
        if self.init_method is None:
            # Use environment variable if available
            self.init_method = os.environ.get("MASTER_ADDR", None)
            if self.init_method:
                master_port = os.environ.get("MASTER_PORT", "29500")
                self.init_method = f"tcp://{self.init_method}:{master_port}"
            else:
                # Default to localhost
                self.init_method = "tcp://127.0.0.1:29500"
        
        logger.info(
            f"Initializing distributed: backend={self.backend}, "
            f"world_size={self.world_size}, rank={self.rank}, "
            f"init_method={self.init_method}"
        )
        
        dist.init_process_group(
            backend=self.backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.rank,
            timeout=self.timeout,
        )
        
        self.initialized = True
        logger.info(f"Distributed training initialized for rank {self.rank}")
        
        # Synchronize all processes
        self.barrier()
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized = False
            logger.info("Distributed training cleaned up")
    
    def barrier(self):
        """Synchronize all processes."""
        if self.initialized:
            dist.barrier()
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        async_op: bool = False
    ):
        """
        All-reduce operation across all processes.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            async_op: Whether to perform asynchronously
            
        Returns:
            Handle for async operation or None
        """
        if self.initialized:
            return dist.all_reduce(tensor, op=op, async_op=async_op)
        return None
    
    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        async_op: bool = False
    ):
        """
        All-gather operation across all processes.
        
        Args:
            tensor_list: List to store gathered tensors
            tensor: Tensor to gather
            async_op: Whether to perform asynchronously
            
        Returns:
            Handle for async operation or None
        """
        if self.initialized:
            return dist.all_gather(tensor_list, tensor, async_op=async_op)
        return None
    
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        async_op: bool = False
    ):
        """
        Broadcast tensor from source to all processes.
        
        Args:
            tensor: Tensor to broadcast
            src: Source rank
            async_op: Whether to perform asynchronously
            
        Returns:
            Handle for async operation or None
        """
        if self.initialized:
            return dist.broadcast(tensor, src=src, async_op=async_op)
        return None
    
    def is_main_process(self) -> bool:
        """Check if current process is the main process."""
        return self.rank == 0 if self.initialized else True
    
    def get_world_size(self) -> int:
        """Get world size."""
        return self.world_size if self.initialized else 1
    
    def get_rank(self) -> int:
        """Get current rank."""
        return self.rank if self.initialized else 0
    
    def wrap_model(
        self,
        model: torch.nn.Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
    ) -> torch.nn.Module:
        """
        Wrap model for distributed data parallel training.
        
        Args:
            model: Model to wrap
            device_ids: GPU devices to use
            output_device: Device for output
            find_unused_parameters: Find unused parameters
            gradient_as_bucket_view: Use gradient as bucket view
            
        Returns:
            Wrapped model
        """
        if not self.initialized:
            return model
        
        if device_ids is None and torch.cuda.is_available():
            device_ids = [torch.cuda.current_device()]
        
        if output_device is None and device_ids:
            output_device = device_ids[0]
        
        model = DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        
        logger.info("Model wrapped for distributed training")
        return model
    
    def scale_learning_rate(self, base_lr: float, scaling: str = "linear") -> float:
        """
        Scale learning rate for distributed training.
        
        Args:
            base_lr: Base learning rate
            scaling: Scaling strategy (linear, sqrt)
            
        Returns:
            Scaled learning rate
        """
        if not self.initialized or self.world_size == 1:
            return base_lr
        
        if scaling == "linear":
            scaled_lr = base_lr * self.world_size
        elif scaling == "sqrt":
            scaled_lr = base_lr * (self.world_size ** 0.5)
        else:
            scaled_lr = base_lr
        
        logger.info(
            f"Learning rate scaled from {base_lr:.6f} to {scaled_lr:.6f} "
            f"(world_size={self.world_size}, scaling={scaling})"
        )
        
        return scaled_lr
    
    def gather_object(self, obj: Any, dst: int = 0) -> Optional[List[Any]]:
        """
        Gather objects from all processes.
        
        Args:
            obj: Object to gather
            dst: Destination rank
            
        Returns:
            List of gathered objects (only on dst rank)
        """
        if not self.initialized:
            return [obj]
        
        if self.rank == dst:
            gathered = [None] * self.world_size
        else:
            gathered = None
        
        dist.gather_object(obj, gathered, dst=dst)
        return gathered
    
    def reduce_dict(
        self,
        input_dict: Dict[str, torch.Tensor],
        average: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Reduce dictionary of tensors across all processes.
        
        Args:
            input_dict: Dictionary of tensors
            average: Whether to average after summing
            
        Returns:
            Reduced dictionary
        """
        if not self.initialized:
            return input_dict
        
        world_size = self.get_world_size()
        if world_size == 1:
            return input_dict
        
        with torch.no_grad():
            keys = sorted(input_dict.keys())
            values = [input_dict[k] for k in keys]
            
            # Stack and reduce
            stacked = torch.stack(values)
            dist.all_reduce(stacked, op=ReduceOp.SUM)
            
            if average:
                stacked /= world_size
            
            # Reconstruct dictionary
            reduced_dict = {k: v for k, v in zip(keys, stacked)}
        
        return reduced_dict


# Global distributed manager instance
_distributed_manager: Optional[DistributedManager] = None


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    local_rank: Optional[int] = None,
    timeout: int = 30,
) -> DistributedManager:
    """
    Setup distributed training.
    
    Args:
        backend: Backend to use
        init_method: Initialization method
        world_size: World size
        rank: Process rank
        local_rank: Local rank
        timeout: Timeout in minutes
        
    Returns:
        Distributed manager instance
    """
    global _distributed_manager
    
    if _distributed_manager is None:
        _distributed_manager = DistributedManager(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        _distributed_manager.setup(local_rank=local_rank)
    
    return _distributed_manager


def cleanup_distributed():
    """Cleanup distributed training."""
    global _distributed_manager
    
    if _distributed_manager is not None:
        _distributed_manager.cleanup()
        _distributed_manager = None


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_initialized() if dist.is_available() else False


def is_main_process() -> bool:
    """Check if current process is main process."""
    if _distributed_manager is not None:
        return _distributed_manager.is_main_process()
    return not is_distributed() or get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if _distributed_manager is not None:
        return _distributed_manager.get_rank()
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Get world size."""
    if _distributed_manager is not None:
        return _distributed_manager.get_world_size()
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    """Get local rank."""
    return int(os.environ.get("LOCAL_RANK", 0))


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    average: bool = True
) -> torch.Tensor:
    """
    All-reduce tensor across processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation
        average: Whether to average after reduction
        
    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor
    
    dist.all_reduce(tensor, op=op)
    
    if average and op == ReduceOp.SUM:
        tensor /= get_world_size()
    
    return tensor


def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    All-gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of gathered tensors
    """
    if not is_distributed():
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    return tensor_list


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source rank.
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        
    Returns:
        Broadcasted tensor
    """
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


@contextmanager
def distributed_zero_grad(model: torch.nn.Module):
    """
    Context manager for zeroing gradients in distributed training.
    
    Args:
        model: Model to zero gradients for
    """
    if is_distributed() and isinstance(model, DDP):
        with model.no_sync():
            yield
    else:
        yield


def reduce_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Reduce loss dictionary across all processes.
    
    Args:
        loss_dict: Dictionary of loss tensors
        
    Returns:
        Dictionary of reduced loss values
    """
    if not is_distributed():
        return {k: v.item() for k, v in loss_dict.items()}
    
    world_size = get_world_size()
    
    reduced_losses = {}
    for key, value in loss_dict.items():
        reduced_value = value.clone()
        dist.all_reduce(reduced_value)
        reduced_losses[key] = (reduced_value / world_size).item()
    
    return reduced_losses


def print_once(*args, **kwargs):
    """Print only on main process."""
    if is_main_process():
        print(*args, **kwargs)


def save_on_master(obj: Any, path: str):
    """
    Save object only on master process.
    
    Args:
        obj: Object to save
        path: Save path
    """
    if is_main_process():
        torch.save(obj, path)
        logger.info(f"Saved to {path}")
    
    # Synchronize to ensure save completes before other processes continue
    barrier()


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU devices.
    
    Returns:
        List of GPU device IDs
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_count = torch.cuda.device_count()
    return list(range(gpu_count))


def auto_select_backend() -> str:
    """
    Automatically select best backend based on hardware.
    
    Returns:
        Backend name
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "nccl"
    else:
        return "gloo"


# Export public API
__all__ = [
    "DistributedManager",
    "setup_distributed",
    "cleanup_distributed",
    "is_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "barrier",
    "all_reduce",
    "all_gather",
    "broadcast",
    "distributed_zero_grad",
    "reduce_loss_dict",
    "print_once",
    "save_on_master",
    "get_available_gpus",
    "auto_select_backend",
]
