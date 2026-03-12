import math
import datetime
import torch
import torch.nn.utils as nn_utils
import torch.distributed as dist

from copy import deepcopy
from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Tuple, Type, Union, Optional, Dict, Any
from abc import ABC, abstractmethod

from exogym.aux.utils import LogModule


# BASE COMMUNICATION
def mps_compatible(func):
    # Wrapper for all_gather which handles tensor_list and tensor
    def all_gather_wrapper(tensor_list, tensor, *args, **kwargs):
        # Check if either is on MPS
        is_tensor_mps = hasattr(tensor, "device") and tensor.device.type == "mps"
        is_list_mps = any(
            hasattr(t, "device") and t.device.type == "mps" for t in tensor_list
        )

        if is_tensor_mps or is_list_mps:
            # Convert tensor to CPU if needed
            if is_tensor_mps:
                cpu_tensor = tensor.data.to("cpu")
            else:
                cpu_tensor = tensor

            # Convert tensor_list to CPU if needed
            cpu_tensor_list = []
            for t in tensor_list:
                if hasattr(t, "device") and t.device.type == "mps":
                    cpu_tensor_list.append(t.data.to("cpu"))
                else:
                    cpu_tensor_list.append(t)

            # Call function with CPU tensors
            result = func(cpu_tensor_list, cpu_tensor, *args, **kwargs)

            # Copy data back to original devices
            if is_tensor_mps:
                tensor.data.copy_(cpu_tensor.to("mps"))

            for i, t in enumerate(tensor_list):
                if hasattr(t, "device") and t.device.type == "mps":
                    t.data.copy_(cpu_tensor_list[i].to("mps"))

            return result
        else:
            return func(tensor_list, tensor, *args, **kwargs)

    # Wrapper for all other functions that handle a single tensor
    def standard_wrapper(tensor, *args, **kwargs):
        if hasattr(tensor, "device") and tensor.device.type == "mps":
            # Move the tensor to CPU
            cpu_tensor = tensor.data.to("cpu")
            # Call the function on CPU
            result = func(cpu_tensor, *args, **kwargs)
            # Copy the result back to mps
            tensor.data.copy_(cpu_tensor.to("mps"))
            return result
        else:
            return func(tensor, *args, **kwargs)

    # Return the appropriate wrapper based on function name
    if func.__name__ == "all_gather":
        return all_gather_wrapper
    else:
        return standard_wrapper


@mps_compatible
def broadcast(tensor, src=0):
    return dist.broadcast(tensor, src=src)


@mps_compatible
def all_reduce(tensor, op=dist.ReduceOp.SUM):
    return dist.all_reduce(tensor, op=op)


@mps_compatible
def all_gather(tensor_list, tensor, group=None, async_op=False):
    return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)


# @mps_compatible
# def reduce_scatter(tensor):
#     return dist.reduce_scatter(tensor)

# @mps_compatible
# def reduce(tensor):
#     return dist.reduce(tensor)

# @mps_compatible
# def gather(tensor):
#     return dist.gather(tensor)


# BASE OPTIMIZER SPEC
@dataclass
class OptimSpec:
    cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    kwargs: Dict[str, Any] = None  # e.g. {'lr': 3e-4}

    def __init__(self, cls: Type[torch.optim.Optimizer], **kwargs: Dict[str, Any]):
        self.cls = cls
        self.kwargs = kwargs

    @classmethod
    def from_string(cls, name: str, **kwargs) -> "OptimSpec":
        """Create OptimSpec from optimizer name string."""
        optimizer_map = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }

        name_lower = name.lower()
        if name_lower not in optimizer_map:
            available = ", ".join(optimizer_map.keys())
            raise ValueError(
                f"Unknown optimizer '{name}'. Available options: {available}"
            )

        return cls(optimizer_map[name_lower], **kwargs)

    def build(self, model):
        return self.cls(model.parameters(), **(self.kwargs or {}))


def ensure_optim_spec(
    optim: Union[str, OptimSpec, None], default: Optional[OptimSpec] = None, **kwargs
) -> OptimSpec:
    """Convert string or OptimSpec to OptimSpec instance."""
    if optim is None:
        if default is None:
            return OptimSpec(torch.optim.AdamW, **kwargs)
        else:
            return default
    elif isinstance(optim, str):
        return OptimSpec.from_string(optim, **kwargs)
    elif isinstance(optim, OptimSpec):
        # If additional kwargs provided, merge them
        if kwargs:
            merged_kwargs = {**(optim.kwargs or {}), **kwargs}
            return OptimSpec(optim.cls, **merged_kwargs)
        return optim
    else:
        raise TypeError(f"Expected str, OptimSpec, or None, got {type(optim)}")


# BASE STRATEGY
class Strategy(ABC, LogModule):
    def __init__(
        self,
        lr_scheduler: str = None,
        lr_scheduler_kwargs: Dict[str, Any] = None,
        **kwargs: Dict[str, Any],
    ):
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.kwargs = kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Initialize scheduler as None; will be set after self.optim is defined in subclasses.
        self.scheduler = None

        # List of callbacks to record learning rate changes.
        self.lr_callbacks = []

        self.max_steps = 1  # Needs to be initialized for first call of lr_lambda.

    def _init_node(self, model, rank, num_nodes):
        self.model = model
        self.rank = rank
        self.num_nodes = num_nodes

        self.local_step = 0

        if hasattr(self, "optim_spec"):
            self.optim = self.optim_spec.build(model)

    @abstractmethod
    def step(self):
        self.nbytes = 0

        if self.scheduler is not None:
            self.scheduler.step()

            if self.rank == 0:
                for callback in self.lr_callbacks:
                    callback(self.scheduler.get_last_lr()[0])

        self.local_step += 1

    def zero_grad(self):
        self.optim.zero_grad()

    def _setup_scheduler(self):
        def lr_lambda(current_step):
            warmup_steps = self.lr_scheduler_kwargs.get("warmup_steps", 1)
            # If max steps not set,
            if "max_steps" in self.lr_scheduler_kwargs:
                max_steps = min(self.lr_scheduler_kwargs["max_steps"], self.max_steps)
            else:
                max_steps = self.max_steps
            cosine_anneal = self.lr_scheduler_kwargs.get("cosine_anneal", False)

            if current_step < warmup_steps:
                return float(current_step) / float(max(warmup_steps, 1))
            elif cosine_anneal:
                min_lr_factor = 0.1
                progress = (current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (1 - min_lr_factor) * cosine_term + min_lr_factor
            else:
                return 1.0

        if self.lr_scheduler == "lambda_cosine":
            self.scheduler = LambdaLR(self.optim, lr_lambda)
        elif self.lr_scheduler is not None:
            lr_sched_kwargs = (
                self.lr_scheduler_kwargs if self.lr_scheduler_kwargs is not None else {}
            )
            self.scheduler = self.lr_scheduler(self.optim, **lr_sched_kwargs)
        else:
            self.scheduler = None

    def __config__(self):
        remove_keys = [
            "iteration",
            "local_step",
            "lr_callbacks",
            "model",
            "optim",
            "scheduler",
        ]

        config = super().__config__(remove_keys)

        config["strategy"] = self.__class__.__name__

        return config


# BASE COMMUNICATE OPTIMIZER STRATEGY
class CommunicationModule(ABC):
    """Abstract base class for communication modules."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """
        Perform communication for the given model.

        Args:
          model: The model to communicate
          rank: Current node rank
          num_nodes: Total number of nodes
          local_step: Current local step count
        """
        pass

    @abstractmethod
    def _init_node(self, model, rank: int, num_nodes: int) -> None:
        """
        Initialize the communication module for the given model.
        """
        pass


class CommunicateOptimizeStrategy(Strategy):
    """
    Base class for strategies that interleave communication and optimization.

    This strategy:
    1. Performs local optimization step
    2. Applies communication modules when the derived strategy decides
    """

    def __init__(
        self,
        communication_modules: List[CommunicationModule],
        optim_spec: Optional[Union[str, OptimSpec]] = None,
        max_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.optim_spec = ensure_optim_spec(optim_spec) or OptimSpec(torch.optim.AdamW)

        self.communication_modules = communication_modules
        self.max_norm = max_norm

        # Set strategy reference in communication modules that need it
        for comm_module in self.communication_modules:
            comm_module.strategy = self

    def step(self):
        # Gradient clipping if specified
        if self.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

        # Local optimization step
        self.optim.step()

        # Communication phase - let derived strategies decide when
        self._communicate()

        super().step()

    def _communicate(self):
        """Apply all communication modules sequentially. Override in derived classes for custom timing."""
        for comm_module in self.communication_modules:
            comm_module.communicate(
                self.model, self.rank, self.num_nodes, self.local_step
            )

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)

        for comm_module in self.communication_modules:
            comm_module._init_node(model, rank, num_nodes)

        self.optim = self.optim_spec.build(model)
        self._setup_scheduler()


# DILOCO COMMUNICATOR
class DiLoCoCommunicator(CommunicationModule):
    """
    Communication module for master-worker setup (like DiLoCo).
    """

    def __init__(
        self,
        H: int = 100,
        outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
        **kwargs,
    ):
        self.H = H
        self.outer_optim_spec = ensure_optim_spec(
            outer_optim_spec,
            OptimSpec(torch.optim.SGD, lr=0.7, nesterov=True, momentum=0.9),
        )
        self.strategy = None  # Will be set by CommunicateOptimizeStrategy
        self.master_model = None
        self.outer_optimizer = None

    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """Perform master-worker communication."""
        if num_nodes > 1 and local_step % self.H == 0 and local_step > 0:
            # First average all models
            for param in model.parameters():
                all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= num_nodes

            # Master does outer optimization step
            if rank == 0 and self.master_model is not None:
                self.outer_optimizer.zero_grad()
                self._set_master_grad(model)
                self.outer_optimizer.step()
                self._synchronize_master_model(model)

            # Broadcast updated parameters
            for param in model.parameters():
                broadcast(param.data, src=0)

    def _init_node(self, model, rank: int, num_nodes: int) -> None:
        """Initialize master model for rank 0."""
        if rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for param in self.master_model.parameters():
                param.requires_grad = True
            self.outer_optimizer = self.outer_optim_spec.build(self.master_model)

    def _set_master_grad(self, model) -> None:
        """Set gradients on master model based on difference between master and worker models."""
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - model.state_dict()[name].data.to("cpu")

    def _synchronize_master_model(self, model) -> None:
        """Synchronize worker model with master model parameters."""
        for name, param in model.named_parameters():
            param.data = self.master_model.state_dict()[name].data.to(param.device)


# DILOCO STRATEGY
class DiLoCoStrategy(CommunicateOptimizeStrategy):
    def __init__(
        self,
        optim_spec: Optional[
            Union[str, OptimSpec]
        ] = None,  # inner optimizer is named optim_spec for consistency
        outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
        H: int = 100,
        **kwargs,
    ):
        self.H = H

        # Ensure optim_spec is properly initialized
        optim_spec = ensure_optim_spec(optim_spec, OptimSpec(torch.optim.AdamW))

        # Create the DiLoCo communicator
        self.diloco_comm = DiLoCoCommunicator(H=H, outer_optim_spec=outer_optim_spec)

        super().__init__(
            optim_spec=optim_spec, communication_modules=[self.diloco_comm], **kwargs
        )


class UniformKBitQuantizer:
    def __init__(self, n_bins: int, range_in_sigmas: float):
        assert n_bins > 0 and (n_bins & (n_bins - 1)) == 0
        self.n_bins = n_bins
        self.range = range_in_sigmas

    @torch.no_grad()
    def quantize(self, val: torch.Tensor):
        offset = self.n_bins // 2
        shift = val.mean()
        centered = val - shift
        std = centered.norm() / math.sqrt(max(centered.numel() - 1, 1))
        scale = self.range * std / self.n_bins
        if scale.item() == 0 or not torch.isfinite(scale):
            scale = torch.tensor(1.0, device=val.device)
        q = ((centered / scale) + offset).round().clamp(0, self.n_bins - 1).to(torch.uint8)
        original_shape = q.shape
        q_flat = q.flatten()
        pack = 4
        pad = (pack - q_flat.numel() % pack) % pack
        if pad:
            q_flat = torch.cat([q_flat, torch.zeros(pad, dtype=torch.uint8, device=q_flat.device)])
        r = q_flat.view(-1, pack)
        packed = r[:, 0] | (r[:, 1] << 2) | (r[:, 2] << 4) | (r[:, 3] << 6)
        return packed, (shift, scale, original_shape, pad)

    @torch.no_grad()
    def dequantize(self, packed: torch.Tensor, meta: Tuple):
        shift, scale, shape, pad = meta
        offset = self.n_bins // 2
        mask = torch.tensor(3, dtype=torch.uint8, device=packed.device)
        unpacked = torch.stack([(packed >> i) & mask for i in (0, 2, 4, 6)], dim=0).transpose(0, 1).flatten()
        if pad:
            unpacked = unpacked[:-pad]
        return (unpacked.float() - offset) * scale + shift.to(unpacked.device)


class SelectiveDiLoCoCommunicator(DiLoCoCommunicator):
    """2-bit EF compression for embedding (wte) only; fp32 all_reduce for transformer layers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = UniformKBitQuantizer(n_bins=4, range_in_sigmas=4)
        self.error_buffers = {}
        self.gloo_group = None

    def _init_node(self, model, rank: int, num_nodes: int) -> None:
        self.gloo_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=60))
        self.master_model = deepcopy(model).to("cpu")
        for param in self.master_model.parameters():
            param.requires_grad = True
        self.outer_optimizer = self.outer_optim_spec.build(self.master_model)
        # Only buffer for embedding params
        self.error_buffers = {
            name: torch.zeros_like(p.data)
            for name, p in self.master_model.named_parameters()
            if "wte" in name
        }

    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        if num_nodes > 1 and local_step % self.H == 0 and local_step > 0:
            # fp32 all_reduce for transformer (non-embedding) params
            for name, param in model.named_parameters():
                if "wte" not in name:
                    all_reduce(param.data, op=dist.ReduceOp.SUM)
                    param.data /= num_nodes

            self.outer_optimizer.zero_grad()
            self._set_master_grad(model)

            # 2-bit EF for embedding (wte) pseudo-gradients
            for name, p in self.master_model.named_parameters():
                if "wte" not in name:
                    p.grad.copy_(p.grad)  # already correct from all_reduce above
                    continue
                delta = p.grad
                delta_ef = delta + self.error_buffers[name]
                q_packed, meta = self.quantizer.quantize(delta_ef)
                shift, scale, shape, pad = meta
                meta_tensor = torch.stack([shift.cpu(), scale.cpu()])
                gathered_metas = [torch.zeros_like(meta_tensor) for _ in range(num_nodes)]
                dist.all_gather(gathered_metas, meta_tensor, group=self.gloo_group)
                gathered_q = [torch.empty_like(q_packed) for _ in range(num_nodes)]
                dist.all_gather(gathered_q, q_packed, group=self.gloo_group)
                delta_sum = torch.zeros_like(p.data)
                for i in range(num_nodes):
                    node_meta = (gathered_metas[i][0], gathered_metas[i][1], shape, pad)
                    delta_sum += self.quantizer.dequantize(gathered_q[i], node_meta).reshape(p.shape)
                delta_hat = delta_sum / num_nodes
                my_decompressed = self.quantizer.dequantize(q_packed, meta).reshape(p.shape)
                self.error_buffers[name].copy_(delta_ef - my_decompressed)
                p.grad.copy_(delta_hat)

            self.outer_optimizer.step()

            # Broadcast master params to all workers
            self._synchronize_master_model(model)
            for param in model.parameters():
                broadcast(param.data, src=0)


class SelectiveDiLoCoStrategy(DiLoCoStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diloco_comm = SelectiveDiLoCoCommunicator(
            H=self.H, outer_optim_spec=self.diloco_comm.outer_optim_spec
        )
        self.communication_modules = [self.diloco_comm]


STRATEGY = SelectiveDiLoCoStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001, weight_decay=0.1),
    outer_optim_spec=OptimSpec(torch.optim.SGD, lr=0.9, nesterov=True, momentum=0.9),
    lr_scheduler="lambda_cosine",
    lr_scheduler_kwargs={
        "warmup_steps": 500,
        "cosine_anneal": True,
    },
    max_norm=1.0,
    H=15,
)
