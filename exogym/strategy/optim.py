import torch

from typing import Type, Union, Optional

from dataclasses import dataclass
from typing import Dict, Any


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
