from .strategy import Strategy
from .communicate import all_gather

from .demo_impl.demo import DeMo


## TODO: This is really slow at the moment...
class DeMoStrategy(Strategy):
    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store DeMo-specific parameters
        self.compression_decay = compression_decay
        self.compression_topk = compression_topk
        self.compression_chunk = compression_chunk
        self.weight_decay = weight_decay

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)

        print("initialising DeMo engine")

        # Create DeMo optimizer with stored parameters
        demo_kwargs = {
            "compression_decay": self.compression_decay,
            "compression_topk": self.compression_topk,
            "compression_chunk": self.compression_chunk,
            "weight_decay": self.weight_decay,
            "custom_all_gather": all_gather,
        }

        # Add any additional optimizer kwargs from strategy config if they exist
        if hasattr(self, "strategy_config") and hasattr(
            self.strategy_config, "optimizer_kwargs"
        ):
            demo_kwargs.update(self.strategy_config.optimizer_kwargs)

        self.optim = DeMo(model.parameters(), **demo_kwargs)
        self._setup_scheduler()

    def step(self):
        # DeMo communicates gradients and then does optimizer step.
        self.optim.step()

        super().step()  # Print number of bytes communicated. This can be put in a different method tbh.
