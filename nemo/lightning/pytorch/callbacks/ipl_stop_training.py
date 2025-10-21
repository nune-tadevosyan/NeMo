from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule

class IPLEpochStopper(Callback):
    """
    Gracefully terminates training at the *end* of the current epoch
    when `enable_stop` is True.

    Typical use-case: iterative pseudo-labelling (IPL) where the outer
    loop (IPLMixin.maybe_do_ipl) needs control after each epoch.

    Args
    ----
    enable_stop: bool
        Set via config.  If True the callback sets `trainer.should_stop`
        at `on_train_epoch_end`, causing Lightning to exit the fit loop
        cleanly after schedulers/logging/ckpt hooks have run.
    """
    def __init__(self, enable_stop: bool = False):
        super().__init__()
        self.enable_stop = enable_stop
        print("**"*80)

    def state_dict(self):
        """Return the state of the callback."""
        return {"enable_stop": self.enable_stop}

    def load_state_dict(self, state_dict):
        """Load the state of the callback."""
        self.enable_stop = state_dict["enable_stop"]

    # ------------------------------------------------------------------ #
    # Call once per epoch *after* all training batches of that epoch run #
    # ------------------------------------------------------------------ #
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.enable_stop:
            # Lightning v2+ : requesting a clean stop
            trainer.should_stop = True      # rank-local flag
            trainer.strategy.request_stop() # multi-process syn
