from types import MethodType
from typing import Any, Dict

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core import LightningModule
from lightning.pytorch import Trainer


class IPLEpochStopper(Callback):
    r"""
    Gracefully terminates training at the *end* of an epoch.
    This is done to generate pseudo-labels dynamically.
    enable_stop : bool, default=False
        If ``True`` the callback will request a stop in
        :py:meth:`on_train_epoch_end`.  If ``False`` it is inert.
    """

    def __init__(self, enable_stop: bool = False) -> None:
      super().__init__()
      self.enable_stop = bool(enable_stop)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
      """
      Sets `should_stop` stop flag to terminate the training.
      """
      super().__init__()
      if self.enable_stop:
          trainer.should_stop = True
  