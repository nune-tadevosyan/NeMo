import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from hydra import initialize_config_dir, compose
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
initialize_config_dir(config_dir="/home/ntadevosyan/Documents/NeMo/")
cfg = compose(config_name="fc_l_rnnt_cache.yaml")

cfg.model.encoder.n_layers = 13
#cfg.model.encoder.d_model= 960
trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

trainable_params = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
print(trainable_params)