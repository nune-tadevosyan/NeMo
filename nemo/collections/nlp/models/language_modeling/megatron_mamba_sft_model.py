# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import DictConfig
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.models.language_modeling.megatron_mamba_model import MegatronMambaModel

__all__ = ['MegatronMambaSFTModel']


class MegatronMambaSFTModel(MegatronGPTSFTModel, MegatronMambaModel):
    """
    Megatron Jamba Supervised Fine-Tuning
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):

        super().__init__(cfg, trainer=trainer)
        self.mcore_gpt = True
        self.validation_param_sync_overlap = self.cfg.get('validation_param_sync_overlap', False)

    def _reset_activation_checkpointing_args(self):
        pass

    def on_validation_model_zero_grad(self) -> None:
        """
        Skip gradient zeroing at the beginning of validation routine.
        This is needed when overlapping the AllGather of the updated parameters with the following valdation step.
        """
        if not self.validation_param_sync_overlap:
            MegatronBaseModel.on_validation_model_zero_grad(self)
