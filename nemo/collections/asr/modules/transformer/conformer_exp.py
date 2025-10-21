
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, open_dict

# from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
# from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.conformer_mamba import ConformerLayerMamba
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    LocalAttRelPositionalEncoding,
    MultiHeadAttention,
    PositionalEncoding,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    StackingSubsampling,
    SubsamplingReductionModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.asr.parts.utils.regularization_utils import compute_stochastic_depth_drop_probs
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, ChannelType, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from nemo.collections.asr.modules.conformer_encoder import *

encoder = ConformerEncoder(feat_in=10,n_layers=1 ,d_model=512)

test_inp  = encoder.input_example()[0].to("cuda")

mamba_layer = Mamba(d_model=256)
print(test_inp[0].shape)
layer = ConformerLayerMamba(d_model=256, d_ff=10,self_attention_model='abs_pos').to("cuda")


criterion = nn.MSELoss()

# Initialize the optimizer
optimizer = optim.Adam(layer.parameters(), lr=0.001)
out = layer(test_inp)

target = torch.randn(out.shape, device="cuda")

print(test_inp[0])
loss = criterion(out, target)

# Zero the gradients
optimizer.zero_grad()

# Backward pass (compute gradients)
loss.backward()

# Update the parameters
optimizer.step()

# Print the output and loss for debugging
print("Output:", out)
print("Loss:", loss.item())