# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.mixins.asr_adapter_mixins import ASRAdapterModelMixin
from nemo.collections.asr.parts.mixins.interctc_mixin import InterCTCMixin
from nemo.collections.asr.parts.mixins.mixins import (
    ASRAdapterModelMixin,
    ASRBPEMixin,
    ASRModuleMixin,
    DiarizationMixin,
    IPLMixin,
)
from nemo.collections.asr.parts.mixins.transcription import (
    ASRTranscriptionMixin,
    TranscribeConfig,
    TranscriptionMixin,
    TranscriptionReturnType,
)
