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

from typing import Dict, Optional, Tuple

import torch
import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    Chunking:
    If `enable_chunking` is True, each audio sample is split into optimally sized chunks
    (see `chunk_audio_sample`). This is useful for long audio inputs,
    allowing the model to process them in manageable segments. Note that when chunking is enabled,
    the same transcript tokens are replicated for each audio chunk.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, tokenizer: TokenizerSpec, return_cuts: bool = False, enable_chunking: bool = False):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.return_cuts = return_cuts
        self.enable_chunking = enable_chunking

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        def _tokens_from_cut(cut):
            return  torch.cat(
                        [
                            torch.as_tensor(s.tokens if hasattr(s, "tokens") else self.tokenizer(s.text or "", s.language))
                            for s in cut.supervisions
                        ],
                        dim=0,
                    )

        base_tokens = [_tokens_from_cut(cut) for cut in cuts]
        if self.enable_chunking:
            # Avoid circular imports
            from nemo.collections.asr.parts.utils.chunking_utils import chunk_audio_sample
            audio, audio_lens= chunk_audio_sample(audio=audio, audio_lens=audio_lens, chunk_range=[240, 300])
            tokens = base_tokens * len(audio)
            # This check will allow to gather the audio from different batches
            if cuts[0].start != 0:
                cuts[0].id = cuts[0].id + '_cut_segmented'
        else:
            tokens = base_tokens

        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        if self.return_cuts:
            return audio, audio_lens, tokens, token_lens, cuts.drop_in_memory_data()
        return audio, audio_lens, tokens, token_lens
