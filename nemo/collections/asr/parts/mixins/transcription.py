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

import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment, ChannelSelectorType
from nemo.collections.asr.parts.utils import manifest_utils
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.data.utils import move_data_to_device
from nemo.utils import logging, logging_mode
from nemo.collections.asr.parts.utils.chunking_utils import merge_all_hypotheses

TranscriptionReturnType = Union[List[str], List[Hypothesis], Tuple[List[str]], Tuple[List[Hypothesis]]]
GenericTranscriptionType = Union[List[Any], List[List[Any]], Tuple[Any], Tuple[List[Any]], Dict[str, List[Any]]]


@dataclass
class InternalTranscribeConfig:
    # Internal values
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    training_mode: bool = False
    logging_level: Optional[Any] = None

    # Preprocessor values
    dither_value: float = 0.0
    pad_to_value: int = 0

    # Scratch space
    temp_dir: Optional[str] = None
    manifest_filepath: Optional[str] = None
    

@dataclass
class TranscribeConfig:
    use_lhotse: bool = True
    batch_size: int = 4
    return_hypotheses: bool = False
    num_workers: Optional[int] = None
    channel_selector: ChannelSelectorType = None
    augmentor: Optional[DictConfig] = None
    timestamps: Optional[bool] = None  # returns timestamps for each word and segments if model supports punctuations
    verbose: bool = True
    # Utility
    partial_hypothesis: Optional[List[Any]] = None
    enable_chunking: Optional[bool] = True
    _internal: Optional[InternalTranscribeConfig] = None


def get_value_from_transcription_config(trcfg, key, default):
    """
    Utility function to get a value from the transcription config.
    If the value is not present in the transcription config, the default value is returned.

    Args:
        trcfg: A dataclass that represents the transcription config.
        key: The name of the arg to retrieve.
        default: The default value to return if the key is not present in the transcription config.

    Returns:
        The value of the key in the transcription config or the default value.
    """
    if hasattr(trcfg, key):
        return getattr(trcfg, key)
    else:
        logging.debug(
            f"Using default value of {default} for {key} because it is not present \
                in the transcription config {trcfg}."
        )
        return default


def resolve_chunking(
    audio: Union[str, List[str], np.ndarray, torch.Tensor, DataLoader],
    enable_chunking: bool,
    batch_size: int,
) -> bool:
    """
    Determine whether chunking should be enabled for transcription based on input constraints.

    Chunking is enabled when:
    - User requested it (enable_chunking=True), AND
    - Input is not a DataLoader, AND
    - Either a single audio input is provided OR batch_size is 1

    Args:
        audio: Audio input (file path, list of paths, tensor, array, or DataLoader).
        enable_chunking: Whether chunking was requested.
        batch_size: Batch size for transcription.

    Returns:
        True if chunking should be enabled, False otherwise.
    """
    if not enable_chunking:
        return False

    if isinstance(audio, DataLoader):
        return False

    if _is_single_audio(audio) or batch_size == 1:
        return True

    logging.warning("Chunking is disabled. Please pass a single audio file or set batch_size to 1.")
    return False


def _is_single_audio(
    audio: Union[str, List[str], Tuple[str, ...], np.ndarray, torch.Tensor, DataLoader]
) -> bool:
    """Check if the audio input represents a single audio sample."""
    # String input (file path or manifest)
    if isinstance(audio, str):
        if audio.endswith((".json", ".jsonl")):
            try:
                with open(audio, "r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
                    return count == 1
            except OSError as e:
                logging.warning(f"Failed to inspect manifest '{audio}': {e}")
                return False
        return True

    # List/tuple of paths or tensors
    if isinstance(audio, (list, tuple)):
        return len(audio) == 1

    # Tensor or array - single if 1D or batch dimension is 1
    if isinstance(audio, (torch.Tensor, np.ndarray)):
        return audio.ndim == 1 or audio.shape[0] == 1

    # DataLoader - cannot determine without iterating
    if isinstance(audio, DataLoader):
        return False

    return False


class TranscriptionTensorDataset(Dataset):
    """
    Dataset for transcribing audio tensors.

    Chunking:
    If `enable_chunking` is True, each audio sample is split into optimally sized chunks
    (see `chunk_audio_sample`). This is useful for long audio inputs,
    allowing the model to process them in manageable segments.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.audio_tensors = config['audio_tensors']
        self.channel_selector = config['channel_selector']
        self.augmentor_cfg = config.get('augmentor', None)
        self.sample_rate = config['sample_rate']
        self.enable_chunking = config.get('enable_chunking', False)
        self.chunk_range = config.get('chunk_range', [240, 300])
        if self.augmentor_cfg is not None:
            self.augmentor = process_augmentations(self.augmentor_cfg, global_rank=0, world_size=1)
        else:
            self.augmentor = None

        self.length = len(self.audio_tensors)

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError(f"Index {index} out of range for dataset of size {self.length}")

        return self.get_item(index)

    def __len__(self):
        return self.length

    def _pad_audio(self, samples: torch.Tensor) -> torch.Tensor:
        """Pad audio to minimum duration, matching Lhotse dataloader behavior."""
        current_len = samples.shape[0]
        if current_len >= self.pad_min_samples:
            return samples

        pad_total = self.pad_min_samples - current_len
        if self.pad_direction == 'both':
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
        elif self.pad_direction == 'left':
            pad_left = pad_total
            pad_right = 0
        else:  # right (default)
            pad_left = 0
            pad_right = pad_total
        samples = torch.nn.functional.pad(samples, (pad_left, pad_right), mode='constant', value=0.0)
        return samples

    def get_item(self, index):
        samples = self.audio_tensors[index]

        if self.augmentor is not None:
            logging.warning(
                "Audio Augmentations are being applied during inference by moving the tensor onto CPU. "
                "This is highly inefficient and therefore not recommended.",
                mode=logging_mode.ONCE,
            )

            original_dtype = samples.dtype
            samples = samples.to(device='cpu', dtype=torch.float32).numpy()
            segment = AudioSegment(
                samples, self.sample_rate, target_sr=self.sample_rate, channel_selector=self.channel_selector
            )
            samples = self.augmentor.perturb(segment)
            samples = torch.tensor(samples.samples, dtype=original_dtype)

        samples = self._pad_audio(samples)
        seq_len = torch.tensor(samples.shape[0], dtype=torch.long)

        # Typically NeMo ASR models expect the mini-batch to be a 4-tuple of (audio, audio_len, text, text_len).
        # For inference, we set text and text_len to None to not disrupt the shape of the tuple.
        return samples, seq_len, None, None


def _speech_collate_fn_with_chunking(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    pad_id: int,
    sample_rate: int = 16000,
    chunk_range: Optional[List[int]] = [240, 300],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Collate function for tensor datasets with chunking support.
    
    This function collates audio samples and applies chunking to split long audio
    into smaller overlapping segments for more efficient processing.
    
    Args:
        batch: List of tuples (audio, audio_len, text, text_len)
        pad_id: Padding ID for text sequences
        sample_rate: Audio sample rate in Hz
        chunk_range: [min_seconds, max_seconds] for chunk size. If None, uses default [240, 300].
    
    Returns:
        Tuple of (audio, audio_lens, text, text_lens) with chunked audio
    """
    # Import here to avoid circular imports
    from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
    from nemo.collections.asr.parts.utils.chunking_utils import chunk_audio_sample

    # First, use the standard collate function
    audio, audio_lens, text, text_lens = _speech_collate_fn(batch, pad_id=pad_id)

    audio, audio_lens = chunk_audio_sample(
        audio=audio,
        audio_lens=audio_lens,
        chunk_range=chunk_range,
        sample_rate=sample_rate,
    )

    return audio, audio_lens, text, text_lens


class TranscriptionMixin(ABC):
    """
    An abstract class for transcribe-able models.

    Creates a template function `transcribe()` that provides an interface to perform transcription of audio tensors or
    filepaths.

    The following abstract classes must be implemented by the subclass:

        - `_transcribe_input_manifest_processing()`:
            Process the provided input arguments (filepaths only) and return a
            config dict for the dataloader. The data loader is should generally operate on NeMo manifests.

        - `_setup_transcribe_dataloader()`:
            Setup the dataloader for transcription. Receives the output from
            `_transcribe_input_manifest_processing()`.

        - `_transcribe_forward()`:
            Implements the model's custom forward pass to return outputs that are processed by
            `_transcribe_output_processing()`.

        - `_transcribe_output_processing()`:
            Implements the post processing of the model's outputs to return the results to
            the user. The result can be a list of objects, list of list of objects, tuple of objects, tuple of list of
            objects, or a dict of list of objects.

    """

    def _is_chunking_compatible_decoding(self) -> bool:
        """
        Check if the current decoding strategy is compatible with chunking.
        
        Beam search with `return_best_hypothesis=False` returns multiple hypotheses per chunk,
        which is incompatible with the current chunking merge implementation that expects
        a single hypothesis per chunk.
        
        Returns:
            bool: True if decoding is compatible with chunking, False otherwise.
        """
        # Check if the model has a decoding attribute (RNNT/TDT models)
        if not hasattr(self, 'decoding'):
            return True
        
        # Check if decoding has an inner decoder (AbstractRNNTDecoding stores decoder as self.decoding)
        inner_decoder = getattr(self.decoding, 'decoding', None)
        if inner_decoder is None:
            return True
        
        # Check if the inner decoder has return_best_hypothesis attribute (beam decoders have this)
        return_best_hypothesis = getattr(inner_decoder, 'return_best_hypothesis', True)
        
        if not return_best_hypothesis:
            logging.warning(
                "Chunking is not compatible with beam search when `return_best_hypothesis=False`. "
                "The decoding returns multiple hypotheses per chunk, but chunking merge expects "
                "a single hypothesis per chunk. Disabling chunking. "
                "Set `decoding.beam.return_best_hypothesis=True` to enable chunking with beam search."
            )
            return False
        
        return True

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        use_lhotse: bool = True,
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        timestamps: Optional[bool] = None,
        enable_chunking: bool = True,
        override_config: Optional[TranscribeConfig] = None,
        **config_kwargs,
    ) -> GenericTranscriptionType:
        """
        Template function that defines the execution strategy for transcribing audio.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray audio array.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds.
                But it is possible to pass a few hours long file if enough GPU memory is available.
            use_lhotse: (bool) If audio is not a dataloder, defines whether to create a lhotse dataloader or a
                non-lhotse dataloader.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from
                multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set
                to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            timestamps: Optional(Bool): timestamps will be returned if set to True as part of hypothesis object
                (output.timestep['segment']/output.timestep['word']). Refer to `Hypothesis` class for more details.
                Default is None and would retain the previous state set by using self.change_decoding_strategy().
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.
            **config_kwargs: (Optional[Dict]) additional arguments to override the default TranscribeConfig.
                Note: If override_config is passed, these arguments will be ignored.

        Returns:
            Output is defined by the subclass implementation of `TranscriptionMixin._transcribe_output_processing()`.
            It can be:

                - List[str/Hypothesis]

                - List[List[str/Hypothesis]]

                - Tuple[str/Hypothesis]

                - Tuple[List[str/Hypothesis]]

                - Dict[str, List[str/Hypothesis]]
        """
        if override_config is None:
            transcribe_cfg = TranscribeConfig(
                use_lhotse=use_lhotse,
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                timestamps=timestamps,
                enable_chunking=enable_chunking,
                **config_kwargs,
            )
        else:
            if not hasattr(override_config, '_internal'):
                raise ValueError(
                    "`transcribe_cfg must have an `_internal` argument, which must be of an object of type "
                    "InternalTranscribeConfig or its subclass."
                )

            if override_config._internal is None:
                override_config._internal = InternalTranscribeConfig()

            transcribe_cfg = override_config

        # Add new internal config
        if transcribe_cfg._internal is None:
            transcribe_cfg._internal = InternalTranscribeConfig()
        else:
            # Check if internal config is valid
            if not isinstance(transcribe_cfg._internal, InternalTranscribeConfig):
                raise ValueError(
                    "`transcribe_cfg._internal` must be of an object of type InternalTranscribeConfig or "
                    "its subclass"
                )
        transcribe_cfg.enable_chunking = resolve_chunking(
            audio=audio,
            enable_chunking=transcribe_cfg.enable_chunking,
            batch_size=transcribe_cfg.batch_size,
        ) and self._is_chunking_compatible_decoding()
        # Hold the results here
        results = None  # type: GenericTranscriptionType

        try:
            generator = self.transcribe_generator(audio, override_config=transcribe_cfg)

            for processed_outputs in generator:
                # Store results
                if isinstance(processed_outputs, list):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = []

                    results.extend(processed_outputs)

                elif isinstance(processed_outputs, dict):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = processed_outputs
                    else:
                        for k, v in processed_outputs.items():
                            results[k].extend(v)

                elif isinstance(processed_outputs, tuple):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = tuple([[] for _ in processed_outputs])

                    # If nested list structure
                    if isinstance(processed_outputs[0], list):
                        for i, processed_output in enumerate(processed_outputs):
                            results[i].extend(processed_output)
                    else:
                        # If flat list structure
                        if len(processed_outputs) != len(results):
                            raise RuntimeError(
                                f"The number of elements in the result ({len(results)}) does not "
                                f"match the results of the current batch ({len(processed_outputs)})."
                            )

                        for i, processed_output in enumerate(processed_outputs):
                            results[i].append(processed_output)

                else:
                    raise NotImplementedError(
                        "Given output result for transcription is not supported. "
                        "Please return a list of results, list of list of results, "
                        "a dict of list of results, or "
                        "a tuple of list of results."
                    )
        except StopIteration:
            pass
        if transcribe_cfg.enable_chunking and isinstance(results, list):
            results = merge_all_hypotheses(
                hypotheses_list=results,
                timestamps=(override_config.timestamps if override_config is not None else timestamps),
                subsampling_factor=self.encoder.subsampling_factor if hasattr(self.encoder, 'subsampling_factor') else 8,
            )
        return results

    def transcribe_generator(self, audio, override_config: Optional[TranscribeConfig]):
        """
        A generator version of `transcribe` function.
        """
        if override_config is None:
            override_config = TranscribeConfig()

        if not hasattr(override_config, '_internal'):
            raise ValueError(
                "`transcribe_cfg must have an `_internal` argument, which must be of an object of type "
                "InternalTranscribeConfig or its subclass."
            )

        # Add new internal config
        if override_config._internal is None:
            override_config._internal = InternalTranscribeConfig()
        else:
            # Check if internal config is valid
            if not isinstance(override_config._internal, InternalTranscribeConfig):
                raise ValueError(
                    "`transcribe_cfg._internal` must be of an object of type InternalTranscribeConfig or "
                    "its subclass"
                )

        transcribe_cfg = override_config

        try:
            # Initialize and assert the transcription environment
            self._transcribe_on_begin(audio, transcribe_cfg)

            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                transcribe_cfg._internal.temp_dir = tmpdir

                # Create a DataLoader if not already present
                if not isinstance(audio, DataLoader):
                    dataloader = self._transcribe_input_processing(audio, transcribe_cfg)
                else:
                    dataloader = audio

                if hasattr(transcribe_cfg, 'verbose'):
                    verbose = transcribe_cfg.verbose
                else:
                    verbose = True

                for test_batch in tqdm(dataloader, desc="Transcribing", disable=not verbose):
                    # Move batch to device
                    test_batch = move_data_to_device(test_batch, transcribe_cfg._internal.device)
                    # Run forward pass
                    model_outputs = self._transcribe_forward(test_batch, transcribe_cfg)
                    processed_outputs = self._transcribe_output_processing(model_outputs, transcribe_cfg)

                    # clear up memory
                    del test_batch, model_outputs

                    # Yield results if generator
                    yield processed_outputs

                    del processed_outputs

        finally:
            # set mode back to its original value
            self._transcribe_on_end(transcribe_cfg)

    """
    Transcribe Execution Flow
    """

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        """
        Internal function to setup the model for transcription. Perform all setup and pre-checks here.

        Args:
            audio: Of type `GenericTranscriptionType`
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        if audio is None:
            return {}

        if isinstance(audio, (str, np.ndarray, torch.Tensor)):
            audio = [audio]

        if isinstance(audio, list) and len(audio) == 0:
            return {}

        _params = next(self.parameters())
        if trcfg._internal.device is None:
            trcfg._internal.device = _params.device

        if trcfg._internal.dtype is None:
            trcfg._internal.dtype = _params.dtype

        # Set num_workers
        num_workers = get_value_from_transcription_config(trcfg, 'num_workers', default=0)

        if num_workers is None:
            _batch_size = get_value_from_transcription_config(trcfg, 'batch_size', default=4)
            num_workers = min(_batch_size, os.cpu_count() - 1)

        # Assign num_workers if available as key in trcfg
        if hasattr(trcfg, 'num_workers'):
            trcfg.num_workers = num_workers

        # Model's mode and device
        trcfg._internal.training_mode = self.training

        # Switch model to evaluation mode
        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                trcfg._internal.dither_value = self.preprocessor.featurizer.dither
                self.preprocessor.featurizer.dither = 0.0

            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                trcfg._internal.pad_to_value = self.preprocessor.featurizer.pad_to
                self.preprocessor.featurizer.pad_to = 0

        # Switch model to evaluation mode
        self.eval()

        # Disable logging
        trcfg._internal.logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

    def _transcribe_input_processing(self, audio, trcfg: TranscribeConfig):
        """
        Internal function to process the input audio data and return a DataLoader. This function is called by
        `transcribe()` and `transcribe_generator()` to setup the input data for transcription.

        Args:
            audio: Of type `GenericTranscriptionType`
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        # Resolve chunking based on input constraints
        if isinstance(audio, (list, tuple)):
            if len(audio) == 0:
                raise ValueError("Input `audio` is empty")
        else:
            audio = [audio]

        # Check if audio is a list of strings (filepaths or manifests)
        if isinstance(audio[0], str):
            if len(audio) == 1 and audio[0].endswith('.json') or audio[0].endswith('.jsonl'):
                # Assume it is a path to a manifest file
                trcfg._internal.manifest_filepath = audio[0]
                audio = manifest_utils.read_manifest(audio[0])

            audio_files = list(audio)

            tmp_dir = trcfg._internal.temp_dir
            ds_config = self._transcribe_input_manifest_processing(audio_files, tmp_dir, trcfg)
            temp_dataloader = self._setup_transcribe_dataloader(ds_config)
            return temp_dataloader

        # Check if audio is a list of numpy or torch tensors
        elif isinstance(audio[0], (np.ndarray, torch.Tensor)):
            audio_tensors = list(audio)

            # Convert numpy tensors to torch tensors
            if any([isinstance(_tensor, np.ndarray) for _tensor in audio_tensors]):
                audio_tensors = [
                    torch.as_tensor(audio_tensor) if isinstance(audio_tensor, np.ndarray) else audio_tensor
                    for audio_tensor in audio_tensors
                ]

            tmp_dir = trcfg._internal.temp_dir
            ds_config = self._transcribe_input_tensor_processing(audio_tensors, tmp_dir, trcfg)
            temp_dataloader = self._setup_transcribe_tensor_dataloader(ds_config, trcfg)
            return temp_dataloader

        else:
            raise ValueError(
                f"Input `audio` is of type {type(audio[0])}. "
                "Only `str` (path to audio file), `np.ndarray`, and `torch.Tensor` "
                "are supported as input."
            )

    def _transcribe_input_tensor_processing(
        self, audio_tensors: List[Union[np.ndarray, torch.Tensor]], temp_dir: str, trcfg: TranscribeConfig
    ):
        """
        Internal function to process the input audio tensors and return a config dict for the dataloader.

        Args:
            audio_tensors: A list of numpy or torch tensors. The user must ensure that they satisfy the correct
                sample rate and channel format.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        # Check if sample rate is set
        sample_rate = None
        if hasattr(self, 'cfg') and 'sample_rate' in self.cfg:
            sample_rate = self.cfg.sample_rate
        elif hasattr(self, 'sample_rate'):
            sample_rate = self.sample_rate

        if sample_rate is None:
            raise RuntimeError(
                "Provided `audio` data contains numpy or torch tensors, however the class "
                "does not have `sample_rate` attribute. Please set `sample_rate` attribute to the model explicitly."
            )

        # Determine model-specific default chunk range
        # Canary models (have transf_decoder) use shorter chunks (30-40s)
        # Parakeet models use longer chunks (240-300s)
        chunk_range = [30, 40] if hasattr(self, 'transf_decoder') else [240, 300]
        ds_config = {
            'use_lhotse': get_value_from_transcription_config(trcfg, 'use_lhotse', True),
            'audio_tensors': audio_tensors,
            'batch_size': get_value_from_transcription_config(trcfg, 'batch_size', 4),
            'temp_dir': temp_dir,
            'num_workers': get_value_from_transcription_config(trcfg, 'num_workers', 0),
            'channel_selector': get_value_from_transcription_config(trcfg, 'channel_selector', None),
            'sample_rate': sample_rate,
            'enable_chunking': get_value_from_transcription_config(trcfg, 'enable_chunking', False),
            'chunk_range': chunk_range,
        }

        augmentor = get_value_from_transcription_config(trcfg, 'augmentor', None)
        if augmentor:
            ds_config['augmentor'] = augmentor

        return ds_config

    @abstractmethod
    def _transcribe_input_manifest_processing(
        self, audio_files: List[str], temp_dir: str, trcfg: TranscribeConfig
    ) -> Dict[str, Any]:
        """
        Internal function to process the input audio filepaths and return a config dict for the dataloader.

        Args:
            audio_files: A list of string filepaths for audio files, or a single string filepath for a manifest file.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        pass

    @abstractmethod
    def _setup_transcribe_dataloader(self, config: Dict) -> DataLoader:
        """
        Internal function to setup the dataloader for transcription. This function is called by
        `transcribe()` and `transcribe_generator()` to setup the input data for transcription.

        Args:
            config: A config dict that is used to setup the dataloader for transcription. It can be generated either
                by `_transcribe_input_manifest_processing()` or `_transcribe_input_tensor_processing()`.

        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        pass

    @abstractmethod
    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        """
        Internal function to perform the model's custom forward pass to return outputs that are processed by
        `_transcribe_output_processing()`.
        This function is called by `transcribe()` and `transcribe_generator()` to perform the model's forward pass.

        Args:
            batch: A batch of input data from the data loader that is used to perform the model's forward pass.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The model's outputs that are processed by `_transcribe_output_processing()`.
        """
        pass

    @abstractmethod
    def _transcribe_output_processing(self, outputs, trcfg: TranscribeConfig) -> GenericTranscriptionType:
        """
        Internal function to process the model's outputs to return the results to the user. This function is called by
        `transcribe()` and `transcribe_generator()` to process the model's outputs.

        Args:
            outputs: The model's outputs that are processed by `_transcribe_forward()`.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The output can be a list of
            objects, list of list of objects, tuple of objects, tuple of list of objects, or a dict of list of objects.
            Its type is defined in `TranscriptionReturnType`.
        """
        pass

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        """
        Internal function to teardown the model after transcription. Perform all teardown and post-checks here.

        Args:
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        # set mode back to its original value
        self.train(mode=trcfg._internal.training_mode)

        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                self.preprocessor.featurizer.dither = trcfg._internal.dither_value

            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                self.preprocessor.featurizer.pad_to = trcfg._internal.pad_to_value

        if trcfg._internal.logging_level is not None:
            logging.set_verbosity(trcfg._internal.logging_level)

    def _setup_transcribe_tensor_dataloader(self, config: Dict, trcfg: TranscribeConfig) -> DataLoader:
        """
        Internal function to setup the dataloader for transcription. This function is called by
        `transcribe()` and `transcribe_generator()` to setup the input data for transcription.

        Args:
            config: A config dict that is used to setup the dataloader for transcription. It can be generated either
                by `_transcribe_input_manifest_processing()` or `_transcribe_input_tensor_processing()`.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        dataset = TranscriptionTensorDataset(config)

        # Import collate function here to avoid circular imports
        from nemo.collections.asr.data.audio_to_text import _speech_collate_fn

        # Calculate pad id
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'pad_id'):
            pad_id = self.tokenizer.pad_id
        elif hasattr(self, 'transcribe_pad_id'):
            logging.info("Pad id is explicitly set to `model.transcribe_pad_id` = {}".format(self.transcribe_pad_id))
            pad_id = self.transcribe_pad_id
        else:
            logging.info(
                "Pad id is being set to 0 because it could not be resolved from the tokenizer. "
                "This can happen for various reasons, especially for character based models. "
                "If pad id is incorrect, please provide the pad id explicitly by setting "
                "`model.transcribe_pad_id`."
            )
            pad_id = 0

        # Choose collate function based on chunking setting
        enable_chunking = config.get('enable_chunking', False)
        if enable_chunking:
            collate_fn = partial[Tuple[Any, Any, Any | None, Any | None]](
                _speech_collate_fn_with_chunking,
                pad_id=pad_id,
                sample_rate=config['sample_rate'],
                chunk_range=config.get('chunk_range', [30, 40]),
            )
        else:
            collate_fn = partial(_speech_collate_fn, pad_id=pad_id)

        return DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )


class ASRTranscriptionMixin(TranscriptionMixin):
    """
    An abstract class for ASR models that can transcribe audio. This class is a subclass of `TranscriptionMixin` that
    implements the default implementation of common abstract methods among the speech recognition model classes.

    The following abstract classes must be implemented by the subclass:

        - _transcribe_forward():
            Implements the model's custom forward pass to return outputs that are processed by
            `_transcribe_output_processing()`.

        - _transcribe_output_processing():
            Implements the post processing of the model's outputs to return the results to
            the user. The result can be a list of objects, list of list of objects, tuple of objects, tuple of list of
    """

    def _transcribe_input_manifest_processing(
        self, audio_files: List[str], temp_dir: str, trcfg: TranscribeConfig
    ) -> Dict[str, Any]:
        """
        Internal function to process the input audio filepaths and return a config dict for the dataloader.
        Specializes to ASR models which can have Encoder-Decoder-Joint architectures.

        Args:
            audio_files: A list of string filepaths for audio files.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        with open(os.path.join(temp_dir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file in audio_files:
                if isinstance(audio_file, str):
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                    fp.write(json.dumps(entry) + '\n')
                elif isinstance(audio_file, dict):
                    fp.write(json.dumps(audio_file) + '\n')
                else:
                    raise ValueError(
                        f"Input `audio` is of type {type(audio_file)}. "
                        "Only `str` (path to audio file) or `dict` are supported as input."
                    )

        ds_config = {
            'use_lhotse': get_value_from_transcription_config(trcfg, 'use_lhotse', True),
            'paths2audio_files': audio_files,
            'batch_size': get_value_from_transcription_config(trcfg, 'batch_size', 4),
            'temp_dir': temp_dir,
            'num_workers': get_value_from_transcription_config(trcfg, 'num_workers', 0),
            'channel_selector': get_value_from_transcription_config(trcfg, 'channel_selector', None),
            'text_field': get_value_from_transcription_config(trcfg, 'text_field', 'text'),
            'lang_field': get_value_from_transcription_config(trcfg, 'lang_field', 'lang'),
            'enable_chunking': get_value_from_transcription_config(trcfg, 'enable_chunking', False),
        }

        augmentor = get_value_from_transcription_config(trcfg, 'augmentor', None)
        if augmentor:
            ds_config['augmentor'] = augmentor

        return ds_config

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        """
        Internal function to setup the model for transcription. Perform all setup and pre-checks here.

        Args:
            audio: Of type `GenericTranscriptionType`
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        super()._transcribe_on_begin(audio, trcfg)

        # Freeze the encoder and decoder modules
        if hasattr(self, 'encoder'):
            self.encoder.freeze()

        if hasattr(self, 'decoder'):
            self.decoder.freeze()

        if hasattr(self, 'joint'):
            self.joint.freeze()

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        """
        Internal function to teardown the model after transcription. Perform all teardown and post-checks here.

        Args:
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        super()._transcribe_on_end(trcfg)

        # Unfreeze the encoder and decoder modules
        if hasattr(self, 'encoder'):
            self.encoder.unfreeze(partial=True)

        if hasattr(self, 'decoder'):
            self.decoder.unfreeze(partial=True)

        if hasattr(self, 'joint'):
            self.joint.unfreeze(partial=True)

    @classmethod
    def get_transcribe_config(cls) -> TranscribeConfig:
        """
        Utility method that returns the default config for transcribe() function.

        Returns:
            A dataclass
        """
        return TranscribeConfig()
