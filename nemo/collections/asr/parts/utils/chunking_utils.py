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

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import get_segment_offsets, get_words_offsets
from nemo.utils import logging


def should_enable_chunking(
    audio: Union[str, List[str], np.ndarray, torch.Tensor, DataLoader],
    *,
    enable_chunking: bool,
    batch_size: int,
    override_batch_size: Optional[int] = None,
    allow_tensor_input: bool = False,
    tensor_input_warning: str = "Chunking is not supported when providing tensors directly to `transcribe`. Disabling chunking.",
    dependency_available: bool = True,
    dependency_warning: Optional[str] = None,
    disabled_warning: Optional[str] = "Chunking is disabled. Please pass a single audio file or set batch_size to 1",
    manifest_extensions: Tuple[str, ...] = ("json", "jsonl"),
) -> bool:
    """
    Determine whether chunking should be enabled for a transcription request.

    Args:
        audio: Original audio input passed to ``transcribe``.
        enable_chunking: Initial user request to enable chunking.
        batch_size: Batch size argument passed to ``transcribe``.
        override_batch_size: Optional batch size taken from override config that supersedes ``batch_size``.
        allow_tensor_input: Whether tensor or dataloader inputs are compatible with chunking.
        tensor_input_warning: Warning message emitted when tensor inputs disable chunking.
        dependency_available: Whether required dependencies for chunking are available (e.g., timestamps model).
        dependency_warning: Optional warning message when dependencies are missing.
        disabled_warning: Warning emitted when chunking requirements are not satisfied.
        manifest_extensions: Extensions treated as manifest files.

    Returns:
        bool: True if chunking should be enabled, False otherwise.
    """
    if not enable_chunking:
        return False

    if not allow_tensor_input and _is_tensor_like(audio):
        if tensor_input_warning:
            logging.warning(tensor_input_warning)
        return False

    if not dependency_available:
        if dependency_warning:
            logging.warning(dependency_warning)
        return False

    is_one_audio = _is_single_audio_input(audio, manifest_extensions)
    effective_batch_size = override_batch_size if override_batch_size is not None else batch_size
    chunking_enabled = is_one_audio or effective_batch_size == 1

    if not chunking_enabled and disabled_warning:
        logging.warning(disabled_warning)

    return chunking_enabled


def _is_tensor_like(audio: Union[np.ndarray, torch.Tensor, DataLoader, Iterable]) -> bool:
    if isinstance(audio, (np.ndarray, torch.Tensor, DataLoader)):
        return True

    if isinstance(audio, (list, tuple)) and audio:
        return isinstance(audio[0], (np.ndarray, torch.Tensor))

    return False


def _is_single_audio_input(
    audio: Union[str, List[str], Tuple[str, ...]], manifest_extensions: Tuple[str, ...]
) -> bool:
    if isinstance(audio, str):
        if audio.endswith(manifest_extensions):
            try:
                with open(audio, "r", encoding="utf-8") as manifest_f:
                    non_empty = 0
                    for line in manifest_f:
                        if line.strip():
                            non_empty += 1
                            if non_empty > 1:
                                break
                    return non_empty == 1
            except OSError as e:
                logging.warning(f"Failed to inspect manifest '{audio}' for chunking: {e}")
                return False
        return True

    if isinstance(audio, (list, tuple)):
        return len(audio) == 1

    return False


def find_optimal_chunk_size(
    total_len: int,
    min_sec: int = 30,
    max_sec: int = 40,
    sample_rate: int = 16000,
    overlap_sec: float = 1.0,
) -> int:
    """
    Determine the chunk size (in samples) that minimizes padding for the final chunk.
    """
    if total_len < max_sec * sample_rate:
        return total_len

    best_chunk_size = min_sec * sample_rate
    best_last_chunk_len = 0
    overlap_size = int(overlap_sec * sample_rate)

    for sec in range(min_sec, max_sec + 1):
        candidate = sec * sample_rate
        step_size = candidate - overlap_size

        if step_size <= 0 or candidate > total_len:
            continue

        n_chunks = (total_len + step_size - 1) // step_size
        last_chunk_len = total_len - step_size * (n_chunks - 1)

        if last_chunk_len > best_last_chunk_len:
            best_last_chunk_len = last_chunk_len
            best_chunk_size = candidate

    return best_chunk_size


def chunk_waveform(
    waveform: torch.Tensor,
    chunk_range: Optional[List],
    overlap_sec: float = 1.0,
    sample_rate: int = 16000,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Split a single waveform into overlapping chunks and record each chunk length.
    """
    total_len = waveform.shape[0]
    if chunk_range is None:
        chunk_size = find_optimal_chunk_size(
            total_len=total_len,
            sample_rate=sample_rate,
            overlap_sec=overlap_sec,
        )
    else:
        if not isinstance(chunk_range, List) or len(chunk_range) != 2:
            raise ValueError("Chunk size should be list with the minimum and maximum length of the chunk.")
        chunk_size = find_optimal_chunk_size(
            total_len=total_len,
            min_sec=chunk_range[0],
            max_sec=chunk_range[1],
            sample_rate=sample_rate,
            overlap_sec=overlap_sec,
        )

    if chunk_size >= total_len:
        return [waveform], [total_len]

    overlap_size = int(overlap_sec * sample_rate)
    step_size = chunk_size - overlap_size

    if step_size <= 0:
        raise ValueError("chunk_size must be greater than the overlap size.")

    chunks: List[torch.Tensor] = []
    chunk_lens: List[int] = []
    start = 0

    while start + overlap_size < total_len:
        end = min(start + chunk_size, total_len)
        chunk = waveform[start:end]
        length = chunk.shape[0]

        if length < chunk_size:
            pad = torch.zeros(chunk_size - length, dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, pad], dim=0)

        chunks.append(chunk)
        chunk_lens.append(length)
        start += step_size

    return chunks, chunk_lens


def chunk_audio_sample(
    audio: torch.Tensor,
    audio_lens: torch.Tensor,
    chunk_range: Optional[List] = None,
    overlap_sec: float = 1.0,
    sample_rate: int = 16000,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Chunk every waveform in ``audio`` and stack the result.

    Returns:
        chunked_audio: Tensor containing all chunks stacked along batch dim.
        chunked_lens: Tensor with the true (unpadded) length of each chunk.
        chunks_per_sample: Number of chunks produced for each original sample.
    """
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise ValueError("chunk_audio_samples expects audio shaped (batch, time) with batch > 0.")

    if audio.shape[0] != 1:
        raise ValueError("chunk_audio_samples currently expects batch size 1.")

    waveform = audio[0, : audio_lens[0]]
    sample_chunks, sample_lengths = chunk_waveform(
        waveform=waveform,
        chunk_range=chunk_range,
        overlap_sec=overlap_sec,
        sample_rate=sample_rate,
    )

    if not sample_chunks:
        return audio, audio_lens

    stacked_chunks = torch.stack(sample_chunks, dim=0)
    stacked_lengths = torch.as_tensor(sample_lengths, dtype=audio_lens.dtype, device=audio_lens.device)
    return stacked_chunks, stacked_lengths


def merge_parallel_chunks(
    hypotheses,
    encoded_len,
    model,
    timestamps,
    subsampling_factor,
    window_stride,
    decoding,
    timestamps_type=None,
    tokenizer=None,
):
    """
    Merges hypotheses from parallel chunks into a single hypothesis with proper text,
    token sequences, and timestamps.

    Args:
        hypotheses: List of Hypothesis objects from each chunk
        encoded_len: Tensor of encoded lengths for each chunk to use for finding offsets
        model: The ASR model instance (needed for LCS alignment)
        timestamps: Timestamps generation is enabled
        subsampling_factor: The encoder's subsampling factor
        window_stride: The preprocessor's window stride
        decoding: The decoding instance for converting tokens to text
        tokenizer: Optional tokenizer to use when normalizing timestamp entries.
            Defaults to ``model.tokenizer`` when not provided.

    Returns:
        Hypothesis: A single merged hypothesis with combined text, tokens, and timestamps
    """
    # we take the overlap to be 1 second, and count number of tokens in it
    delay = int(1 / (subsampling_factor / 100))
    # Merge tokens from character level timestamps if timestamps are enabled.
    tokenizer = tokenizer or getattr(model, 'tokenizer', None)

    timestamps_requested = bool(timestamps)

    if timestamps_requested:
        if tokenizer is None:
            raise ValueError("Tokenizer is required when timestamps are enabled.")

        # Function to normalize tokens from TDT/RNNT
        def ensure_char_token(entry):
            char_value = entry.get('char', '')
            if isinstance(char_value, List):
                char_value = char_value[0] if char_value else ''
                entry['char'] = char_value

            token_id = entry.get('token_id')
            if isinstance(token_id, (list, tuple)):
                entry['token_id'] = token_id[0]

            if 'token' not in entry or entry['token'] is None:
                token = tokenizer.ids_to_tokens(token_id)
                entry['token'] = token[1] if len(token) > 1 else token[0]
            return entry

        if hypotheses[0].timestamp['char']:
            merged_tokens = []
            for char in hypotheses[0].timestamp['char']:
                char = ensure_char_token(char)
                merged_tokens.append(char['token_id'])
        else:
            if hypotheses[0].text != '':
                logging.warning("Cannot provide reliable timestamps for the current audio file.")
            merged_tokens = hypotheses[0].y_sequence.tolist()
    else:
        merged_tokens = hypotheses[0].y_sequence.tolist()
    # avoid circular import
    from nemo.collections.asr.parts.utils.streaming_utils import lcs_alignment_merge_buffer

    for i in range(1, len(hypotheses)):
        if timestamps_requested:

            if hypotheses[i].timestamp['char']:
                data = []
                for char in hypotheses[i].timestamp['char']:
                    char = ensure_char_token(char)
                    data.append(char['token_id'])
            else:
                if hypotheses[0].text != '':
                    logging.warning("Cannot provide reliable timestamps for the current audio file.")
                data = hypotheses[i].y_sequence.tolist()
        else:
            data = hypotheses[i].y_sequence.tolist()
        merged_tokens = lcs_alignment_merge_buffer(
            buffer=merged_tokens,
            data=data[: int(delay * 0.6)],  # only approximately 60% of the tokens are non blank
            delay=delay,
            model=model,
            max_steps_per_timestep=2,
            min_lcs_length=1,
            parallel_chunking=True,
        )
        merged_tokens += data[int(delay * 0.6) :]

    # Convert merged tokens to text
    final_text = decoding.decode_tokens_to_str(merged_tokens)
    merged_hypotheses = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp=([] if not timestamps_requested else {'word': [], 'segment': []}),
    )
    merged_hypotheses.y_sequence = torch.tensor(merged_tokens)
    merged_hypotheses.text = final_text
    # Merge timestamps and add word and segment level timestamps
    if timestamps_requested:
        chunk_offsets = [0] + [
            (x * subsampling_factor - 100) if i >= 1 else (x * subsampling_factor)
            for i, x in enumerate(encoded_len.tolist(), start=1)
        ]
        merged_hypotheses = join_timestamp_and_add_word_and_segment_level_timestamps(
            merged_hypotheses,
            hypotheses,
            chunk_offsets,
            subsampling_factor,
            window_stride,
            decoding,
            merged_tokens,
            timestamps_type,
        )

    return merged_hypotheses


def update_timestamps(hypotheses, decoding, tokenizer=None, timestamps_type=None):
    """
    Generate word and segment timestamps from character timestamps.

    Args:
        char_timestamps: Character-level timestamp data
        merged_hypotheses: Hypothesis to update with timestamps
        decoding: Decoding instance for token-to-string conversion

    Returns:
        Hypothesis: Updated merged_hypotheses with word and segment timestamps
    """
    # Create encoded_char_offsets for word/segment generation
    char_timestamps = hypotheses.timestamp['char']
    encoded_char_offsets = []
    for char_offset in char_timestamps:
        enc_char_offset = char_offset.copy()
        token = enc_char_offset.get('token', None)
        enc_char_offset['char'] = token if token is not None else tokenizer.ids_to_tokens(enc_char_offset['token_id'])
        encoded_char_offsets.append(enc_char_offset)

    # Generate word-level timestamps from combined char timestamps

    word_offsets = get_words_offsets(
        char_offsets=char_timestamps,
        decode_tokens_to_str=decoding.decode_tokens_to_str,
        encoded_char_offsets=encoded_char_offsets,
        supported_punctuation={',', '.', '!', '?'},
    )
    # Generate segment-level timestamps from word timestamps
    segment_offsets = get_segment_offsets(word_offsets=word_offsets, segment_delimiter_tokens={'.', '!', '?', "..."})
    # Update the merged hypothesis with word and segment timestamps
    if timestamps_type is not None:
        if 'word' in timestamps_type or 'all' in timestamps_type:
            hypotheses.timestamp['word'] = word_offsets
        if 'segment' in timestamps_type or 'all' in timestamps_type:
            hypotheses.timestamp['segment'] = segment_offsets
        if 'char' in timestamps_type or 'all' in timestamps_type:
            hypotheses.timestamp['char'] = char_timestamps
    else:
        hypotheses.timestamp['word'] = word_offsets
        hypotheses.timestamp['segment'] = segment_offsets
    return hypotheses


def join_timestamp_and_add_word_and_segment_level_timestamps(
    merged_hypotheses,
    hypotheses,
    chunk_offsets,
    subsampling_factor,
    window_stride,
    decoding,
    merged_tokens=None,
    timestamps_type=None,
):
    """
    Combine character-level timestamps from chunks and generate word/segment timestamps.

    Args:
        merged_hypotheses: Target hypothesis to update with timestamps
        hypotheses: List of hypotheses from different chunks
        chunk_offsets: Frame offsets for each chunk
        subsampling_factor: Subsampling factor of the encoder
        window_stride: Time stride per frame in seconds
        decoding: Decoding that is used for decoding tokens into text in `get_words_offsets`
        merged_tokens: Optional token sequence for filtering (default: None)

    Returns:
        Hypothesis: Updated merged_hypotheses with word and segment timestamps
    """

    # First, combine char-level timestamps from all chunks
    char_timestamps = join_char_level_timestamps(
        hypotheses, chunk_offsets, subsampling_factor, window_stride, merged_tokens
    )
    merged_hypotheses.timestamp['char'] = char_timestamps

    return update_timestamps(merged_hypotheses, decoding, timestamps_type)


def join_char_level_timestamps(
    hypotheses,
    chunk_offsets,
    subsampling_factor,
    window_stride,
    merged_tokens=None,
):
    """
    Merge per-chunk character-level timestamps into a single global timeline.

    This function stitches together character timestamp dictionaries coming from
    consecutive chunks of the same audio. It shifts each chunk's offsets into a
    global frame-of-reference and converts subsampled frame offsets to seconds.

    Args:
        hypotheses: List of hypotheses.
        chunk_offsets: List of raw-frame offsets (one per chunk) used for shifting.
        subsampling_factor: Encoder subsampling factor (int). Number of raw
            frames per one subsampled step.
        window_stride: Time (in seconds) per raw input frame (float).
        merged_tokens: Optional list of global token ids. If provided, only
            characters whose `token_id` matches the next id in this list are
            retained; leading overlapped characters within a chunk are trimmed.

    Returns:
        List[dict]: Character timestamp dicts placed on a global timeline
    """
    char_timestamps = []
    cumulative_offset = 0  # raw (pre-subsampling) frames already emitted
    j_token = 0  # cursor in merged_tokens

    subsamp = subsampling_factor
    stride = window_stride  # sec per raw frame
    for i, h in enumerate(hypotheses):
        chunk_frame_offset = chunk_offsets[i] // subsamp
        cumulative_offset += chunk_frame_offset

        # 1) figure out how much of the *front* of this chunk we will drop
        for char in h.timestamp['char']:
            if not char:
                continue
            current_token_id = char['token_id']
            keep = merged_tokens is None or (
                j_token < len(merged_tokens) and current_token_id == merged_tokens[j_token]
            )
            if not keep:
                continue
            # adjust offsets: chunk start + global chunk shift − total removed
            upd = dict(char)
            if char['start_offset'] != -1:
                upd['start_offset'] = char['start_offset'] + cumulative_offset  # place chunk globally
            if char['end_offset'] != -1:
                upd['end_offset'] = char['end_offset'] + cumulative_offset

            if char_timestamps:
                if upd['start_offset'] != -1 and upd['start_offset'] < char_timestamps[-1]['end_offset']:
                    upd['start_offset'] = char_timestamps[-1]['end_offset']
                    upd['end_offset'] = char_timestamps[-1]['end_offset']
            # convert to seconds
            upd['start'] = -1 if upd['start_offset'] == -1 else upd['start_offset'] * stride * subsamp
            upd['end'] = -1 if upd['end_offset'] == -1 else upd['end_offset'] * stride * subsamp

            char_timestamps.append(upd)
            j_token += 1

    return char_timestamps


def _normalize_hypothesis_group_id(hypothesis_id: str) -> str:
    """
    Normalize hypothesis IDs so that segmented continuations share the same group ID.

    IDs ending with `_cut_segmented` represent continuations of the chunk whose ID
    shares the same prefix but ends with `-0`. Only the substring after the final
    `-` is replaced so prefixes containing additional dashes remain unchanged.
    """
    if not isinstance(hypothesis_id, str):
        return hypothesis_id
    if 'cut_segmented' not in hypothesis_id:
        return hypothesis_id

    base_id = hypothesis_id.split('_cut_segmented', 1)[0]
    if '-' not in base_id:
        return base_id

    prefix, _ = base_id.rsplit('-', 1)
    if not prefix:
        return base_id

    return f'{prefix}-0'


def merge_all_hypotheses(
    hypotheses_list, timestamps, subsampling_factor, chunk_duration_seconds=3600, timestamps_type=None
):
    """
    Group hypotheses by ID and merge each group into a single hypothesis.

    Args:
        hypotheses_list: List of hypothesis objects with 'id' attributes
        timestamps: True if timestamps generation is enabled
        subsampling_factor: Subsampling factor of the encoder
        chunk_duration_seconds: Duration of each chunk in seconds (default: 3600)

    Returns:
        List[Hypothesis]: List of merged hypotheses, one per unique ID
    """
    same_audio_hypotheses = []
    all_merged_hypotheses = []
    prev_id = None
    for h in hypotheses_list:

        # This will form the current ids of the same audio file
        current_id = _normalize_hypothesis_group_id(h.id)

        # If this is a new ID (different from previous), process the accumulated hypotheses
        if prev_id is not None and current_id != prev_id:
            if same_audio_hypotheses:  # Only merge if we have hypotheses to merge

                all_merged_hypotheses.append(
                    merge_hypotheses_of_same_audio(
                        same_audio_hypotheses, timestamps, subsampling_factor, chunk_duration_seconds, timestamps_type
                    )
                )
            same_audio_hypotheses = []

        # Add current hypothesis to the group
        same_audio_hypotheses.append(h)
        prev_id = current_id

    # Process the final group of hypotheses
    if same_audio_hypotheses:
        all_merged_hypotheses.append(
            merge_hypotheses_of_same_audio(
                same_audio_hypotheses, timestamps, subsampling_factor, chunk_duration_seconds, timestamps_type
            )
        )
    return all_merged_hypotheses


def merge_hypotheses_of_same_audio(
    hypotheses_list, timestamps, subsampling_factor, chunk_duration_seconds=3600, timestamps_type=None
):
    """
    Merge hypotheses from the same audio source into a single hypothesis.
    Used for combining results when long audio is split into hour-long segments
    processed as separate batches.

    Args:
        hypotheses_list: List of hypothesis objects from time chunks
        timestamps: True if timestamps generation is enabled
        subsampling_factor: Subsampling factor of the encoder
        chunk_duration_seconds: Duration of each chunk in seconds (default: 3600)

    Returns:
        Hypothesis: Single merged hypothesis
    """

    # Create merged hypothesis with empty initial values
    if timestamps:
        if timestamps_type and 'all' in timestamps_type:
            timestamp_dict = {'char': [], 'word': [], 'segment': []}
        elif timestamps_type:
            timestamp_dict = {timestamps_type[0]: []}
        else:
            timestamp_dict = {'word': [], 'segment': []}
    else:
        timestamp_dict = []

    merged_hypothesis = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp=timestamp_dict,
    )
    merged_hypothesis.y_sequence = torch.cat([h.y_sequence for h in hypotheses_list])

    # Create final text by joining text from all hypotheses
    text_parts = []
    for hyp in hypotheses_list:
        if hyp.text:
            text_parts.append(hyp.text.strip())
    merged_hypothesis.text = ' '.join(text_parts)

    # Handle timestamps with proper time offsets (word and segment only)
    if timestamps and len(hypotheses_list) > 0 and getattr(hypotheses_list[0], "timestamp", {}):
        # Calculate time offsets for each chunk (in seconds)
        merged_word_timestamps = []
        merged_segment_timestamps = []
        merged_char_timestamps = []

        for chunk_idx, hyp in enumerate(hypotheses_list):
            if not hasattr(hyp, 'timestamp') or not hyp.timestamp:
                continue

            # Time offset for this chunk
            time_offset = chunk_idx * chunk_duration_seconds
            # Frame offset for this chunk (convert time to frames)
            frame_offset = int(time_offset * 1000 / subsampling_factor)

            # Merge word timestamps with offset
            if 'word' in hyp.timestamp and hyp.timestamp['word']:
                for word_info in hyp.timestamp['word']:
                    if isinstance(word_info, dict):
                        adjusted_word = word_info.copy()
                        # Adjust start and end times
                        if (
                            'start' in adjusted_word
                            and adjusted_word['start'] is not None
                            and adjusted_word['start'] != -1
                        ):
                            adjusted_word['start'] += time_offset
                        if 'end' in adjusted_word and adjusted_word['end'] is not None and adjusted_word['end'] != -1:
                            adjusted_word['end'] += time_offset
                        # Adjust start and end offsets (frame counts)
                        if (
                            'start_offset' in adjusted_word
                            and adjusted_word['start_offset'] is not None
                            and adjusted_word['start_offset'] != -1
                        ):
                            adjusted_word['start_offset'] += frame_offset
                        if (
                            'end_offset' in adjusted_word
                            and adjusted_word['end_offset'] is not None
                            and adjusted_word['end_offset'] != -1
                        ):
                            adjusted_word['end_offset'] += frame_offset
                        merged_word_timestamps.append(adjusted_word)
                    else:
                        merged_word_timestamps.append(word_info)

            # Merge segment timestamps with offset
            if 'segment' in hyp.timestamp and hyp.timestamp['segment']:
                for segment_info in hyp.timestamp['segment']:
                    if isinstance(segment_info, dict):
                        adjusted_segment = segment_info.copy()
                        # Adjust start and end times
                        if (
                            'start' in adjusted_segment
                            and adjusted_segment['start'] is not None
                            and adjusted_segment['start'] != -1
                        ):
                            adjusted_segment['start'] += time_offset
                        if (
                            'end' in adjusted_segment
                            and adjusted_segment['end'] is not None
                            and adjusted_segment['end'] != -1
                        ):
                            adjusted_segment['end'] += time_offset
                        # Adjust start and end offsets (frame counts)
                        if (
                            'start_offset' in adjusted_segment
                            and adjusted_segment['start_offset'] is not None
                            and adjusted_segment['start_offset'] != -1
                        ):
                            adjusted_segment['start_offset'] += frame_offset
                        if (
                            'end_offset' in adjusted_segment
                            and adjusted_segment['end_offset'] is not None
                            and adjusted_segment['end_offset'] != -1
                        ):
                            adjusted_segment['end_offset'] += frame_offset
                        merged_segment_timestamps.append(adjusted_segment)
                    else:
                        merged_segment_timestamps.append(segment_info)
            # Merge char timestamps with offset (for RNNT models)
            if timestamps_type:
                if 'char' in hyp.timestamp and hyp.timestamp['char']:
                    for char_info in hyp.timestamp['char']:
                        if isinstance(char_info, dict):
                            adjusted_char = char_info.copy()
                            # Adjust start and end times
                            if (
                                'start' in adjusted_char
                                and adjusted_char['start'] is not None
                                and adjusted_char['start'] != -1
                            ):
                                adjusted_char['start'] += time_offset
                            if (
                                'end' in adjusted_char
                                and adjusted_char['end'] is not None
                                and adjusted_char['end'] != -1
                            ):
                                adjusted_char['end'] += time_offset
                            # Adjust start and end offsets (frame counts)
                            if (
                                'start_offset' in adjusted_char
                                and adjusted_char['start_offset'] is not None
                                and adjusted_char['start_offset'] != -1
                            ):
                                adjusted_char['start_offset'] += frame_offset
                            if (
                                'end_offset' in adjusted_char
                                and adjusted_char['end_offset'] is not None
                                and adjusted_char['end_offset'] != -1
                            ):
                                adjusted_char['end_offset'] += frame_offset
                            merged_char_timestamps.append(adjusted_char)
                        else:
                            merged_char_timestamps.append(char_info)

        # Set the merged timestamps
        if timestamps:
            if timestamps_type:
                merged_hypothesis.timestamp['char'] = merged_char_timestamps
            merged_hypothesis.timestamp['word'] = merged_word_timestamps
            merged_hypothesis.timestamp['segment'] = merged_segment_timestamps
    elif len(hypotheses_list) == 1 and timestamps:
        if timestamps_type:
            merged_hypothesis.timestamp['char'] = hypotheses_list[0].timestamp['char']
        merged_hypothesis.timestamp['word'] = hypotheses_list[0].timestamp['word']
        merged_hypothesis.timestamp['segment'] = hypotheses_list[0].timestamp['segment']
    return merged_hypothesis
