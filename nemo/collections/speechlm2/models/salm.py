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
import warnings
from collections import defaultdict
from itertools import repeat
from pathlib import Path
from typing import Any, List, Optional

import torch
from lhotse import CutSet
from lightning import LightningModule
from omegaconf import DictConfig
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import GenerationConfig

from nemo.collections.asr.parts.utils.aligner_utils import (
    create_encoded_char_offsets_from_timestamps,
    create_timestamps_from_dtw_path,
    dtw_alignment,
)
from nemo.collections.asr.parts.utils.timestamp_utils import get_words_offsets, get_segment_offsets
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.modules.perception import AudioTranscriptionPerceptionModule
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, move_embedding, setup_speech_encoder
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging

class PreSoftmaxCaptureHook:
    """
    A helper class to capture attention scores BEFORE softmax is applied.
    
    This works by monkey-patching torch.nn.functional.softmax temporarily during generation.
    
    Usage:
        hook_manager = PreSoftmaxCaptureHook(model)
        hook_manager.register_hooks()
        # ... run model forward/generate ...
        pre_softmax_scores = hook_manager.get_captured_scores()
        hook_manager.remove_hooks()
    """
    def __init__(self, model):
        self.model = model
        self.captured_scores = []
        self.original_softmax = None
        self._is_active = False
        
    def patched_softmax(self, input, dim=None, _stacklevel=3, dtype=None):
        """
        Replacement for F.softmax that captures input before applying softmax.
        We filter to only capture attention-related softmax calls.
        """
        # Capture the pre-softmax scores
        # We check the shape to ensure it's likely an attention matrix
        # Typical attention shape: (batch, num_heads, seq_len, seq_len)
        if input.dim() >= 3:  # Likely attention scores
            self.captured_scores.append(input.detach().clone())
        
        # Call the original softmax
        return self.original_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    
    def register_hooks(self):
        """
        Register hooks by monkey-patching F.softmax.
        This captures ALL softmax calls during generation.
        """
        self.captured_scores = []  # Reset
        
        if not self._is_active:
            import torch.nn.functional as F
            self.original_softmax = F.softmax
            F.softmax = self.patched_softmax
            self._is_active = True
            logging.info("Pre-softmax capture hook activated (monkey-patching F.softmax)")
    
    def remove_hooks(self):
        """Remove hooks by restoring original F.softmax."""
        if self._is_active:
            import torch.nn.functional as F
            F.softmax = self.original_softmax
            self._is_active = False
            logging.info(f"Pre-softmax capture hook deactivated. Captured {len(self.captured_scores)} tensors.")
    
    def get_captured_scores(self):
        """Return the captured pre-softmax attention scores."""
        return self.captured_scores
    
    def get_structured_scores(self, num_layers=None):
        """
        Reorganize the flat list of captured scores into a structured format.
        
        During generation, scores are captured in order:
        [layer0_token0, layer1_token0, ..., layerN_token0, layer0_token1, layer1_token1, ...]
        
        Returns:
            dict: {
                'scores_by_token': list of lists, where scores_by_token[token_idx][layer_idx] 
                                   gives the pre-softmax scores for that token and layer
                'num_layers': number of layers detected
                'num_tokens': number of generated tokens
                'raw_scores': original flat list
            }
        """
        if not self.captured_scores:
            return {'scores_by_token': [], 'num_layers': 0, 'num_tokens': 0, 'raw_scores': []}
        
        total_captures = len(self.captured_scores)
        
        # Auto-detect num_layers if not provided
        if num_layers is None:
            # Try to infer from the model
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
                num_layers = self.model.config.num_hidden_layers
            else:
                # Fallback: assume it's a common value or let user specify
                logging.warning(f"Could not auto-detect num_layers. Total captures: {total_captures}")
                num_layers = total_captures  # Worst case: treat each as separate
        
        num_tokens = total_captures // num_layers
        
        # Reorganize into [token_idx][layer_idx]
        scores_by_token = []
        for token_idx in range(num_tokens):
            token_scores = []
            for layer_idx in range(num_layers):
                flat_idx = token_idx * num_layers + layer_idx
                if flat_idx < total_captures:
                    token_scores.append(self.captured_scores[flat_idx])
            scores_by_token.append(token_scores)
        
        logging.info(f"Structured scores: {num_tokens} tokens × {num_layers} layers = {total_captures} captures")
        
        return {
            'scores_by_token': scores_by_token,
            'num_layers': num_layers,
            'num_tokens': num_tokens,
            'raw_scores': self.captured_scores,
        }
    
    def get_scores_by_token_and_layer(self, num_layers=None):
        """
        Reorganize flat list into nested list: [token_idx][layer_idx].
        
        Args:
            num_layers: Number of layers in the model. If None, tries to auto-detect.
        
        Returns:
            list of lists: scores_by_token[token_idx][layer_idx] gives the 
                          pre-softmax attention scores for that token and layer.
        """
        if not self.captured_scores:
            return []
        
        # Auto-detect num_layers if not provided
        if num_layers is None:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
                num_layers = self.model.config.num_hidden_layers
            else:
                logging.warning(f"Could not auto-detect num_layers from model config")
                return []
        
        total_captures = len(self.captured_scores)
        num_tokens = total_captures // num_layers
        
        # Build nested list: [token_idx][layer_idx]
        scores_by_token = []
        for token_idx in range(num_tokens):
            token_scores = []
            for layer_idx in range(num_layers):
                flat_idx = token_idx * num_layers + layer_idx
                if flat_idx < total_captures:
                    token_scores.append(self.captured_scores[flat_idx])
            scores_by_token.append(token_scores)
        
        return scores_by_token
    
    def __enter__(self):
        """Context manager support."""
        self.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - automatically removes hooks."""
        self.remove_hooks()

class SALM(LightningModule, HFHubMixin):
    def __init__(self, cfg) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to SALM as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.audio_locator_tag = self.cfg.audio_locator_tag

        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_locator_tag]})
        self.llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights)
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        setup_speech_encoder(self, pretrained_weights=self.cfg.pretrained_weights)

        assert isinstance(self.perception, AudioTranscriptionPerceptionModule)

        # Load pretrained weights if provided
        if (init_from_path := self.cfg.get("init_from_path", None)) is not None:
            init_from_path = Path(init_from_path)
            assert init_from_path.is_dir(), "init_from_path must be a directory containing HF checkpoint"
            logging.warning(f"Loading pretrained weights from {str(init_from_path)}")
            from safetensors import safe_open

            tensors = {}
            with safe_open(init_from_path / "model.safetensors", framework="pt") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            missing_keys, unexpected_keys = self.load_state_dict(tensors, strict=False)
            logging.warning(f"Missing keys: {missing_keys}")
            logging.warning(f"Unexpected keys: {unexpected_keys}")

        maybe_install_lora(self)

        self._use_fsdp = False
        self._use_tp = False

    def _detect_space_token_info(self) -> tuple[str, int]:
        """
        Auto-detect the space prefix character and space token ID from the tokenizer.

        Different tokenizers represent leading spaces differently:
        - GPT-2 / Llama (BPE): 'Ġ' (U+0120), token ID 220
        - SentencePiece (T5, mBART): '▁' (U+2581), variable ID
        - Others may use different conventions

        Returns:
            (space_prefix_char, space_token_id)
        """
        tokens = self.tokenizer.text_to_tokens(" a")
        if tokens and tokens[0] and tokens[0][0] not in ('a', ' '):
            space_prefix_char = tokens[0][0]
        else:
            space_prefix_char = None
            for candidate in ['Ġ', '▁']:
                if self.tokenizer.tokens_to_ids([candidate])[0] != self.tokenizer.tokens_to_ids(['<unk>'])[0]:
                    space_prefix_char = candidate
                    break
            if space_prefix_char is None:
                logging.warning(
                    "Could not auto-detect space prefix character from tokenizer. "
                    "Falling back to 'Ġ'. Override space_prefix_char / space_token_id if this is wrong."
                )
                space_prefix_char = 'Ġ'

        space_tokens = self.tokenizer.text_to_tokens(" ")
        if space_tokens:
            space_token_id = self.tokenizer.tokens_to_ids(space_tokens)[0]
        else:
            space_token_id = self.tokenizer.tokens_to_ids([space_prefix_char])[0]

        logging.info(
            f"Detected space prefix char: {repr(space_prefix_char)} (U+{ord(space_prefix_char):04X}), "
            f"space token ID: {space_token_id}"
        )
        return space_prefix_char, space_token_id

    @property
    def space_prefix_char(self) -> str:
        """The character used by the tokenizer to represent a leading space (e.g. 'Ġ' or '▁')."""
        if not hasattr(self, '_space_prefix_char'):
            self._space_prefix_char, self._space_token_id = self._detect_space_token_info()
        return self._space_prefix_char

    @property
    def space_token_id(self) -> int:
        """The token ID for a standalone space token in the tokenizer's vocabulary."""
        if not hasattr(self, '_space_token_id'):
            self._space_prefix_char, self._space_token_id = self._detect_space_token_info()
        return self._space_token_id

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.embed_tokens.num_embeddings

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "the text tokenizer has no <pad> or <unk> tokens available, using id 0 for padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def audio_locator_tag_id(self) -> int:
        return self.tokenizer.token_to_id(self.audio_locator_tag)

    @property
    def token_equivalent_duration(self) -> float:
        """
        Returns the audio duration corresponding to a single frame/token at the output of ``self.perception``.
        """
        return self.perception.token_equivalent_duration

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor = None,
        cache=None,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

        |speech and text embeddings| -> |llm| -> |lm_head| -> |token ids|

        """
        # input_embeds and out: (B, T, H)
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        ans = {"logits": out['logits']}  # (B, T, text_vocab_size)
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.
        """
        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Audio embeddings: (B, T, H)
        audio_embs, audio_emb_lens = self.perception(
            input_signal=batch["audios"], input_signal_length=batch["audio_lens"]
        )
        audio_embs = [emb[:emblen] for emb, emblen in zip(audio_embs, audio_emb_lens)]
        input_ids_to_embed = torch.where(batch["input_ids"] == self.audio_locator_tag_id, 0, batch["input_ids"])
        text_embs = self.embed_tokens(input_ids_to_embed)
        input_embs, target_ids, attention_mask = replace_placeholders_and_build_targets(
            input_ids=batch["input_ids"],
            embeds=text_embs,
            padding_id=self.text_pad_id,
            placeholder_id=self.audio_locator_tag_id,
            replacements=audio_embs,
            target_ids=batch["input_ids"].where(batch["loss_mask"], -100),  # CrossEntropyLoss().ignore_index
        )
        input_embs = input_embs[:, :-1]
        attention_mask = attention_mask[:, :-1]
        target_ids = target_ids[:, 1:]

        # Combine target audio and text into a single tensor to slice them together.
        # It will also help us truncate the sequence lengths to be divisible by TP world size,
        # when TP is enabled.
        # Input ids: (B, T, K+1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_embs.shape[1] - 1) % tp_world_size) != 0:
                # Truncate some tokens from the end to make the sequence lenght shape divisible by tensor parallelism
                # world size. Otherwise, sequence parallelism will change the input shape making leading to mismatches.
                input_embs = input_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]

        return {
            "input_embeds": input_embs,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
        num_frames = (inputs["target_ids"] != -100).long().sum()
        with loss_parallel():
            loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["target_ids"].flatten(0, 1),
                    reduction="sum",
                    ignore_index=-100,
                )
                / num_frames
            )

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "target_to_input_ratio": num_frames / (B * T),
            "padding_ratio": (batch["input_ids"] != self.text_pad_id).long().sum() / batch["input_ids"].numel(),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses = defaultdict(list)
        self._partial_accuracies = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)
        self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)

        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)
        self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        self._partial_val_losses.clear()
        self._partial_accuracies.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
            num_frames = (inputs["target_ids"] != -100).long().sum()
            with loss_parallel():
                loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["logits"].flatten(0, 1),
                        inputs["target_ids"].flatten(0, 1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    / num_frames
                )

            preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
            refs = inputs["target_ids"].reshape(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            accuracy = preds.eq(refs).float().mean()

            self._partial_accuracies[name].append(accuracy)
            self._partial_val_losses[name].append(loss)

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor = None,
        audio_lens: torch.Tensor = None,
        generation_config: GenerationConfig = None,
        timestamps: bool = False,
        ground_truth_texts: list[str] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Generate LLM answers given text or mixed text+audio prompts.

        Example 1. High-level API using ``prompts`` to provide both text and audio::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [
            ...             {
            ...                 "role": "user",
            ...                 "content": f"Transcribe the following: {model.audio_locator_tag}",
            ...                 "audio": ["path/to/audio.wav"],
            ...             }
            ...         ]
            ...    ],
            ...    max_new_tokens=128,
            ... )

        You may also include a ``transformers.GenerationConfig`` object to customize decoding strategy::

            >>> answer_ids = model.generate(..., generation_config=GenerationConfig(do_sample=True, num_beams=5))

        Example 2. Lower-level API, using ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}],
            ...        [{"role": "user", "content": f"Transcribe the following in Polish: {model.audio_locator_tag}"}],
            ...    ],
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Example 3. Lower-level API, using pre-tokenized and pre-formatted ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=prompts,  # torch.Tensor, int64, of shape (batch, num_tokens)
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Inputs:
            prompts: batch of prompts Tensor or as list[dict] each in the following format
                [
                  # batch example id 0
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}]
                  # batch example id 1
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following in Polish: {model.audio_locator_tag}"}]
                ]
                "role" is LLM-specific, you can pass multiple turns as well.
                If ``prompts`` is a Tensor, we assume it was already formatted in the relevant chat template
                and tokenized with the model's tokenizer.
            audios: Optional. Time-domain audio signal zero-padded batch of shape (B, T).
                The number of audios must correspond to the number of occurrences of <audio_locator_tag> in prompts.
                Each prompt can have multiple audios.
            audio_lens: Optional. Length of each audio example.
            generation_config: Optional HuggingFace GenerationConfig object.
            ground_truth_texts: Optional list of ground-truth transcription strings (one per batch item).
                When provided together with ``timestamps=True``, these texts are tokenized and used
                for the timestamp-extraction pipeline instead of the LLM-generated tokens.
                LLM generation is skipped entirely when this is set.
            generation_kwargs: Keyword arguments passed directly to the underlying LLM's ``generate`` method.
        """
        # Encode prompt dicts into int token ids.
        if isinstance(prompts, torch.Tensor):
            tokens = prompts
        else:
            if (
                maybe_audio := _resolve_audios_in_prompt(prompts, sampling_rate=self.sampling_rate, device=self.device)
            ) is not None:
                assert (
                    audios is None and audio_lens is None
                ), "Audios cannot be provided via ``prompts`` and ``audios``/``audio_lens`` arguments simultaneously."
                audios, audio_lens = maybe_audio
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            tokens = left_collate_vectors(
                [formatter.encode_dialog(turns=prompt)["input_ids"] for prompt in prompts],
                padding_value=self.text_pad_id,
            ).to(self.device)
        if audios is not None:
            # Audio + text input for generation.
            # Prepare token embeddings and audio embeddings.
            tokens_to_embed = tokens.where(tokens != self.audio_locator_tag_id, 0)
            token_embeds = self.embed_tokens(tokens_to_embed)
            # TODO: temporary workaround to perform batch_size=1 inference for audio encoder
            #   due to accuracy issues at bs>1
            audio_embeds, audio_embed_lens = self.perception(audios, audio_lens)
            audio_embeds = [audio_embeds[i, :elen] for i, elen in enumerate(audio_embed_lens)]
            # Insert audio embeddings into relevant positions in text embeddings.
            input_embeds, _, attention_mask = replace_placeholders_and_build_targets(
                input_ids=tokens,
                embeds=token_embeds,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                replacements=audio_embeds,
                target_ids=None,
            )
            generation_inputs = {"inputs_embeds": input_embeds, "attention_mask": attention_mask}
        else:
            # Text-only generation.
            attention_mask = tokens != self.text_pad_id
            generation_inputs = {"input_ids": tokens, "attention_mask": attention_mask}
        if ground_truth_texts is not None and timestamps:
            assert audios is not None, "audios must be provided when using ground_truth_texts with timestamps"
            assert len(ground_truth_texts) == len(audio_embeds), (
                f"ground_truth_texts length ({len(ground_truth_texts)}) must match "
                f"number of audio examples ({len(audio_embeds)})"
            )
            gt_token_ids_per_item = []
            for gt_text in ground_truth_texts:
                ids = self.tokenizer.text_to_ids(gt_text)
                gt_token_ids_per_item.append(torch.tensor(ids, dtype=torch.long, device=self.device))
            answer_tokens = gt_token_ids_per_item
        else:
            if generation_config is None:
                generation_config = GenerationConfig(
                    bos_token_id=self.text_bos_id,
                    eos_token_id=self.text_eos_id,
                    pad_token_id=self.text_pad_id,
                )
            with move_embedding(self):
                original_attn_impl = self.llm.config._attn_implementation
                self.llm.config._attn_implementation = 'eager'
                answer_tokens = self.llm.generate(
                    **generation_inputs,
                    **generation_kwargs,
                    generation_config=generation_config,
                )

        if audios is None or not timestamps:
            return answer_tokens

        return_answer_tokens = []
        for batch_idx in range(0, len(audio_embeds)):
            batch_token_ids = answer_tokens[batch_idx]
            new_tokens, new_token_ids = self.retokenize_with_separate_space(batch_token_ids)
            if new_token_ids:
                new_token_ids[0] = self.space_token_id
            text = self.tokenizer.ids_to_text(batch_token_ids)
            text = text.strip('!')
            text_embeds = self.embed_tokens(torch.tensor(new_token_ids, device=self.device))
            audio_and_text_embeds = torch.cat([audio_embeds[batch_idx].unsqueeze(0), text_embeds.unsqueeze(0)],dim=1)
            audio_and_text_attention_mask = torch.ones(
                1,
                audio_and_text_embeds.shape[1],
                dtype=torch.bool,
                device=self.device,
            )

            with move_embedding(self):
                if hasattr(self.llm.config, '_attn_implementation'):
                    original_attn_impl = self.llm.config._attn_implementation
                    self.llm.config._attn_implementation = 'eager'
                with PreSoftmaxCaptureHook(self.llm) as hook_manager:
                    self.llm(inputs_embeds=audio_and_text_embeds, attention_mask=audio_and_text_attention_mask,use_cache=False)
                    scores_by_token = hook_manager.get_scores_by_token_and_layer()
            num_text_tokens = len(new_token_ids)
            needed_scores = scores_by_token[0]
            audio_len = audio_embed_lens[batch_idx]
            attention_matrices = torch.stack(
                [
                    needed_scores[layer_idx][
                        :, :, -num_text_tokens:, 1 : audio_len
                    ]
                    for layer_idx in range(len(needed_scores))
                ],
                dim=0,
            ).squeeze(1)

            attention_matrix = self._process_attention_matrix(attention_matrices)
            
            dtw_input = torch.tensor(attention_matrix.unsqueeze(0), device=attention_matrix.device).double()
            _, path = dtw_alignment(dtw_input, allow_vertical=True)
            timestamps = create_timestamps_from_dtw_path(path, torch.tensor(new_token_ids), self.tokenizer)
            encoded_char_offsets, new_char_timestamps= create_encoded_char_offsets_from_timestamps(
                timestamps, torch.tensor(new_token_ids), self.tokenizer
            )
            word_offsets = get_words_offsets(
                char_offsets=encoded_char_offsets,
                decode_tokens_to_str=self.decode_tokens_to_str,
                encoded_char_offsets=new_char_timestamps,
                supported_punctuation={",", ".", "!", "?","¿"},
            )
            for word in word_offsets:
                if  word['start_offset'] > 0:
                    word['start_offset'] = word['start_offset'] - 1
                    word['end_offset'] = word['end_offset'] - 1
                    word['start'] = word['start'] - 0.08
                    word['end'] = word['end'] - 0.08
            
            segment_offsets = get_segment_offsets(word_offsets=word_offsets, segment_delimiter_tokens={'.', '!', '?', "...", "¿"})
            return_answer_tokens.append((answer_tokens[batch_idx].cpu(), word_offsets, segment_offsets))

        return return_answer_tokens

    def decode_tokens_to_str(self, tokens: List[str], lang: Optional[str] = None) -> str:
        return self.tokenizer.tokens_to_text(tokens)

    def retokenize_with_separate_space(self, ids: torch.Tensor | list[int]):
        """
        Retokenize BPE token IDs by separating the space-prefix character from tokens.

        Splits tokens whose first character matches ``self.space_prefix_char`` into
        a standalone space token and the remainder.  Prepends an empty-string token
        (mapped to whatever ID the tokenizer assigns to ``''``).

        Args:
            ids: Token IDs (list or tensor) to retokenize

        Returns:
            tuple: (tokens, token_ids)
        """
        bpe_tokens = self.tokenizer.ids_to_tokens(ids)
        sp = self.space_prefix_char

        processed_tokens = [""]
        for token, id_token in zip(bpe_tokens, ids):
            if isinstance(id_token, torch.Tensor):
                id_token = id_token.item()
            if id_token == 0:
                continue
            if token.startswith(sp):
                rest_of_token = token[len(sp):]
                if rest_of_token:
                    processed_tokens.append(sp)
                    processed_tokens.append(token)
                else:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)

        token_ids = []
        for token in processed_tokens:
            token_ids.append(self.tokenizer.tokens_to_ids(token))
        return processed_tokens, token_ids

    def _process_attention_matrix(
        self,
        attention_matrix: torch.Tensor,
        kernel_size: tuple[int, int, int] = (1, 1, 1),
        qk_scale_factor: float = 1.0,
    ) -> torch.Tensor:
        from scipy.ndimage import median_filter
        
        L, H, T, F = attention_matrix.shape
        # flattening with respect to layers and heads
        attention_matrix = attention_matrix.reshape(L*H, T, F)
        # applying median filter
        #attention_matrix = median_filter(attention_matrix.double().cpu().numpy(), kernel_size)
        attention_matrix = attention_matrix.double().cpu().numpy()
        # applying softmax to the coloumns
        attention_matrix = torch.tensor(attention_matrix * qk_scale_factor).softmax(dim=-1)
        # averaging across layers and heads
        attention_matrix = attention_matrix.mean(axis=(0))
        #import pdb; pdb.set_trace()
        # normalizing the attention matrix
        # attention_matrix[1:,0] = 0
        # attention_matrix[0,0] = 0
        attention_matrix = attention_matrix/attention_matrix.norm(dim=-2, keepdim=True)
        
        return attention_matrix

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            # TODO: Distributing embeddings with TP in this setup is tricky
            #       because we're adding with the output of a non-parallelized
            #       speech encoder.
            # for m in (self.embed_tokens, self.embed_audio_tokens):
            #     parallelize_module(
            #         m,
            #         tp_mesh,
            #         ColwiseParallel(
            #             # input_layouts=Shard(1),
            #             # # Optional: Shard the output along the class dimension to compute the loss in parallel.
            #             # # See `loss_parallel` in `train.py`
            #             # output_layouts=Shard(1),
            #             # use_local_output=False,
            #         ),
            #     )

            # # Parallelize the first embedding and the last linear out projection
            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            # Parallelize each transformer block
            for transformer_block in llm.model.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                # Apply the plan for the current transformer block
                parallelize_module(transformer_block, tp_mesh, plan)

            parallelize_module(
                llm.lm_head,
                tp_mesh,
                ColwiseParallel(
                    input_layouts=Shard(1),
                    # Optional: Shard the output along the class dimension to compute the loss in parallel.
                    # See `loss_parallel` in `train.py`
                    output_layouts=Shard(-1),
                    use_local_output=False,
                ),
            )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1  # Hybrid-sharding not supported
            self._use_fsdp = True
            fsdp_config = {"mesh": dp_mesh}
            for idx, layer in enumerate(llm.model.layers):
                llm.model.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            llm.lm_head = fully_shard(llm.lm_head, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.text_vocab_size,
                },
                {"name": "loss_mask", "type": NeuralType(("B", "T"), MaskType()), "seq_length": "output"},
            ],
        }


def replace_placeholders_and_build_targets(
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    padding_id: int,
    placeholder_id: int,
    replacements: list[torch.Tensor],
    target_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Replaces each occurrence of the placeholder_id in input_ids with the corresponding tensor
    from the replacements list in the embeds tensor, and creates corresponding adjusted target_ids.

    Note: when padding is necessary, we apply left-padding to the examples not to introduce
        anomalies at generation time.

    Args:
      input_ids (Tensor): shape (batch, sequence_length); input token ids.
      embeds (Tensor): shape (batch, sequence_length, hidden_dim); embeddings for each token.
      padding_id (int): these IDs will be marked as ignore_index in target_ids.
      placeholder_id (int): an id to be replaced.
      replacements (list of Tensor): each Tensor has shape (L_i, hidden_dim), with L_i arbitrary.
      target_ids (Tensor): shape (batch, sequence_length); target token ids.

    Returns:
      Tuple[Tensor, Tensor, Tensor]:
        - Tensor of shape (batch, max_new_sequence_length, hidden_dim) corresponding to
          ``embeds`` after replacements.
        - Tensor of shape (batch, max_new_sequence_length) with adjusted target IDs where:
          * Original target values are preserved where input was not a placeholder or padding
          * Positions that were placeholders, padding, or added by replacements are set to -100
          Will be None if target_ids input was None.
        - Tensor of shape (batch, max_new_sequence_length) with attention padding masks
          updated to account for shape changes due to replacements.
    """
    batch_size, seq_len = input_ids.size()
    if target_ids is not None:
        assert target_ids.size() == input_ids.size(), "target_ids must have the same shape as input_ids"

    hidden_dim = embeds.size(2)
    device, dtype = embeds.device, embeds.dtype
    ignore_index = -100  # Standard ignore_index value for CrossEntropyLoss

    # Un-pad the tensors because we'll need to re-apply new padding after replacements anyway.
    input_ids, embeds, target_ids = _unpad_inputs(input_ids, embeds, target_ids, padding_id)

    output_sequences = []
    output_target_ids = []
    output_att_masks = []
    replacement_idx = 0

    for i in range(batch_size):
        # Find all placeholder positions at once using tensor operations
        placeholder_positions = (input_ids[i] == placeholder_id).nonzero(as_tuple=True)[0]

        # Handle the case with no placeholders more efficiently
        if len(placeholder_positions) == 0:
            output_sequences.append(embeds[i])

            # Start with original target_ids and replace positions where input was padding
            if target_ids is not None:
                new_target_ids = target_ids[i].clone()
                new_target_ids[input_ids[i] == padding_id] = ignore_index
                output_target_ids.append(new_target_ids)
            output_att_masks.append(input_ids[i] != padding_id)
            continue

        # Build segments between placeholders
        segments = []  # For embeddings
        target_segments = []  # For target IDs
        att_masks = []
        prev_pos = 0

        for pos in placeholder_positions:
            # Add segment before placeholder (if any)
            if pos > prev_pos:
                segments.append(embeds[i][prev_pos:pos])

                # For target IDs: keep original targets but mark positions that were padding in input
                if target_ids is not None:
                    segment_target_ids = target_ids[i][prev_pos:pos].clone()
                    segment_target_ids[segment_target_ids == padding_id] = ignore_index
                    target_segments.append(segment_target_ids)
                att_masks.append(input_ids[i][prev_pos:pos] != padding_id)

            # Add replacement for embeddings
            rep = replacements[replacement_idx]
            segments.append(rep)

            # For target IDs: all replacement positions get ignore_index
            target_segments.append(torch.full((rep.size(0),), ignore_index, dtype=torch.long, device=device))
            att_masks.append(torch.ones((rep.size(0),), dtype=torch.bool, device=device))

            replacement_idx += 1
            prev_pos = pos + 1  # Skip placeholder

        # Add remaining segment after last placeholder (if any)
        if prev_pos < seq_len:
            segments.append(embeds[i][prev_pos:seq_len])

            # For target IDs: keep original targets but mark positions that were padding in input
            if target_ids is not None:
                segment_target_ids = target_ids[i][prev_pos:seq_len].clone()
                segment_target_ids[segment_target_ids == padding_id] = ignore_index
                target_segments.append(segment_target_ids)
            att_masks.append(input_ids[i][prev_pos:seq_len] != padding_id)

        # Concatenate all segments for this example
        output_sequences.append(torch.cat(segments, dim=0))
        output_att_masks.append(torch.cat(att_masks, dim=0))
        if target_ids is not None:
            output_target_ids.append(torch.cat(target_segments, dim=0))

    # Verify all replacements were used
    if replacement_idx != len(replacements):
        raise ValueError(f"Expected {len(replacements)} replacements but used {replacement_idx}")

    # Create padded output tensors
    max_seq_length = max(seq.size(0) for seq in output_sequences)
    output = torch.zeros(batch_size, max_seq_length, hidden_dim, device=device, dtype=dtype)
    if target_ids is not None:
        new_target_ids = torch.full((batch_size, max_seq_length), ignore_index, dtype=torch.long, device=device)
    else:
        new_target_ids = None
    attention_masks = torch.zeros((batch_size, max_seq_length), dtype=torch.bool, device=device)

    if target_ids is None:
        output_target_ids = repeat(None)
    for i, (seq, tgt, att) in enumerate(zip(output_sequences, output_target_ids, output_att_masks)):
        seq_len = seq.size(0)
        output[i, -seq_len:] = seq
        if tgt is not None:
            new_target_ids[i, -seq_len:] = tgt
        attention_masks[i, -seq_len:] = att

    return output, new_target_ids, attention_masks


def _unpad_inputs(
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    target_ids: Optional[torch.Tensor],
    padding_id: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    def first_index_not_value(tensor, value):
        mask = tensor != value
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.numel() > 0:
            return indices[0].item()
        else:
            return -1

    input_ids_unpad, embeds_unpad = [], []
    target_ids_unpad = [] if target_ids is not None else None
    for i in range(input_ids.shape[0]):
        idx = first_index_not_value(input_ids[i], padding_id)
        input_ids_unpad.append(input_ids[i, idx:])
        embeds_unpad.append(embeds[i, idx:])
        if target_ids is not None:
            target_ids_unpad.append(target_ids[i, idx:])
    return input_ids_unpad, embeds_unpad, target_ids_unpad


def _resolve_audios_in_prompt(
    prompts: list[list[dict]], sampling_rate: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    from lhotse import Recording

    paths = []
    for conversation in prompts:
        for turn in conversation:
            if "audio" in turn:
                turn_audio = turn["audio"]
                if isinstance(turn_audio, (str, Path)):
                    turn_audio = [turn_audio]
                for p in turn_audio:
                    assert isinstance(p, (str, Path)), f"Invalid value under prompt key 'audio': {p}"
                    paths.append(p)
    if not paths:
        return None
    cuts = CutSet([Recording.from_file(p).to_cut() for p in paths])
    with torch.device("cpu"):  # workaround for a Lhotse issue when default device is CUDA during collation
        audio, audio_lens = cuts.resample(sampling_rate).load_audio(collate=True)
    return (
        torch.as_tensor(audio).to(device, non_blocking=True),
        torch.as_tensor(audio_lens).to(device, non_blocking=True),
    )
