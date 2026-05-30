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
from re import I
import warnings
from collections import defaultdict
from typing import Any
from pathlib import Path
import random

import torch
from lightning import LightningModule
from omegaconf import DictConfig, open_dict
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method
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
from typing import List, Optional
from nemo.collections.asr.models import ASRModel
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.models.salm import _resolve_audios_in_prompt, replace_placeholders_and_build_targets
from nemo.collections.speechlm2.modules import AudioPerceptionModule
from nemo.collections.speechlm2.modules.perception import AudioTranscriptionPerceptionModule
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    load_pretrained_nemo,
    move_embedding,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging
from nemo.collections.common.data.lhotse.text_adapters import TextTurn
from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.dataloader import tokenize_with_prompt
from lhotse import fastcopy
from lhotse.serialization import SequentialJsonlWriter

from nemo.collections.asr.parts.utils.aligner_utils import (
    create_encoded_char_offsets_from_timestamps,
    create_timestamps_from_dtw_path,
    dtw_alignment,
)
from nemo.collections.asr.parts.utils.timestamp_utils import get_words_offsets

from nemo.collections.asr.parts.utils.aligner_utils import dtw_alignment
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



class SALMWithAsrDecoder(LightningModule, HFHubMixin):
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
        if not hasattr(self.llm, "model") and hasattr(self.llm, "backbone"):
            type(self.llm).model = property(lambda self: self.backbone)
        if not hasattr(self.llm.model, "embed_tokens") and hasattr(self.llm.model, "embeddings"):
            self.llm.model.embed_tokens = self.llm.model.embeddings
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        setup_speech_encoder_with_asr(self, pretrained_weights=self.cfg.pretrained_weights)
        assert isinstance(self.perception, AudioTranscriptionPerceptionModule)

        # Load pretrained weights if provided
        if (init_from_path := self.cfg.get("init_from_path", None)) is not  None:
            init_from_path = "//home/ntadevosyan/models/archive/nemotron_omni/valid_+llama8b-multilayer-asr/"
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
        import pdb; pdb.set_trace()
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
        # encoded, encoded_len = self.perception.forward_encoder(
        #     input_signal=batch["audios"], input_signal_length=batch["audio_lens"]
        # )
        # asr_hyps = self.perception.transcribe_encoded(encoded=encoded, encoded_len=encoded_len)
        # # During training, we randomly drop the transcript
        # for hyp in asr_hyps:
        #     if self.training and random.random() < self.cfg.get("asr_transcript_drop_prob", 0.0):
        #         hyp.text = ""
        # asr_tokens = [
        #     torch.as_tensor(self.tokenizer.text_to_ids(f">> {hyp.text} <<" if hyp.text else ">> <<"))
        #     for hyp in asr_hyps
        # ]
        # asr_tokens_len = [at.shape[0] for at in asr_tokens]
        # asr_tokens = torch.cat(asr_tokens, dim=0).unsqueeze(0).to(self.device)
        # transcript_embs = torch.split(self.embed_tokens(asr_tokens).squeeze(0), asr_tokens_len, dim=0)
        # audio_embs, audio_emb_lens = self.perception(encoded=encoded, encoded_len=encoded_len)
        # audio_embs = [
        #     torch.cat([aemb[:aemblen], temb], dim=0)
        #     for aemb, aemblen, temb in zip(audio_embs, audio_emb_lens, transcript_embs)
        # ]

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

        # collect generations per validation set (per-rank)
        self._val_generations = defaultdict(list)

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

        # Gather and write generations to a single file per dataset (rank 0 only)
        if self.cfg.get("val_save_path", None) is not None:
            dist = torch.distributed
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                gathered = [None for _ in range(world_size)]
                dist.all_gather_object(gathered, dict(self._val_generations))
                is_global_zero = dist.get_rank() == 0
            else:
                gathered = [dict(self._val_generations)]
                is_global_zero = True

            if is_global_zero:
                merged = defaultdict(list)
                for per_rank_dict in gathered:
                    for name, items in per_rank_dict.items():
                        merged[name].extend(items)

                val_save_path = Path(self.cfg.val_save_path) / f"{self.global_step:06d}"
                val_save_path.mkdir(parents=True, exist_ok=True)
                for name, items in merged.items():
                    out_path = val_save_path / f"{name}.jsonl"
                    with SequentialJsonlWriter(out_path) as writer:
                        for obj in items:
                            writer.write(obj)

        self._val_generations.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            
            try:
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

            except Exception as e:
                # Skip the dataset if there is an error, e.g., the dataset does not have answers
                logging.warning_once(f"Error in validation step for dataset {name}: {e}")

            # Run autoregressive generation and collect results (writing happens at epoch end)
            if self.cfg.get("val_save_path", None) is not None:
                convs_no_answer = [strip_response_if_any(conv) for conv in dataset_batch["conversations"]]
                convs_no_answer = [tokenize_with_prompt(conv, self.tokenizer, self.cfg.prompt_format) for conv in convs_no_answer]
                answer_ids = self.generate(
                    prompts=left_collate_vectors([c.input_ids for c in convs_no_answer], padding_value=self.text_pad_id).to(self.device),
                    audios=dataset_batch["audios"].to(self.device, non_blocking=True),
                    audio_lens=dataset_batch["audio_lens"].to(self.device, non_blocking=True),
                    generation_config=GenerationConfig(
                        max_new_tokens=128,
                        bos_token_id=self.text_bos_id,
                        eos_token_id=[self.text_eos_id],
                        pad_token_id=self.text_pad_id,
                        do_sample=False,
                        num_beams=1,  # greedy decoding
                    ),
                )
                answer_ids = answer_ids.cpu()
                answer_ids = [parse_hyp(ans, [self.text_eos_id]) for ans in answer_ids]
                batch_answers = [self.tokenizer.ids_to_text(ans) for ans in answer_ids]
                for conv, ans in zip(convs_no_answer, batch_answers):
                    conv.turns.append(TextTurn(role="assistant", value=ans))
                    for k, v in list(conv.custom.items()):
                        if isinstance(v, torch.Tensor):
                            del conv.custom[k]
                    self._val_generations[name].append(conv.to_dict())

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)
    
    def decode_tokens_to_str(self, tokens: List[str], lang: Optional[str] = None) -> str:
        # if lang is not None:
        #     hypothesis = self.tokenizer.tokens_to_text(tokens, lang)
        # else:
        hypothesis = self.tokenizer.tokens_to_text(tokens)
        return hypothesis
    
    def filter_attention_sink_heads_preserve_layers(
        self,
        attention_matrices: torch.Tensor,
        num_edge_frames: int = 2,
        sigma_threshold: float = 2.0
    ) -> tuple[torch.Tensor, list]:
        """Corrects outlier edge frames by replacing them with average of middle frames."""
        L, H, T, F = attention_matrices.shape
        
        # Work on a copy to avoid modifying the original
        corrected_attention = attention_matrices.clone()
        
        corrected_heads_info = []
        
        # Process each head individually
        for layer_idx in range(L):
            for head_idx in range(H):
                head_attention = corrected_attention[layer_idx, head_idx]  # [T, F]
                
                # Compute norm for each frame (column)
                frame_norms = torch.norm(head_attention, p=2, dim=0)  # [F]
                
                # Split into edge and middle frames
                first_edge_norms = frame_norms[:num_edge_frames]  # First 2 frames
                last_edge_norms = frame_norms[-num_edge_frames:]   # Last 2 frames
                middle_norms = frame_norms[num_edge_frames:-num_edge_frames]  # All middle frames
                
                # Compute statistics from middle frames only
                if len(middle_norms) > 0:
                    mean_norm = middle_norms.mean()
                    std_norm = middle_norms.std()
                    
                    lower_bound = mean_norm - sigma_threshold * std_norm
                    upper_bound = mean_norm + sigma_threshold * std_norm
                    
                    # Compute average of middle frames
                    middle_frames = head_attention[:, num_edge_frames:-num_edge_frames]  # [T, middle_F]
                    avg_middle_frame = middle_frames.mean(dim=1, keepdim=True)  # [T, 1]
                    
                    outlier_frames = []
                    
                    # Check and replace first edge frames if outliers
                    for i in range(num_edge_frames):
                        if (first_edge_norms[i] < lower_bound) or (first_edge_norms[i] > upper_bound):
                            head_attention[:, i] = avg_middle_frame.squeeze()
                            outlier_frames.append(f"first_{i}")
                    
                    # Check and replace last edge frames if outliers
                    for i in range(num_edge_frames):
                        frame_idx = -num_edge_frames + i
                        if (last_edge_norms[i] < lower_bound) or (last_edge_norms[i] > upper_bound):
                            head_attention[:, frame_idx] = avg_middle_frame.squeeze()
                            outlier_frames.append(f"last_{i}")
                    
                    if outlier_frames:
                        corrected_heads_info.append({
                            'layer': layer_idx,
                            'head': head_idx,
                            'corrected_frames': outlier_frames
                        })
        
        logging.info(f"Corrected {len(corrected_heads_info)} heads with outlier edge frames")
        for info in corrected_heads_info[:10]:  # Log first 10 for brevity
            logging.info(f"  Layer {info['layer']}, Head {info['head']}: {info['corrected_frames']}")
        
        return corrected_attention, corrected_heads_info



    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor = None,
        audio_lens: torch.Tensor = None,
        generation_config: GenerationConfig = None,
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
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=self.text_bos_id,
                eos_token_id=self.text_eos_id,
                pad_token_id=self.text_pad_id,
                # output_attentions=True,
                # return_dict_in_generate=True,
            )
        # else:
        #     generation_config.output_attentions = True
        #     generation_config.return_dict_in_generate = True
        # Generate the answers using HF Generate API.
        # Note: we need to put the text embedding layer back to the LLM for processing.
        with move_embedding(self):
            #with PreSoftmaxCaptureHook(self.llm) as hook_manager:
            original_attn_impl = self.llm.config._attn_implementation
            self.llm.config._attn_implementation = 'eager'
            answer_tokens = self.llm.generate(
                **generation_inputs,
                **generation_kwargs,
                generation_config=generation_config,
            )  
        return_answer_tokens = []
        for batch_idx in range(0,len(audio_embeds)):
            #
            # import pdb; pdb.set_trace()
            new_tokens, new_token_ids = self.retokenize_with_separate_space(answer_tokens[batch_idx])
            #import pdb; pdb.set_trace()
            # #answer_tokens[batch_idx] = answer_tokens[batch_idx].unsqueeze(0)
            new_token_ids[0] = self.space_token_id
            text = self.tokenizer.ids_to_text(answer_tokens[batch_idx])
            text = text.strip('!')
            # new_tokens, new_token_ids = self.retokenize_with_separate_space_no_punctuation(text)
            # # #import pdb; pdb.set_trace()
            # new_token_ids = new_token_ids[:-1] + [220] + [128009]
            #import pdb; pdb.set_trace()
            text_embeds = self.embed_tokens(torch.tensor(new_token_ids, device=self.device))
            audio_and_text_embeds = torch.cat([audio_embeds[batch_idx].unsqueeze(0), text_embeds.unsqueeze(0)],dim=1)
            audio_and_text_attention_mask = torch.ones(1, audio_and_text_embeds.shape[1], dtype=torch.bool, device=self.device)

            #import pdb; pdb.set_trace()
            with move_embedding(self):
                if hasattr(self.llm.config, '_attn_implementation'):
                    original_attn_impl = self.llm.config._attn_implementation
                    self.llm.config._attn_implementation = 'eager'
                with PreSoftmaxCaptureHook(self.llm) as hook_manager:
                    self.llm(inputs_embeds=audio_and_text_embeds, attention_mask=audio_and_text_attention_mask,use_cache=False)

                    scores_by_token = hook_manager.get_scores_by_token_and_layer()
            num_text_tokens = len(new_token_ids)
            needed_scores = scores_by_token[0] # [batch, heads, query, key]
            #Forcing first frame to be 0
            attention_matrices = torch.stack([
                needed_scores[l][:, :, -num_text_tokens:, 1:audio_embed_lens[batch_idx]] for l in range(0, len(needed_scores))
            ], dim=0).squeeze(1)
            #import pdb; pdb.set_trace()
            #filtered_attention_matrices = self.filter_attention_sink_heads_preserve_layers(attention_matrices,num_edge_frames=4, sigma_threshold=3)
            #self._visualize_attention_steps(filtered_attention_matrices[0], base_dir='./filtered_attention', batch_idx=batch_idx)
            # attention_matrix_no_norm = self._process_attention_matrix_no_normalization(filtered_attention_matrices[0],kernel_size=(1, 1, 3))
            # self._visualize_attention_steps(attention_matrix_no_norm.unsqueeze(0).unsqueeze(0), base_dir='./sigma_3_no_norm_filtered_attention', batch_idx=batch_idx)
            attention_matrix = self._process_attention_matrix(attention_matrices)
            

            # self._visualize_attention_steps(attention_matrix.unsqueeze(0).unsqueeze(0), base_dir='./sigma_3_filtered_attention', batch_idx=batch_idx)
            #import pdb; pdb.set_trace()
            # attention_matrices = torch.stack([
            #     needed_scores[l][:, :, -num_text_tokens:, prompted_embeds.shape[0]: prompted_embeds.shape[0] + audio_embed_lens[batch_idx]] for l in range(0, len(needed_scores))
            # ], dim=0).squeeze(1)
            # attention_matrices = torch.stack([
            #     needed_scores[l][:, :, -num_text_tokens:,6: 6 + audio_embed_lens[batch_idx]] for l in range(0, len(needed_scores))
            # ], dim=0).squeeze(1)

            # attention_matrices = torch.stack([
            #     needed_scores[l][:, :, -num_text_tokens:, audio_start_index:audio_embed_lens[batch_idx]+audio_start_index]
            #     for l in range(5, len(needed_scores))
            # ], dim=0).squeeze(1)
            # if len(audios[0]) == 41440:
#             remaining_frames = torch.stack([
#     needed_scores[l][:, :, -num_text_tokens:, 6:6 + audio_embed_lens[batch_idx]-1] for l in range(0, len(needed_scores))
# ], dim=0).squeeze(1)  # Shape: [num_layers, num_heads, num_text_tokens, audio_embed_lens[batch_idx]]

            #import pdb; pdb.set_trace()
           


            # attention_matrices = torch.stack([
            #     needed_scores[l][:, :, -num_text_tokens: ,audio_start_index : audio_start_index + audio_embed_lens[batch_idx]] for l in range(0, len(needed_scores))
            # ], dim=0).squeeze(1)

            # attention_matrices = torch.stack([
            #     needed_scores[l][:, :, -num_text_tokens-1:-1, audio_start_index+num_text_tokens:audio_start_index + num_text_tokens  + audio_embed_lens[batch_idx]] for l in range(0, len(needed_scores))
            # ], dim=0).squeeze(1)
            
           # import pdb; pdb.set_trace()
            #mport pdb; pdb.set_trace()
            # num_rows = attention_matrix.shape[0]
            # even_rows = num_rows - (num_rows % 2)
            # if even_rows > 0:
            #     attention_matrix[:even_rows] = attention_matrix[:even_rows].view(-1, 2, attention_matrix.shape[1]).flip(1).reshape(even_rows, -1)

            #import pdb; pdb.set_trace()
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
                supported_punctuation={',', '.', '!', '?'},
            )

            for word in word_offsets:
                if  word['start_offset'] > 0:
                    word['start_offset'] = word['start_offset'] - 1
                    word['end_offset'] = word['end_offset'] - 1
                    word['start'] = word['start'] - 0.08
                    word['end'] = word['end'] - 0.08
            #import pdb; pdb.set_trace()
            #word_offsets
            #import pdb; pdb.set_trace()
        
            return_answer_tokens.append((answer_tokens[batch_idx].cpu(),word_offsets))
        return return_answer_tokens
                    # This contains valid (non -inf) values because audio positions can attend to all previous text
            # new_decoding_result = self.decoding.decode_predictions_tensor(
            #     encoder_hidden_states=enc_states[batch_idx].unsqueeze(0),
            #     encoder_input_mask=enc_mask[batch_idx].unsqueeze(0),
            #     decoder_input_ids=new_decoder_input_ids,
            #     return_hypotheses=trcfg.return_hypotheses,
            # )

            # new_hypotheses, new_xatt_scores = new_decoding_result
            # new_final_tensor = torch.stack([
            #     torch.stack([new_xatt_scores[step][layer][:, :,] for step in range(len(new_xatt_scores))], dim=0)
            #     for layer in range(0)
            # ], dim=0)
            # new_final_tensor = new_final_tensor.permute(2, 0, 3, 1, 4, 5).squeeze(-2) # 
            # valid_lengths = enc_mask.sum(dim=-1).long()  # Shape: [batch_size]

            # valid_len = valid_lengths[batch_idx].item()
            # # slicing each batch item to its valid length
            # new_attention_matrix = new_final_tensor[0, :, :, :, :valid_len]  # [layers, heads, decoder, valid_len]
            
            # # Optional: Visualize attention step-by-step (uncomment to enable)
            # self._visualize_attention_steps(new_attention_matrix, base_dir='./v2_attention_visualizations_retokenized', batch_idx=batch_idx)
            # # appling all the transformations to the attention matrix
            # new_attention_matrix = self._process_attention_matrix(new_attention_matrix, kernel_size=(1, 1, 3))
            # # DTW takes as an input tensor with batch dimension
            # new_dtw_input = torch.tensor(new_attention_matrix.unsqueeze(0), device=new_attention_matrix.device).double()
            # import pdb; pdb.set_trace()
            # self._visualize_avereged(new_attention_matrix.unsqueeze(0).unsqueeze(0),base_dir='./v2_attention')
            # from nemo.collections.asr.parts.utils.aligner_utils import dtw_alignment
            # new_cost, new_path = dtw_alignment(new_dtw_input, allow_vertical=True)
            # new_timestamps = create_timestamps_from_dtw_path(new_path, torch.tensor(token_ids), self.tokenizer)
            # new_encoded_char_offsets = create_encoded_char_offsets_from_timestamps(
            #     new_timestamps, torch.tensor(token_ids), self.tokenizer
            # )
            #print(f"new_path {new_path}")
            # import pdb; pdb.set_trace()
            # new_word_offsets = get_words_offsets(
            #     char_offsets=new_encoded_char_offsets,  
            #     decode_tokens_to_str=self.decoding.decode_tokens_to_str,
            #     encoded_char_offsets=new_timestamps['char'],
            #     supported_punctuation={',', '.', '!', '?'},
                
            #)
        return answer_tokens
    def _visualize_attention_steps(
        self,
        attention_matrix: torch.Tensor,
        base_dir: str = './attention_visualizations_batch',
        batch_idx: int = 0
    ):
        """
        Visualize attention matrices step-by-step through transformations.
        Saves layer-wise and head-wise plots at each step.
        
        Args:
            attention_matrix: Input tensor [layers, heads, decoder_steps, encoder_steps]
            base_dir: Base directory for saving visualizations
            batch_idx: Batch index for labeling
        """
        import os

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.ndimage import median_filter
        base_dir += f'_batch_{batch_idx}'
        L, H, T, F = attention_matrix.shape
        
        # Step 1: Raw attention (after masking)
        step_dir = os.path.join(base_dir, f'1_raw_attention')
        for layer_idx in range(L):
            for head_idx in range(H):
                os.makedirs(os.path.join(step_dir, f'layer_{layer_idx}'), exist_ok=True)
                plt.figure(figsize=(10, 6))
                plt.imshow(attention_matrix[layer_idx, head_idx].cpu().float().numpy(), 
                          cmap='viridis', aspect='auto', origin='lower')
                plt.colorbar(label='Attention Value')
                plt.xlabel('Encoder Steps (Audio Frames)')
                plt.ylabel('Decoder Steps (Text Tokens)')
                plt.title(f'Raw - Batch {batch_idx} - Layer {layer_idx} - Head {head_idx}')
                plt.savefig(os.path.join(step_dir, f'layer_{layer_idx}', f'head_{head_idx}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
        
        # Step 2: After median filter
        attention_filtered = median_filter(attention_matrix.double().cpu().numpy(), (1, 1, 1, 1))
        step_dir = os.path.join(base_dir, '2_after_median_filter')
        for layer_idx in range(L):
            for head_idx in range(H):
                os.makedirs(os.path.join(step_dir, f'layer_{layer_idx}'), exist_ok=True)
                plt.figure(figsize=(10, 6))
                plt.imshow(attention_filtered[layer_idx, head_idx], 
                          cmap='viridis', aspect='auto', origin='lower')
                plt.colorbar(label='Attention Value')
                plt.xlabel('Encoder Steps (Audio Frames)')
                plt.ylabel('Decoder Steps (Text Tokens)')
                plt.title(f'After Filter - Batch {batch_idx} - Layer {layer_idx} - Head {head_idx}')
                plt.savefig(os.path.join(step_dir, f'layer_{layer_idx}', f'head_{head_idx}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
        
        # Step 3: After softmax
        attention_softmax = torch.tensor(attention_filtered).softmax(dim=-1)
        step_dir = os.path.join(base_dir, '3_after_softmax')
        for layer_idx in range(L):
            for head_idx in range(H):
                os.makedirs(os.path.join(step_dir, f'layer_{layer_idx}'), exist_ok=True)
                plt.figure(figsize=(10, 6))
                plt.imshow(attention_softmax[layer_idx, head_idx].cpu().float().numpy(), 
                          cmap='viridis', aspect='auto', origin='lower')
                plt.colorbar(label='Attention Value')
                plt.xlabel('Encoder Steps (Audio Frames)')
                plt.ylabel('Decoder Steps (Text Tokens)')
                plt.title(f'After Softmax - Batch {batch_idx} - Layer {layer_idx} - Head {head_idx}')
                plt.savefig(os.path.join(step_dir, f'layer_{layer_idx}', f'head_{head_idx}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
        
        # Step 4: After averaging heads (per-layer only)
        attention_avg_heads = attention_softmax.mean(dim=1)  # [L, T, F]
        step_dir = os.path.join(base_dir, '4_after_avg_heads')
        for layer_idx in range(L):
            os.makedirs(os.path.join(step_dir, f'layer_{layer_idx}'), exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.imshow(attention_avg_heads[layer_idx].cpu().float().numpy(), 
                      cmap='viridis', aspect='auto', origin='lower')
            plt.colorbar(label='Attention Value')
            plt.xlabel('Encoder Steps (Audio Frames)')
            plt.ylabel('Decoder Steps (Text Tokens)')
            plt.title(f'After Avg Heads - Batch {batch_idx} - Layer {layer_idx}')
            plt.savefig(os.path.join(step_dir, f'layer_{layer_idx}', f'aggregated.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Step 5: After normalization (per-layer)
        attention_normalized = attention_avg_heads / (attention_avg_heads.norm(dim=-2, keepdim=True) + 1e-8)
        step_dir = os.path.join(base_dir, '5_after_normalization')
        for layer_idx in range(L):
            os.makedirs(os.path.join(step_dir, f'layer_{layer_idx}'), exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.imshow(attention_normalized[layer_idx].cpu().float().numpy(), 
                      cmap='viridis', aspect='auto', origin='lower')
            plt.colorbar(label='Attention Value')
            plt.xlabel('Encoder Steps (Audio Frames)')
            plt.ylabel('Decoder Steps (Text Tokens)')
            plt.title(f'After Normalization - Batch {batch_idx} - Layer {layer_idx}')
            plt.savefig(os.path.join(step_dir, f'layer_{layer_idx}', f'normalized.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {base_dir}/ - {L} layers, {H} heads per layer")


    def retokenize_with_separate_space_no_punctuation(self, text: str):
        """
        Retokenize BPE text by removing all punctuations first, then separating the
        space-prefix character from tokens.

        1. Removes all punctuation characters (preserving trailing period if present)
        2. Splits tokens whose first character matches ``self.space_prefix_char`` into
           a standalone space token and the remainder
        3. Prepends a space token and appends the EOS token

        Args:
            text: Input text string to retokenize

        Returns:
            tuple: (tokens, token_ids)
        """
        import string
        ends_with_period = text.rstrip().endswith('.')

        text_no_punct = text.translate(str.maketrans('', '', string.punctuation))

        if ends_with_period:
            text_no_punct = text_no_punct.rstrip() + '.'

        bpe_tokens = self.tokenizer.text_to_tokens(text_no_punct)
        sp = self.space_prefix_char

        processed_tokens = [sp]

        for token in bpe_tokens:
            if token.startswith(sp):
                rest_of_token = token[len(sp):]
                if rest_of_token:
                    processed_tokens.append(sp)
                    processed_tokens.append(token)
                else:
                    processed_tokens.append(sp)
            else:
                processed_tokens.append(token)

        processed_tokens.append(self.tokenizer.ids_to_tokens([self.tokenizer.eos_id])[0])
        token_ids = []
        for token in processed_tokens:
            token_ids.append(self.tokenizer.tokens_to_ids(token))
        return processed_tokens, token_ids

    def retokenize_with_separate_space(self, ids):
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

        processed_tokens = ['']
        for token, id_token in zip(bpe_tokens, ids):
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

    def _process_attention_matrix(self,
        attention_matrix:torch.Tensor,
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


    def _process_attention_matrix_no_normalization(self,
        attention_matrix:torch.Tensor,
        kernel_size: tuple[int, int, int] = (1, 1, 1),
        qk_scale_factor: float = 1.0,
        ) -> torch.Tensor:
        from scipy.ndimage import median_filter
        
        L, H, T, F = attention_matrix.shape
        # flattening with respect to layers and heads
        attention_matrix = attention_matrix.reshape(L*H, T, F)
        # applying median filter
        attention_matrix = median_filter(attention_matrix.double().cpu().numpy(), kernel_size)
        # applying softmax to the coloumns
        attention_matrix = torch.tensor(attention_matrix * qk_scale_factor).softmax(dim=-1)
        # averaging across layers and heads
        attention_matrix = attention_matrix.mean(axis=(0))
    
        return attention_matrix
    
    def _extract_audio_attention_from_scores(
        self,
        scores_by_token: list[list[torch.Tensor]],
        tokens: torch.Tensor,
        placeholder_id: int,
        audio_embed_lens: torch.Tensor | list[int],
        padding_id: int,
    ) -> list[list[torch.Tensor]]:
        """
        Extract attention scores that attend only to audio embedding positions.
        
        For the first token (prefill phase), extracts attention from the last query position.
        For subsequent tokens, extracts attention from the single new query position.
        
        Args:
            scores_by_token: list[token_idx][layer_idx] of attention scores with shape
                            [batch, num_heads, query_len, key_len]
            tokens: Original token IDs before replacement, shape [batch, seq_len]
            placeholder_id: ID of the placeholder token that gets replaced by audio
            audio_embed_lens: Tensor or list of audio embedding lengths
            padding_id: ID of padding tokens
        
        Returns:
            list[token_idx][layer_idx] of attention scores with shape 
            [batch, num_heads, num_audio_positions] containing attention from 
            the new generated token to audio positions only.
        """
        # Convert to list if tensor
        if isinstance(audio_embed_lens, torch.Tensor):
            audio_embed_lens = audio_embed_lens.tolist()
        
        # Find audio positions in the final sequence after replacement
        audio_positions = self._find_audio_positions_in_final_sequence(
            tokens, placeholder_id, audio_embed_lens, padding_id
        )
        
        # Extract attention to audio positions for each generated token
        audio_attention_scores = []
        
        for token_idx, token_layers in enumerate(scores_by_token):
            token_audio_scores = []
            
            for layer_idx, attn_scores in enumerate(token_layers):
                # attn_scores shape: [batch, num_heads, query_len, key_len]
                # For first token (prefill): query_len > 1, we want the last query position
                # For subsequent tokens: query_len = 1, we want that single position
                
                batch_size = attn_scores.shape[0]
                batch_audio_attn = []
                
                for batch_idx in range(batch_size):
                    if batch_idx < len(audio_positions) and len(audio_positions[batch_idx]) > 0:
                        audio_pos = audio_positions[batch_idx]
                        # Extract last query position attending to audio key positions
                        # [num_heads, query_len, key_len] -> [num_heads, num_audio_positions]
                        audio_attn = attn_scores[batch_idx, :, -1, audio_pos]
                        batch_audio_attn.append(audio_attn)
                
                if batch_audio_attn:
                    # Stack across batch: [batch, num_heads, num_audio_positions]
                    token_audio_scores.append(torch.stack(batch_audio_attn, dim=0))
                else:
                    token_audio_scores.append(None)
            
            audio_attention_scores.append(token_audio_scores)
        
        return audio_attention_scores
    
    def _find_audio_positions_in_final_sequence(
        self,
        tokens: torch.Tensor,
        placeholder_id: int,
        audio_embed_lens: list[int],
        padding_id: int,
    ) -> list[list[int]]:
        """
        Find positions where audio embeddings are located in the final sequence.
        
        This mimics the logic of replace_placeholders_and_build_targets to determine
        where audio embeddings end up after placeholder replacement.
        
        Args:
            tokens: Original token IDs tensor of shape [batch, seq_len]
            placeholder_id: ID of the audio placeholder token
            audio_embed_lens: List of lengths for each audio embedding
            padding_id: ID of padding tokens
        
        Returns:
            List of lists, where each inner list contains the indices of audio positions
            for that batch element in the final (after-replacement) sequence.
        """
        batch_size = tokens.shape[0]
        audio_positions = []
        audio_idx = 0
        
        for batch_idx in range(batch_size):
            positions = []
            current_pos = 0
            
            for token_idx in range(tokens.shape[1]):
                token_id = tokens[batch_idx, token_idx].item()
                
                # Skip padding tokens (they are removed by _unpad_inputs in replace_placeholders_and_build_targets)
                if token_id == padding_id:
                    continue
                
                if token_id == placeholder_id:
                    # This placeholder will be replaced by audio embeddings
                    audio_len = audio_embed_lens[audio_idx]
                    # Audio embeddings occupy positions [current_pos, current_pos + audio_len)
                    positions.extend(range(current_pos, current_pos + audio_len))
                    current_pos += audio_len
                    audio_idx += 1
                else:
                    # Regular token occupies one position
                    current_pos += 1
            
            audio_positions.append(positions)
        
        return audio_positions

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
            #self.perception.modality_adapter = fully_shard(self.perception.modality_adapter, **fsdp_config)
            #self.perception.asr.preprocessor = fully_shard(self.perception.asr.preprocessor **fsdp_config)
            #self.perception.asr.encoder = fully_shard(self.perception.asr.encoder, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            register_fsdp_forward_method(self.perception, "forward_encoder")
            # register_fsdp_forward_method(self.perception, "transcribe_encoded")

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


def setup_speech_encoder_with_asr(model: torch.nn.Module, pretrained_weights: bool = True):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel``.
    The result is assigned to ``model.perception`` attribute and is trainable.
    """
    with open_dict(model.cfg):
        model.cfg.output_dim = model.llm.config.hidden_size
    model.perception = AudioTranscriptionPerceptionModule(model.cfg.perception, model.cfg.pretrained_asr).train()

    # from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs

    # WithOptionalCudaGraphs.disable_cuda_graphs_recursive(model.perception.asr, attribute_path="decoding.decoding")


def parse_hyp(answer: torch.Tensor, eos_tokens: list[int]):
    end = torch.isin(answer, torch.tensor(eos_tokens)).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    end = end[0]
    return answer[:end]

def strip_response_if_any(
    conversation: NeMoMultimodalConversation,
) -> NeMoMultimodalConversation:
    turns = conversation.turns
    while turns[-1].role == "assistant":
        turns = turns[:-1]
    return fastcopy(conversation, turns=turns)
