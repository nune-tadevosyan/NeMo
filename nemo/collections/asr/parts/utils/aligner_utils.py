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

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Tuple
from typing import Dict
from typing import Callable
from typing import Optional
from typing import Set
from typing import Union    


from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging

BLANK_TOKEN = "<b>"
SPACE_TOKEN = "<space>"


@dataclass
class Token:
    text: str = None
    text_cased: str = None
    token_id: int = None
    token: str = None
    s_start: int = None
    s_end: int = None
    t_start: float = None
    t_end: float = None


@dataclass
class Word:
    text: str = None
    s_start: int = None
    s_end: int = None
    t_start: float = None
    t_end: float = None
    tokens: List[Token] = field(default_factory=list)


@dataclass
class Segment:
    text: str = None
    s_start: int = None
    s_end: int = None
    t_start: float = None
    t_end: float = None
    words_and_tokens: List[Union[Word, Token]] = field(default_factory=list)


@dataclass
class Utterance:
    token_ids_with_blanks: List[int] = field(default_factory=list)
    segments_and_tokens: List[Union[Segment, Token]] = field(default_factory=list)
    text: Optional[str] = None
    pred_text: Optional[str] = None
    audio_filepath: Optional[str] = None
    utt_id: Optional[str] = None
    saved_output_files: dict = field(default_factory=dict)


def is_sub_or_superscript_pair(ref_text, text):
    """returns True if ref_text is a subscript or superscript version of text"""
    sub_or_superscript_to_num = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }

    if text in sub_or_superscript_to_num:
        if sub_or_superscript_to_num[text] == ref_text:
            return True
    return False


def restore_token_case(word: str, word_tokens: List[str]) -> List[str]:

    # remove repeated "▁" and "_" from word as that is what the tokenizer will do
    while "▁▁" in word:
        word = word.replace("▁▁", "▁")

    while "__" in word:
        word = word.replace("__", "_")

    word_tokens_cased = []
    word_char_pointer = 0

    for token in word_tokens:
        token_cased = ""

        for token_char in token:
            if token_char == word[word_char_pointer]:
                token_cased += token_char
                word_char_pointer += 1

            else:
                if token_char.upper() == word[word_char_pointer] or is_sub_or_superscript_pair(
                    token_char, word[word_char_pointer]
                ):
                    token_cased += token_char.upper()
                    word_char_pointer += 1
                else:
                    if token_char == "▁" or token_char == "_":
                        if (
                            word[word_char_pointer] == "▁"
                            or word[word_char_pointer] == "_"
                            or word[word_char_pointer] == " "
                        ):
                            token_cased += token_char
                            word_char_pointer += 1
                        elif word_char_pointer == 0:
                            token_cased += token_char
                    else:
                        raise RuntimeError(
                            f"Unexpected error - failed to recover capitalization of tokens for word {word}"
                        )

        word_tokens_cased.append(token_cased)

    return word_tokens_cased


def get_char_tokens(text, model):
    tokens = []
    for character in text:
        if character in model.decoder.vocabulary:
            tokens.append(model.decoder.vocabulary.index(character))
        else:
            tokens.append(len(model.decoder.vocabulary))  # return unk token (same as blank token)

    return tokens


def _get_utt_id(audio_filepath, audio_filepath_parts_in_utt_id):
    fp_parts = Path(audio_filepath).parts[-audio_filepath_parts_in_utt_id:]
    utt_id = Path("_".join(fp_parts)).stem
    utt_id = utt_id.replace(" ", "-")  # replace any spaces in the filepath with dashes
    return utt_id


def get_utt_obj(
    text: str,
    T: int,
    model: ASRModel,
    segment_separators: Union[str, List[str]] = ['.', '?', '!', '...'],
    word_separator: Optional[str] = " ",
    audio_filepath: Optional[str] = None,
    utt_id: Optional[str] = None,
):
    """
    Function to create an Utterance object and add all necessary information to it except
        for timings of the segments / words / tokens according to the alignment - that will
        be done later in a different function, after the alignment is done.

        The Utterance object has a list segments_and_tokens which contains Segment objects and
        Token objects (for blank tokens in between segments).
        Within the Segment objects, there is a list words_and_tokens which contains Word objects and
        Token objects (for blank tokens in between words).
        Within the Word objects, there is a list tokens tokens which contains Token objects for
        blank and non-blank tokens.
        We will be building up these lists in this function. This data structure will then be useful for
        generating the various output files that we wish to save.

    Args:
        text: the text to be aligned.
        model: the ASR model to be used for alignment. It must have a `tokenizer` or `vocabulary` attribute.
        separator: a string or a list of strings, that will be used to split the text into segments.
        T: the number of frames in the log_probs.
        audio_filepath: the path to the audio file.
        utt_id: the ID of the utterance.

    Returns:
        Utterance object with the following fields:
            - text: the text to be aligned.
            - audio_filepath: the path to the audio file if provided.
            - utt_id: the ID of the utterance if provided.
            - token_ids_with_blanks: a list of token IDs with blanks.
    """

    utt = Utterance(
        text=text,
        audio_filepath=audio_filepath,
        utt_id=utt_id,
    )

    if not segment_separators:  # if separator is not defined - treat the whole text as one segment
        segments = [text.strip()]
    else:
        segment_separators = [segment_separators] if isinstance(segment_separators, str) else segment_separators
        segments = []
        last_sep_idx = -1

        for i, letter in enumerate(text):
            # check if the current letter is a separator and the next letter is a space
            # the additional space check is done to avoid splitting the words like "a.m." into segments
            next_letter = text[i + 1] if i + 1 < len(text) else ""
            if letter in segment_separators and next_letter == " ":
                segments.append(text[last_sep_idx + 1 : i + 1].strip())
                last_sep_idx = i + 1

        if last_sep_idx < len(text):
            segments.append(text[last_sep_idx + 1 :].strip())

    # remove any empty segments
    segments = [seg for seg in segments if len(seg) > 0]

    # build up lists: token_ids_with_blanks, segments_and_tokens.
    # The code for these is different depending on whether we use char-based tokens or not
    if hasattr(model, 'tokenizer'):
        if hasattr(model, 'blank_id'):
            BLANK_ID = model.blank_id
        else:
            BLANK_ID = model.tokenizer.vocab_size

        if hasattr(model.tokenizer, 'unk_id'):
            UNK_ID = model.tokenizer.unk_id
        else:
            UNK_ID = 0

        UNK_WORD = model.tokenizer.ids_to_text([UNK_ID]).strip()
        UNK_TOKEN = model.tokenizer.ids_to_tokens([UNK_ID])[0]

        utt.token_ids_with_blanks = [BLANK_ID]

        if len(text) == 0:
            return utt

        # check for # tokens being > T
        all_tokens = model.tokenizer.text_to_ids(text)
        if len(all_tokens) > T:
            logging.info(
                f"Utterance with ID: {utt_id} has too many tokens compared to the audio file duration."
                " Will not generate output alignment files for this utterance."
            )
            return utt

        # build up data structures containing segments/words/tokens
        utt.segments_and_tokens.append(
            Token(
                text=BLANK_TOKEN,
                text_cased=BLANK_TOKEN,
                token_id=BLANK_ID,
                s_start=0,
                s_end=0,
            )
        )

        segment_s_pointer = 1  # first segment will start at s=1 because s=0 is a blank
        word_s_pointer = 1  # first word will start at s=1 because s=0 is a blank

        for segment in segments:
            # add the segment to segment_info and increment the segment_s_pointer
            segment_tokens = []
            sub_segments = segment.split(UNK_WORD)
            for i, sub_segment in enumerate(sub_segments):
                sub_segment_tokens = model.tokenizer.text_to_tokens(sub_segment.strip())
                segment_tokens.extend(sub_segment_tokens)
                if i < len(sub_segments) - 1:
                    segment_tokens.append(UNK_ID)
            # segment_tokens do not contain blanks => need to muliply by 2
            # s_end needs to be the index of the final token (including blanks) of the current segment:
            # segment_s_pointer + len(segment_tokens) * 2 is the index of the first token of the next segment =>
            # => need to subtract 2

            s_end = segment_s_pointer + len(segment_tokens) * 2 - 2
            utt.segments_and_tokens.append(
                Segment(
                    text=segment,
                    s_start=segment_s_pointer,
                    s_end=s_end,
                )
            )
            segment_s_pointer = s_end + 2

            words = segment.split(word_separator) if word_separator not in [None, ""] else [segment]

            for word_i, word in enumerate(words):

                if word == UNK_WORD:
                    word_tokens = [UNK_TOKEN]
                    word_token_ids = [UNK_ID]
                    word_tokens_cased = [UNK_TOKEN]
                elif UNK_WORD in word:
                    word_tokens = []
                    word_token_ids = []
                    word_tokens_cased = []
                    for sub_word in word.split(UNK_WORD):
                        sub_word_tokens = model.tokenizer.text_to_tokens(sub_word)
                        sub_word_token_ids = model.tokenizer.text_to_ids(sub_word)
                        sub_word_tokens_cased = restore_token_case(sub_word, sub_word_tokens)

                        word_tokens.extend(sub_word_tokens)
                        word_token_ids.extend(sub_word_token_ids)
                        word_tokens_cased.extend(sub_word_tokens_cased)
                        word_tokens.append(UNK_TOKEN)
                        word_token_ids.append(UNK_ID)
                        word_tokens_cased.append(UNK_TOKEN)

                    word_tokens = word_tokens[:-1]
                    word_token_ids = word_token_ids[:-1]
                    word_tokens_cased = word_tokens_cased[:-1]
                else:
                    word_tokens = model.tokenizer.text_to_tokens(word)
                    word_token_ids = model.tokenizer.text_to_ids(word)
                    word_tokens_cased = restore_token_case(word, word_tokens)

                # add the word to word_info and increment the word_s_pointer
                word_s_end = word_s_pointer + len(word_tokens) * 2 - 2
                utt.segments_and_tokens[-1].words_and_tokens.append(
                    Word(text=word, s_start=word_s_pointer, s_end=word_s_end)
                )
                word_s_pointer = word_s_end + 2

                for token_i, (token, token_id, token_cased) in enumerate(
                    zip(word_tokens, word_token_ids, word_tokens_cased)
                ):
                    # add the text tokens and the blanks in between them
                    # to our token-based variables
                    utt.token_ids_with_blanks.extend([token_id, BLANK_ID])
                    # adding Token object for non-blank token
                    utt.segments_and_tokens[-1].words_and_tokens[-1].tokens.append(
                        Token(
                            text=token,
                            text_cased=token_cased,
                            token_id=token_id,
                            # utt.token_ids_with_blanks has the form [...., <this non-blank token>, <blank>] =>
                            # => if do len(utt.token_ids_with_blanks) - 1 you get the index of the final <blank>
                            # => we want to do len(utt.token_ids_with_blanks) - 2 to get the index of <this non-blank token>
                            s_start=len(utt.token_ids_with_blanks) - 2,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 2,
                        )
                    )

                    # adding Token object for blank tokens in between the tokens of the word
                    # (ie do not add another blank if you have reached the end)
                    if token_i < len(word_tokens) - 1:
                        utt.segments_and_tokens[-1].words_and_tokens[-1].tokens.append(
                            Token(
                                text=BLANK_TOKEN,
                                text_cased=BLANK_TOKEN,
                                token_id=BLANK_ID,
                                # utt.token_ids_with_blanks has the form [...., <this blank token>] =>
                                # => if do len(utt.token_ids_with_blanks) -1 you get the index of this <blank>
                                s_start=len(utt.token_ids_with_blanks) - 1,
                                # s_end is same as s_start since the token only occupies one element in the list
                                s_end=len(utt.token_ids_with_blanks) - 1,
                            )
                        )

                # add a Token object for blanks in between words in this segment
                # (but only *in between* - do not add the token if it is after the final word)
                if word_i < len(words) - 1:
                    utt.segments_and_tokens[-1].words_and_tokens.append(
                        Token(
                            text=BLANK_TOKEN,
                            text_cased=BLANK_TOKEN,
                            token_id=BLANK_ID,
                            # utt.token_ids_with_blanks has the form [...., <this blank token>] =>
                            # => if do len(utt.token_ids_with_blanks) -1 you get the index of this <blank>
                            s_start=len(utt.token_ids_with_blanks) - 1,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 1,
                        )
                    )

            # add the blank token in between segments/after the final segment
            utt.segments_and_tokens.append(
                Token(
                    text=BLANK_TOKEN,
                    text_cased=BLANK_TOKEN,
                    token_id=BLANK_ID,
                    # utt.token_ids_with_blanks has the form [...., <this blank token>] =>
                    # => if do len(utt.token_ids_with_blanks) -1 you get the index of this <blank>
                    s_start=len(utt.token_ids_with_blanks) - 1,
                    # s_end is same as s_start since the token only occupies one element in the list
                    s_end=len(utt.token_ids_with_blanks) - 1,
                )
            )

        return utt

    elif hasattr(model.decoder, "vocabulary"):  # i.e. tokenization is simply character-based

        BLANK_ID = len(model.decoder.vocabulary)
        SPACE_ID = model.decoder.vocabulary.index(" ")

        utt.token_ids_with_blanks = [BLANK_ID]

        # check for text being 0 length
        if len(text) == 0:
            return utt

        # check for # tokens + token repetitions being > T
        all_tokens = get_char_tokens(text, model)
        n_token_repetitions = 0
        for i_tok in range(1, len(all_tokens)):
            if all_tokens[i_tok] == all_tokens[i_tok - 1]:
                n_token_repetitions += 1

        if len(all_tokens) + n_token_repetitions > T:
            logging.info(
                f"Utterance {utt_id} has too many tokens compared to the audio file duration."
                " Will not generate output alignment files for this utterance."
            )
            return utt

        # build up data structures containing segments/words/tokens
        utt.segments_and_tokens.append(
            Token(
                text=BLANK_TOKEN,
                text_cased=BLANK_TOKEN,
                token_id=BLANK_ID,
                s_start=0,
                s_end=0,
            )
        )

        segment_s_pointer = 1  # first segment will start at s=1 because s=0 is a blank
        word_s_pointer = 1  # first word will start at s=1 because s=0 is a blank

        for i_segment, segment in enumerate(segments):
            # add the segment to segment_info and increment the segment_s_pointer
            segment_tokens = get_char_tokens(segment, model)
            utt.segments_and_tokens.append(
                Segment(
                    text=segment,
                    s_start=segment_s_pointer,
                    # segment_tokens do not contain blanks => need to muliply by 2
                    # s_end needs to be the index of the final token (including blanks) of the current segment:
                    # segment_s_pointer + len(segment_tokens) * 2 is the index of the first token of the next segment =>
                    # => need to subtract 2
                    s_end=segment_s_pointer + len(segment_tokens) * 2 - 2,
                )
            )

            # for correct calculation: multiply len(segment_tokens) by 2 to account for blanks (which are not present in segment_tokens)
            # and + 2 to account for [<token for space in between segments>, <blank token after that space token>]
            segment_s_pointer += len(segment_tokens) * 2 + 2

            words = segment.split(" ")  # we define words to be space-separated substrings
            for i_word, word in enumerate(words):

                # convert string to list of characters
                word_tokens = list(word)
                # convert list of characters to list of their ids in the vocabulary
                word_token_ids = get_char_tokens(word, model)

                # add the word to word_info and increment the word_s_pointer
                utt.segments_and_tokens[-1].words_and_tokens.append(
                    # note for s_end:
                    # word_tokens do not contain blanks => need to muliply by 2
                    # s_end needs to be the index of the final token (including blanks) of the current word:
                    # word_s_pointer + len(word_tokens) * 2 is the index of the first token of the next word =>
                    # => need to subtract 2
                    Word(text=word, s_start=word_s_pointer, s_end=word_s_pointer + len(word_tokens) * 2 - 2)
                )

                # for correct calculation: multiply len(word_tokens) by 2 to account for blanks (which are not present in word_tokens)
                # and + 2 to account for [<token for space in between words>, <blank token after that space token>]
                word_s_pointer += len(word_tokens) * 2 + 2

                for token_i, (token, token_id) in enumerate(zip(word_tokens, word_token_ids)):
                    # add the text tokens and the blanks in between them
                    # to our token-based variables
                    utt.token_ids_with_blanks.extend([token_id])
                    utt.segments_and_tokens[-1].words_and_tokens[-1].tokens.append(
                        Token(
                            text=token,
                            text_cased=token,
                            token_id=token_id,
                            # utt.token_ids_with_blanks has the form [..., <this non-blank token>]
                            # => do len(utt.token_ids_with_blanks) - 1 to get the index of this non-blank token
                            s_start=len(utt.token_ids_with_blanks) - 1,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 1,
                        )
                    )

                    if token_i < len(word_tokens) - 1:  # only add blank tokens that are in the middle of words
                        utt.token_ids_with_blanks.extend([BLANK_ID])
                        utt.segments_and_tokens[-1].words_and_tokens[-1].tokens.append(
                            Token(
                                text=BLANK_TOKEN,
                                text_cased=BLANK_TOKEN,
                                token_id=BLANK_ID,
                                # utt.token_ids_with_blanks has the form [..., <this blank token>]
                                # => do len(utt.token_ids_with_blanks) - 1 to get the index of this blank token
                                s_start=len(utt.token_ids_with_blanks) - 1,
                                # s_end is same as s_start since the token only occupies one element in the list
                                s_end=len(utt.token_ids_with_blanks) - 1,
                            )
                        )

                # add space token (and the blanks around it) unless this is the final word in a segment
                if i_word < len(words) - 1:
                    utt.token_ids_with_blanks.extend([BLANK_ID, SPACE_ID, BLANK_ID])
                    utt.segments_and_tokens[-1].words_and_tokens.append(
                        Token(
                            text=BLANK_TOKEN,
                            text_cased=BLANK_TOKEN,
                            token_id=BLANK_ID,
                            # utt.token_ids_with_blanks has the form
                            # [..., <final token of previous word>, <blank token>, <space token>, <blank token>]
                            # => do len(utt.token_ids_with_blanks) - 3 to get the index of the blank token before the space token
                            s_start=len(utt.token_ids_with_blanks) - 3,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 3,
                        )
                    )
                    utt.segments_and_tokens[-1].words_and_tokens.append(
                        Token(
                            text=SPACE_TOKEN,
                            text_cased=SPACE_TOKEN,
                            token_id=SPACE_ID,
                            # utt.token_ids_with_blanks has the form
                            # [..., <final token of previous word>, <blank token>, <space token>, <blank token>]
                            # => do len(utt.token_ids_with_blanks) - 2 to get the index of the space token
                            s_start=len(utt.token_ids_with_blanks) - 2,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 2,
                        )
                    )
                    utt.segments_and_tokens[-1].words_and_tokens.append(
                        Token(
                            text=BLANK_TOKEN,
                            text_cased=BLANK_TOKEN,
                            token_id=BLANK_ID,
                            # utt.token_ids_with_blanks has the form
                            # [..., <final token of previous word>, <blank token>, <space token>, <blank token>]
                            # => do len(utt.token_ids_with_blanks) - 1 to get the index of the blank token after the space token
                            s_start=len(utt.token_ids_with_blanks) - 1,
                            # s_end is same as s_start since the token only occupies one element in the list
                            s_end=len(utt.token_ids_with_blanks) - 1,
                        )
                    )

            # add a blank to the segment, and add a space after if this is not the final segment
            utt.token_ids_with_blanks.extend([BLANK_ID])
            utt.segments_and_tokens.append(
                Token(
                    text=BLANK_TOKEN,
                    text_cased=BLANK_TOKEN,
                    token_id=BLANK_ID,
                    # utt.token_ids_with_blanks has the form [..., <this blank token>]
                    # => do len(utt.token_ids_with_blanks) - 1 to get the index of this blank token
                    s_start=len(utt.token_ids_with_blanks) - 1,
                    # s_end is same as s_start since the token only occupies one element in the list
                    s_end=len(utt.token_ids_with_blanks) - 1,
                )
            )

            if i_segment < len(segments) - 1:
                utt.token_ids_with_blanks.extend([SPACE_ID, BLANK_ID])
                utt.segments_and_tokens.append(
                    Token(
                        text=SPACE_TOKEN,
                        text_cased=SPACE_TOKEN,
                        token_id=SPACE_ID,
                        # utt.token_ids_with_blanks has the form
                        # [..., <space token>, <blank token>]
                        # => do len(utt.token_ids_with_blanks) - 2 to get the index of the space token
                        s_start=len(utt.token_ids_with_blanks) - 2,
                        # s_end is same as s_start since the token only occupies one element in the list
                        s_end=len(utt.token_ids_with_blanks) - 2,
                    )
                )
                utt.segments_and_tokens.append(
                    Token(
                        text=BLANK_TOKEN,
                        text_cased=BLANK_TOKEN,
                        token_id=BLANK_ID,
                        # utt.token_ids_with_blanks has the form
                        # [..., <space token>, <blank token>]
                        # => do len(utt.token_ids_with_blanks) - 1 to get the index of the blank token
                        s_start=len(utt.token_ids_with_blanks) - 1,
                        # s_end is same as s_start since the token only occupies one element in the list
                        s_end=len(utt.token_ids_with_blanks) - 1,
                    )
                )

        return utt

    else:
        raise RuntimeError(
            "Cannot get tokens of this model as it does not have a `tokenizer` or `vocabulary` attribute."
        )


def add_t_start_end_to_utt_obj(utt_obj: Utterance, alignment_utt: List[int], output_timestep_duration: float):
    """
    Function to add t_start and t_end (representing time in seconds) to the Utterance object utt_obj.
    Args:
        utt_obj: Utterance object to which we will add t_start and t_end for its
            constituent segments/words/tokens.
        alignment_utt: a list of ints indicating which token does the alignment pass through at each
            timestep (will take the form [0, 0, 1, 1, ..., <num of tokens including blanks in uterance>]).
        output_timestep_duration: a float indicating the duration of a single output timestep from
            the ASR Model.

    Returns:
        utt_obj: updated Utterance object.
    """

    # General idea for the algorithm of how we add t_start and t_end
    # the timestep where a token s starts is the location of the first appearance of s_start in alignment_utt
    # the timestep where a token s ends is the location of the final appearance of s_end in alignment_utt
    # We will make dictionaries num_to_first_alignment_appearance and
    # num_to_last_alignment_appearance and use that to update all of
    # the t_start and t_end values in utt_obj.
    # We will put t_start = t_end = -1 for tokens that are skipped (should only be blanks)

    num_to_first_alignment_appearance = dict()
    num_to_last_alignment_appearance = dict()

    prev_token_idx = -1  # use prev_token_idx to keep track of when the token_idx changes
    for timestep, token_idx in enumerate(alignment_utt):
        if token_idx > prev_token_idx:
            num_to_first_alignment_appearance[token_idx] = timestep

            if prev_token_idx >= 0:  # dont record prev_token_idx = -1
                num_to_last_alignment_appearance[prev_token_idx] = timestep - 1
        prev_token_idx = token_idx
    # add last appearance of the final s
    num_to_last_alignment_appearance[prev_token_idx] = len(alignment_utt) - 1

    # update all the t_start and t_end in utt_obj
    for segment_or_token in utt_obj.segments_and_tokens:
        if type(segment_or_token) is Segment:
            segment = segment_or_token

            if segment.s_start in num_to_first_alignment_appearance:
                segment.t_start = num_to_first_alignment_appearance[segment.s_start] * output_timestep_duration
            else:
                segment.t_start = -1

            if segment.s_end in num_to_last_alignment_appearance:
                segment.t_end = (num_to_last_alignment_appearance[segment.s_end] + 1) * output_timestep_duration
            else:
                segment.t_end = -1

            for word_or_token in segment.words_and_tokens:
                if type(word_or_token) is Word:
                    word = word_or_token

                    if word.s_start in num_to_first_alignment_appearance:
                        word.t_start = num_to_first_alignment_appearance[word.s_start] * output_timestep_duration
                    else:
                        word.t_start = -1

                    if word.s_end in num_to_last_alignment_appearance:
                        word.t_end = (num_to_last_alignment_appearance[word.s_end] + 1) * output_timestep_duration
                    else:
                        word.t_end = -1

                    for token in word.tokens:
                        if token.s_start in num_to_first_alignment_appearance:
                            token.t_start = num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                        else:
                            token.t_start = -1

                        if token.s_end in num_to_last_alignment_appearance:
                            token.t_end = (
                                num_to_last_alignment_appearance[token.s_end] + 1
                            ) * output_timestep_duration
                        else:
                            token.t_end = -1
                else:
                    token = word_or_token
                    if token.s_start in num_to_first_alignment_appearance:
                        token.t_start = num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                    else:
                        token.t_start = -1

                    if token.s_end in num_to_last_alignment_appearance:
                        token.t_end = (num_to_last_alignment_appearance[token.s_end] + 1) * output_timestep_duration
                    else:
                        token.t_end = -1

        else:
            token = segment_or_token
            if token.s_start in num_to_first_alignment_appearance:
                token.t_start = num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
            else:
                token.t_start = -1

            if token.s_end in num_to_last_alignment_appearance:
                token.t_end = (num_to_last_alignment_appearance[token.s_end] + 1) * output_timestep_duration
            else:
                token.t_end = -1

    return utt_obj


def viterbi_decoding(
    log_probs_batch: torch.Tensor,
    y_batch: torch.Tensor,
    T_batch: torch.Tensor,
    U_batch: torch.Tensor,
    viterbi_device: Optional[torch.device] = None,
    padding_value: float = -3.4e38,
):
    """
    Do Viterbi decoding with an efficient algorithm (the only for-loop in the 'forward pass' is over the time dimension).
    Args:
        log_probs_batch: tensor of shape (B, T_max, V). The parts of log_probs_batch which are 'padding' are filled
            with 'padding_value'
        y_batch: tensor of shape (B, U_max) - contains token IDs including blanks. The parts of
            y_batch which are padding are filled with the number 'V'. V = the number of tokens in the vocabulary + 1 for
            the blank token.
        T_batch: tensor of shape (B) - contains the durations of the log_probs_batch (so we can ignore the
            parts of log_probs_batch which are padding)
        U_batch: tensor of shape (B) - contains the lengths of y_batch (so we can ignore the parts of y_batch
            which are padding).
        viterbi_device: the torch device on which Viterbi decoding will be done.
        padding_value: - a large negative number which represents a very low probability. Default to -3.4e38, the smallest number in torch.float32.

    Returns:
        alignments_batch: list of lists containing locations for the tokens we align to at each timestep.
            Looks like: [[0, 0, 1, 2, 2, 3, 3, ...,  ], ..., [0, 1, 2, 2, 2, 3, 4, ....]].
            Each list inside alignments_batch is of length T_batch[location of utt in batch].
    """

    if viterbi_device is None:
        viterbi_device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T_max, _ = log_probs_batch.shape
    U_max = y_batch.shape[1]

    # transfer all tensors to viterbi_device
    log_probs_batch = log_probs_batch.to(viterbi_device)
    y_batch = y_batch.to(viterbi_device)
    T_batch = T_batch.to(viterbi_device)
    U_batch = U_batch.to(viterbi_device)

    # make tensor that we will put at timesteps beyond the duration of the audio
    padding_for_log_probs = padding_value * torch.ones((B, T_max, 1), device=viterbi_device)
    # make log_probs_padded tensor of shape (B, T_max, V +1 ) where all of
    # log_probs_padded[:,:,-1] is the 'padding_value'
    log_probs_padded = torch.cat((log_probs_batch, padding_for_log_probs), dim=2)

    # initialize v_prev - tensor of previous timestep's viterbi probabilies, of shape (B, U_max)
    v_prev = padding_value * torch.ones((B, U_max), device=viterbi_device)

    v_prev[:, :2] = torch.gather(input=log_probs_padded[:, 0, :], dim=1, index=y_batch[:, :2])

    # initialize backpointers_rel - which contains values like 0 to indicate the backpointer is to the same u index,
    # 1 to indicate the backpointer pointing to the u-1 index and 2 to indicate the backpointer is pointing to the u-2 index
    backpointers_rel = -99 * torch.ones((B, T_max, U_max), dtype=torch.int8, device=viterbi_device)

    # Make a letter_repetition_mask the same shape as y_batch
    # the letter_repetition_mask will have 'True' where the token (including blanks) is the same
    # as the token two places before it in the ground truth (and 'False everywhere else).
    # We will use letter_repetition_mask to determine whether the Viterbi algorithm needs to look two tokens back or
    # three tokens back
    y_shifted_left = torch.roll(y_batch, shifts=2, dims=1)
    letter_repetition_mask = y_batch - y_shifted_left
    letter_repetition_mask[:, :2] = 1  # make sure dont apply mask to first 2 tokens
    letter_repetition_mask = letter_repetition_mask == 0

    for t in range(1, T_max):

        # e_current is a tensor of shape (B, U_max) of the log probs of every possible token at the current timestep
        e_current = torch.gather(input=log_probs_padded[:, t, :], dim=1, index=y_batch)

        # apply a mask to e_current to cope with the fact that we do not keep the whole v_matrix and continue
        # calculating viterbi probabilities during some 'padding' timesteps
        t_exceeded_T_batch = t >= T_batch

        U_can_be_final = torch.logical_or(
            torch.arange(0, U_max, device=viterbi_device).unsqueeze(0) == (U_batch.unsqueeze(1) - 0),
            torch.arange(0, U_max, device=viterbi_device).unsqueeze(0) == (U_batch.unsqueeze(1) - 1),
        )

        mask = torch.logical_not(
            torch.logical_and(
                t_exceeded_T_batch.unsqueeze(1),
                U_can_be_final,
            )
        ).long()

        e_current = e_current * mask

        # v_prev_shifted is a tensor of shape (B, U_max) of the viterbi probabilities 1 timestep back and 1 token position back
        v_prev_shifted = torch.roll(v_prev, shifts=1, dims=1)
        # by doing a roll shift of size 1, we have brought the viterbi probability in the final token position to the
        # first token position - let's overcome this by 'zeroing out' the probabilities in the firest token position
        v_prev_shifted[:, 0] = padding_value

        # v_prev_shifted2 is a tensor of shape (B, U_max) of the viterbi probabilities 1 timestep back and 2 token position back
        v_prev_shifted2 = torch.roll(v_prev, shifts=2, dims=1)
        v_prev_shifted2[:, :2] = padding_value  # zero out as we did for v_prev_shifted
        # use our letter_repetition_mask to remove the connections between 2 blanks (so we don't skip over a letter)
        # and to remove the connections between 2 consective letters (so we don't skip over a blank)
        v_prev_shifted2.masked_fill_(letter_repetition_mask, padding_value)

        # we need this v_prev_dup tensor so we can calculated the viterbi probability of every possible
        # token position simultaneously
        v_prev_dup = torch.cat(
            (
                v_prev.unsqueeze(2),
                v_prev_shifted.unsqueeze(2),
                v_prev_shifted2.unsqueeze(2),
            ),
            dim=2,
        )

        # candidates_v_current are our candidate viterbi probabilities for every token position, from which
        # we will pick the max and record the argmax
        candidates_v_current = v_prev_dup + e_current.unsqueeze(2)
        # we straight away save results in v_prev instead of v_current, so that the variable v_prev will be ready for the
        # next iteration of the for-loop
        v_prev, bp_relative = torch.max(candidates_v_current, dim=2)

        backpointers_rel[:, t, :] = bp_relative

    # trace backpointers
    alignments_batch = []
    for b in range(B):
        T_b = int(T_batch[b])
        U_b = int(U_batch[b])

        if U_b == 1:  # i.e. we put only a blank token in the reference text because the reference text is empty
            current_u = 0  # set initial u to 0 and let the rest of the code block run as usual
        else:
            current_u = int(torch.argmax(v_prev[b, U_b - 2 : U_b])) + U_b - 2
        alignment_b = [current_u]
        for t in range(T_max - 1, 0, -1):
            current_u = current_u - int(backpointers_rel[b, t, current_u])
            alignment_b.insert(0, current_u)
        alignment_b = alignment_b[:T_b]
        alignments_batch.append(alignment_b)

    return alignments_batch


def get_batch_variables(
    audio: Union[str, List[str], np.ndarray, DataLoader, Hypothesis],
    model: ASRModel,
    segment_separators: Union[str, List[str]] = ['.', '?', '!', '...'],
    word_separator: Optional[str] = " ",
    align_using_pred_text: bool = False,
    audio_filepath_parts_in_utt_id: int = 1,
    gt_text_batch: Union[List[str], str] = None,
    output_timestep_duration: Optional[float] = None,
    simulate_cache_aware_streaming: bool = False,
    use_buffered_chunked_streaming: bool = False,
    buffered_chunk_params: dict = {},
    padding_value: float = -3.4e38,
    has_hypotheses: bool = False,
):
    """
    Args:
        audio: a single audio file, a list of audio files, a numpy array, or a DataLoader that needs to be transcribed.
            Note: if using streaming mode, audio must be a list of audio files or a single audio file (not a numpy array or a DataLoader).
        model: an ASRModel, supports transcribe and transcribe_simulate_cache_aware_streaming methods.
        separator: a string or a list of strings, that will be used to split the text into segments.
        align_using_pred_text: a boolean, if True, the predicted text will be used for alignment.
        audio_filepath_parts_in_utt_id: an integer, the number of parts of the audio filepath to use for the utterance ID.
        gt_text_batch: a list of ground truth texts for the audio files. If provided, it will be used for alignment instead of the predicted text.
        output_timestep_duration: a float, the duration of a single frame (output timestep) in seconds.
        simulate_cache_aware_streaming: a boolean, if True, the cache-aware streaming will be used.
        use_buffered_chunked_streaming: a boolean, if True, the buffered chunked streaming will be used.
        buffered_chunk_params: a dictionary, containing the parameters for the buffered chunked streaming.
        padding_value: a float, the value to use for padding the log_probs tensor.
        has_hypotheses: a boolean, if True, the audio has already been processed and hypotheses are provided.

    Returns:
        log_probs_batch: a tensor of shape (B, T_max, V) - contains the log probabilities of the tokens for each utterance in the batch.
            The parts of log_probs_batch which are 'padding' are filled with 'padding_value', which is a large negative number.
        y_batch: a tensor of shape (B, U_max) - contains token IDs including blanks in every other position.
            The parts of y_batch which are padding are filled with the number 'V'. V = the number of tokens in the vocabulary + 1 for the blank token.
        T_batch: a tensor of shape (B) - contains the number of frames in the log_probs for each utterance in the batch.
        U_batch: a tensor of shape (B) - contains the lengths of token_ids_with_blanks for each utterance in the batch.
        utt_obj_batch: a list of Utterance objects for every utterance in the batch. Each Utterance object contains the token_ids_with_blanks, segments_and_tokens, text, pred_text, audio_filepath, utt_id, and saved_output_files.
        output_timestep_duration: a float indicating the duration of a single output timestep from
            the ASR Model in milliseconds.
    """

    if output_timestep_duration is None:
        try:
            output_timestep_duration = model.cfg['preprocessor']['window_stride'] * model.encoder.subsampling_factor
            logging.info(
                f"`output_timestep_duration` is not provided, so we calculated that the model output_timestep_duration is {output_timestep_duration} ms."
                " -- will use this for all batches"
            )
        except:
            raise ValueError("output_timestep_duration is not provided and cannot be calculated from the model.")

    if simulate_cache_aware_streaming or use_buffered_chunked_streaming:
        if not (isinstance(audio, list) and all(isinstance(item, str) for item in audio)) and not isinstance(
            audio, str
        ):
            raise ValueError("Audio must be a list of audio files or a single audio file when using streaming mode.")

        if isinstance(audio, str):
            audio = [audio]

    if gt_text_batch is not None:
        if isinstance(gt_text_batch, str):
            gt_text_batch = [gt_text_batch]
        if len(gt_text_batch) != len(audio):
            raise ValueError("`gt_text_batch` must be the same length as `audio` for performing alignment.")

    # get hypotheses by calling 'transcribe'
    # we will use the output log_probs, the duration of the log_probs,
    # and (optionally) the predicted ASR text from the hypotheses

    batch_size = len(audio)
    log_probs_list_batch = []  # log_probs is the output of the ASR model, with a shape (T, V+1)
    T_list_batch = []  # T is the number of frames in the log_probs
    pred_text_batch = []  # pred_text is the predicted text from the ASR model

    if not use_buffered_chunked_streaming:
        if not simulate_cache_aware_streaming:
            with torch.no_grad():
                if has_hypotheses:
                    hypotheses = audio
                else:
                    hypotheses = model.transcribe(audio, return_hypotheses=True, batch_size=batch_size)
        else:
            assert isinstance(audio, list) or isinstance(
                audio, str
            ), "audio must be a list of audio files or a single audio file"
            with torch.no_grad():
                hypotheses = model.transcribe_simulate_cache_aware_streaming(
                    audio, return_hypotheses=True, batch_size=batch_size
                )

        # if hypotheses form a tuple (from Hybrid model), extract just "best" hypothesis
        if type(hypotheses) == tuple and len(hypotheses) == 2:
            hypotheses = hypotheses[0]

        for hypothesis in hypotheses:
            log_probs_list_batch.append(hypothesis.y_sequence)
            T_list_batch.append(hypothesis.y_sequence.shape[0])
            pred_text_batch.append(hypothesis.text)
    else:
        delay = buffered_chunk_params["delay"]
        model_stride_in_secs = buffered_chunk_params["model_stride_in_secs"]
        tokens_per_chunk = buffered_chunk_params["tokens_per_chunk"]
        for audio_sample in tqdm(audio, desc="Sample:"):
            model.reset()
            model.read_audio_file(audio_sample, delay, model_stride_in_secs)
            hyp, logits = model.transcribe(tokens_per_chunk, delay, keep_logits=True)
            log_probs_list_batch.append(logits)
            T_list_batch.append(logits.shape[0])
            pred_text_batch.append(hyp)

    # we loop over every line in the manifest that is in our current batch,
    # and record the y (list of tokens, including blanks), U (list of lengths of y) and
    # token_info_batch, word_info_batch, segment_info_batch

    y_list_batch = []  # List of lists of token IDs with blanks, where each token is followed by a blank
    U_list_batch = []  # List of lengths of y_list_batch
    utt_obj_batch = []  # List of Utterance objects for every utterance in the batch

    for idx, sample in enumerate(audio):

        if align_using_pred_text:
            # normalizes the predicted text by removing extra spaces
            gt_text_for_alignment = pred_text_batch[idx]
        else:
            gt_text_for_alignment = gt_text_batch[idx]

        gt_text_for_alignment = " ".join(gt_text_for_alignment.split())

        utt_obj = get_utt_obj(
            text=gt_text_for_alignment,
            model=model,
            segment_separators=segment_separators,
            word_separator=word_separator,
            T=T_list_batch[idx],
            audio_filepath=sample if isinstance(sample, str) else f"audio_{idx}",
            utt_id=_get_utt_id(sample if isinstance(sample, str) else f"audio_{idx}", audio_filepath_parts_in_utt_id),
        )

        if len(gt_text_for_alignment) == 0:
            logging.info(
                f"'text' of utterance with ID: {utt_obj.utt_id} is empty - we will not generate"
                " any output alignment files for this utterance"
            )

        if align_using_pred_text:
            utt_obj.pred_text = pred_text_batch[idx]
            utt_obj.text = " ".join(pred_text_batch[idx].split()) if pred_text_batch[idx] else None

        y_list_batch.append(utt_obj.token_ids_with_blanks)
        U_list_batch.append(len(utt_obj.token_ids_with_blanks))
        utt_obj_batch.append(utt_obj)

    # turn log_probs, y, T, U into dense tensors for fast computation during Viterbi decoding
    T_max = max(T_list_batch)
    U_max = max(U_list_batch)

    #  V = the number of tokens in the vocabulary + 1 for the blank token.
    if hasattr(model, 'tokenizer'):
        V = len(model.tokenizer.vocab) + 1
    else:
        V = len(model.decoder.vocabulary) + 1

    T_batch = torch.tensor(T_list_batch)
    U_batch = torch.tensor(U_list_batch)

    # make log_probs_batch tensor of shape (B x T_max x V)
    log_probs_batch = padding_value * torch.ones((batch_size, T_max, V))
    for b, log_probs_utt in enumerate(log_probs_list_batch):
        t = T_list_batch[b]
        log_probs_batch[b, :t, :] = log_probs_utt

    # make y tensor of shape (B x U_max)
    # populate it initially with all 'V' numbers so that the 'V's will remain in the areas that
    # are 'padding'. This will be useful for when we make 'log_probs_reorderd' during Viterbi decoding
    # in a different function.
    y_batch = V * torch.ones((batch_size, U_max), dtype=torch.int64)
    for b, y_utt in enumerate(y_list_batch):
        U_utt = U_batch[b]
        y_batch[b, :U_utt] = torch.tensor(y_utt)

    return (
        log_probs_batch,
        y_batch,
        T_batch,
        U_batch,
        utt_obj_batch,
        output_timestep_duration,
    )
import torch
import torch.nn.functional as F

def dtw_alignment(attention_matrix, return_path=True):
    """
    Compute DTW alignment from attention matrix to cost matrix.
    
    Args:
        attention_matrix: Tensor of shape [batch_size, decoder_steps, encoder_steps]
                         or [decoder_steps, encoder_steps] for single batch
        return_path: Whether to return the optimal alignment path
    
    Returns:
        cost: DTW cost
        path: Optimal alignment path (if return_path=True)
    """
    if len(attention_matrix.shape) == 2:
        attention_matrix = attention_matrix.reshape(1, attention_matrix.shape[0], attention_matrix.shape[1])
    
    batch_size, M, N = attention_matrix.shape  # M=decoder_steps, N=encoder_steps
    print("attention_matrix.shape: ", attention_matrix.shape)
    cost_matrix = -attention_matrix
    import pdb; pdb.set_trace()
    plot_raw_cost_matrix(cost_matrix)
    import pdb; pdb.set_trace()
    
    results = []
    
    for b in range(batch_size):
        cost = cost_matrix[b]  # [M, N]
        
        # Initialize DTW matrix
        dtw_matrix = torch.full((M + 1, N + 1), float('inf'), device=cost.device)
        dtw_matrix[0, 0] = 0.0
        
        # Fill DTW matrix
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                dtw_cost = cost[i-1, j-1]
                dtw_matrix[i, j] = dtw_cost + torch.min(torch.stack([
                    dtw_matrix[i-1, j],     # insertion
                    dtw_matrix[i, j-1],     # deletion
                    dtw_matrix[i-1, j-1]    # match
                ]))
        
        total_cost = dtw_matrix[M, N]
        
        if return_path:
            # Backtrack to find optimal path
            path = []
            i, j = M, N
            
            while i > 0 and j > 0:
                path.append((i-1, j-1))  # Convert back to 0-indexed
                
                # Find which direction gave minimum cost
                candidates = [
                    (dtw_matrix[i-1, j-1], (i-1, j-1)),  # diagonal
                    (dtw_matrix[i-1, j], (i-1, j)),      # up
                    (dtw_matrix[i, j-1], (i, j-1))       # left
                ]
                _, (i, j) = min(candidates, key=lambda x: x[0])
            
            path.reverse()
            results.append((total_cost, path))
        else:
            results.append(total_cost)
    
    if len(results) == 1:
        return results[0]
    return results

def dtw_alignment_vectorized(attention_matrix):
    """
    Vectorized DTW implementation (faster but more memory intensive).
    
    Args:
        attention_matrix: Tensor of shape [batch_size, decoder_steps, encoder_steps]
    
    Returns:
        dtw_costs: Tensor of DTW costs for each batch item
    """
    if attention_matrix.dim() == 2:
        attention_matrix = attention_matrix.unsqueeze(0)
    
    batch_size, M, N = attention_matrix.shape
    
    # Convert attention to cost
    cost_matrix = 1.0 - attention_matrix
    
    # Initialize DTW matrix for all batches
    dtw_matrix = torch.full((batch_size, M + 1, N + 1), float('inf'), device=attention_matrix.device)
    dtw_matrix[:, 0, 0] = 0.0
    
    # Fill DTW matrix
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            dtw_cost = cost_matrix[:, i-1, j-1]
            candidates = torch.stack([
                dtw_matrix[:, i-1, j],     # insertion
                dtw_matrix[:, i, j-1],     # deletion
                dtw_matrix[:, i-1, j-1]    # match
            ], dim=1)
            dtw_matrix[:, i, j] = dtw_cost + torch.min(candidates, dim=1)[0]
    
    return dtw_matrix[:, M, N]

# Usage with your attention scores:
def apply_dtw_to_attention(final_tensor):
    """
    Apply DTW to your processed attention tensor.
    
    Args:
        final_tensor: Shape [1, 17, 46] - your averaged attention scores
    
    Returns:
        dtw_cost: DTW alignment cost
        alignment_path: Optimal alignment path
    """
    # Ensure attention scores are normalized (optional)
    attention_normalized = F.softmax(final_tensor, dim=-1)
    
    # Apply DTW
    dtw_cost, alignment_path = dtw_alignment(attention_normalized, return_path=True)
    
    return dtw_cost, alignment_path

def dtw_path_to_timestamps(
    path: List[Tuple[int, int]], 
    tokens: List[str], 
    audio_duration: float,
    frame_duration: float = None,
    char_level: bool = True
) -> Dict[str, List[Dict]]:
    """
    Convert DTW alignment path to timestamp format.
    
    Args:
        path: DTW alignment path as list of (decoder_step, encoder_step) tuples
        tokens: List of tokens/characters corresponding to decoder steps
        audio_duration: Total audio duration in seconds
        frame_duration: Duration of each encoder frame in seconds (if None, calculated from audio_duration)
        char_level: Whether to create character-level timestamps (True) or token-level (False)
    
    Returns:
        Dictionary with timestamp information in NeMo format
    """
    if frame_duration is None:
        # Calculate frame duration from total audio duration and number of encoder steps
        max_encoder_step = max(encoder_step for _, encoder_step in path)
        frame_duration = audio_duration / (max_encoder_step + 1)
    
    # Group consecutive decoder steps that map to the same or consecutive encoder steps
    timestamp_segments = []
    current_segment = {
        'decoder_start': None,
        'decoder_end': None,
        'encoder_start': None,
        'encoder_end': None,
        'chars': []
    }
    
    for i, (decoder_step, encoder_step) in enumerate(path):
        if current_segment['decoder_start'] is None:
            # Start new segment
            current_segment['decoder_start'] = decoder_step
            current_segment['encoder_start'] = encoder_step
            current_segment['decoder_end'] = decoder_step
            current_segment['encoder_end'] = encoder_step
            if decoder_step < len(tokens):
                current_segment['chars'].append(tokens[decoder_step])
        else:
            # Check if we should continue current segment or start new one
            prev_decoder_step = current_segment['decoder_end']
            
            if decoder_step == prev_decoder_step:
                # Same decoder step, extend encoder range
                current_segment['encoder_end'] = encoder_step
            elif decoder_step == prev_decoder_step + 1:
                # Next decoder step, extend both ranges
                current_segment['decoder_end'] = decoder_step
                current_segment['encoder_end'] = encoder_step
                if decoder_step < len(tokens):
                    current_segment['chars'].append(tokens[decoder_step])
            else:
                # Gap in decoder steps, finish current segment and start new one
                timestamp_segments.append(current_segment)
                current_segment = {
                    'decoder_start': decoder_step,
                    'decoder_end': decoder_step,
                    'encoder_start': encoder_step,
                    'encoder_end': encoder_step,
                    'chars': [tokens[decoder_step]] if decoder_step < len(tokens) else []
                }
    
    # Add the last segment
    if current_segment['decoder_start'] is not None:
        timestamp_segments.append(current_segment)
    
    # Convert to timestamp format
    result_timestamps = []
    
    for segment in timestamp_segments:
        if not segment['chars']:  # Skip empty segments
            continue
            
        # Calculate time boundaries
        start_time = segment['encoder_start'] * frame_duration
        end_time = (segment['encoder_end'] + 1) * frame_duration  # +1 because end is inclusive
        
        # Calculate character offsets (assuming character-level tokens)
        start_offset = segment['decoder_start']
        end_offset = segment['decoder_end'] + 1  # +1 because end is exclusive in offset
        
        if char_level:
            # Create individual character timestamps
            chars_in_segment = segment['chars']
            segment_duration = end_time - start_time
            char_duration = segment_duration / len(chars_in_segment) if len(chars_in_segment) > 0 else 0
            
            for i, char in enumerate(chars_in_segment):
                char_start_time = start_time + (i * char_duration)
                char_end_time = start_time + ((i + 1) * char_duration)
                char_start_offset = start_offset + i
                char_end_offset = start_offset + i + 1
                
                result_timestamps.append({
                    "char": [char],
                    "start_offset": char_start_offset,
                    "end_offset": char_end_offset,
                    "start": round(char_start_time, 3),
                    "end": round(char_end_time, 3)
                })
        else:
            # Create token-level timestamp
            result_timestamps.append({
                "char": segment['chars'],
                "start_offset": start_offset,
                "end_offset": end_offset,
                "start": round(start_time, 3),
                "end": round(end_time, 3)
            })
    
    return {"char": result_timestamps}


def create_timestamps_from_dtw_path(
    path, 
    y_sequence,  # Your token IDs 
    tokenizer,   # Your model's tokenizer
    frame_duration_ms=80.0
):
    """
    Create timestamps from DTW path and token sequence.
    
    Args:
        path: DTW alignment path like [(0, 0), (0, 1), (0, 2), ...]
        y_sequence: List of token IDs from decoder
        tokenizer: Model's tokenizer to convert tokens to text
        frame_duration_ms: Duration per frame in milliseconds (default: 80ms)
    
    Returns:
        Dictionary with character-level timestamps in the format you requested
    """
    import torch
    
    frame_duration_s = frame_duration_ms / 1000.0
    
    # Group path segments by decoder steps
    segments = []
    current_segment = None
    
    for decoder_step, encoder_step in path:
        if current_segment is None or current_segment['decoder_step'] != decoder_step:
            # Start new segment
            if current_segment is not None:
                segments.append(current_segment)
            current_segment = {
                'decoder_step': decoder_step,
                'encoder_start': encoder_step,
                'encoder_end': encoder_step
            }
        else:
            # Extend current segment
            current_segment['encoder_end'] = encoder_step
    
    if current_segment is not None:
        segments.append(current_segment)
    
    # Create character-level timestamps
    result_timestamps = []
    char_offset = 0
    
    for segment in segments:
        decoder_step = segment['decoder_step']
        
        # Skip if decoder step is out of range
        if decoder_step >= len(y_sequence):
            continue
            
        # Get the token text for this decoder step
        token_id = y_sequence[decoder_step]
        # Convert token ID to text using tokenizer
        if hasattr(tokenizer, 'ids_to_tokens'):
            token_text = tokenizer.ids_to_tokens([token_id.item()])
        elif hasattr(tokenizer, 'decode_ids_to_str'):
            token_text = tokenizer.decode_ids_to_str([token_id.item()])
        elif hasattr(tokenizer, 'decode'):
            token_text = tokenizer.decode([token_id.item()])
        elif hasattr(tokenizer, 'ids_to_text'):
            token_text = tokenizer.ids_to_text([token_id.item()])[0]
        else:
            # Fallback - convert to string
            token_text = str(token_id)
        
        # Clean up token text (remove special tokens like <unk>, <pad>, etc.)
        # if token_text.startswith('<') and token_text.endswith('>'):
        #     continue  # Skip special tokens
        
        # Calculate timing
        start_time = segment['encoder_start'] * frame_duration_s
        end_time = segment['encoder_end'] * frame_duration_s
        
        # Create character-level timestamps for this token
        if len(token_text) > 0:
            
            result_timestamps.append({
                "char": token_text,
                "start_offset": segment['encoder_start'],
                "end_offset": segment['encoder_end'],
                "start": round(start_time, 2),
                "end": round(end_time, 2)
            })
            
    
    return {"char": result_timestamps}



# Usage with your data:
# timestamps = create_timestamps_from_dtw_path(path, y_sequence, model.tokenizer)

# Example usage with your path:
# def create_timestamps_from_dtw(path, tokens, audio_duration):
#     """
#     Simplified wrapper for your specific use case.
    
#     Args:
#         path: Your DTW path like [(0, 0), (0, 1), (0, 2), ...]
#         tokens: List of tokens/characters from your model
#         audio_duration: Total audio duration in seconds
#     """
#     return dtw_path_to_timestamps(
#         path=path,
#         tokens=tokens,
#         audio_duration=audio_duration,
#         char_level=True  # Set to False for token-level timestamps
#     )

# Usage example:
# Assuming you have:
# path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 8), ...]
# tokens = ["F", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", ...]  # Your decoded tokens
# audio_duration = 2.3  # seconds
# 
# timestamps = create_timestamps_from_dtw(path, tokens, audio_duration)
# print(timestamps)

# Example usage:
# dtw_input = final_tensor.mean(dim=(1, 2))  # Shape: [1, 17, 46]
# cost, path = apply_dtw_to_attention(dtw_input)
# print(f"DTW cost: {cost}")
# print(f"Alignment path length: {len(path)}")

def create_encoded_char_offsets_from_timestamps(timestamps, y_sequence, tokenizer):
    """
    Create encoded_char_offsets from timestamps and token sequence.
    
    Args:
        timestamps: Dictionary with 'char' key containing character timestamps
        y_sequence: List of token IDs
        tokenizer: Model's tokenizer
    
    Returns:
        List of encoded character offsets with string tokens (not IDs)
    """
    char_timestamps = timestamps['char']
    encoded_char_offsets = []
    
    # # Convert token IDs to string tokens
    # if hasattr(tokenizer, 'ids_to_tokens'):
    #     string_tokens = tokenizer.text_ids(y_sequence)
    # else:
    #     # Fallback - convert IDs to strings
    #     string_tokens = [str(token_id) for token_id in y_sequence]
    
    # Map character timestamps to token strings
    token_idx = 0
    import pdb; pdb.set_trace()
    for i, char_offset in enumerate(char_timestamps):
        encoded_offset = char_offset.copy()
        
        # Use string token instead of token ID
        encoded_offset['char'] = [tokenizer.tokens_to_text(char_offset['char'])]
        # Keep token_id for reference if needed
        
        encoded_char_offsets.append(encoded_offset)
        

        # # Simple mapping: advance token index
        # # You might need to adjust this based on your tokenizer's behavior
        # token_idx += 1
    
    return encoded_char_offsets

# Alternative: If you know the exact mapping between characters and tokens
def create_encoded_char_offsets_with_mapping(timestamps, y_sequence, tokenizer):
    """
    Create encoded_char_offsets with proper character-to-token mapping.
    """
    char_timestamps = timestamps['char']
    
    # Get string tokens from IDs
    if hasattr(tokenizer, 'ids_to_tokens'):
        string_tokens = tokenizer.ids_to_tokens(y_sequence)
    else:
        string_tokens = [str(token_id) for token_id in y_sequence]
    
    # Decode full text to understand character distribution
    if hasattr(tokenizer, 'ids_to_text'):
        full_text = tokenizer.ids_to_text(y_sequence)
    elif hasattr(tokenizer, 'decode_ids_to_str'):
        full_text = tokenizer.decode_ids_to_str(y_sequence)
    else:
        full_text = ''.join(string_tokens)
    
    encoded_char_offsets = []
    token_idx = 0
    char_in_token = 0
    
    for char_offset in char_timestamps:
        encoded_offset = char_offset.copy()
        
        if token_idx < len(string_tokens):
            # Use the string token, not the ID
            encoded_offset['char'] = string_tokens[token_idx]
            encoded_offset['token_id'] = y_sequence[token_idx] if token_idx < len(y_sequence) else None
            
            # Move to next token when current token's characters are exhausted
            current_token = string_tokens[token_idx]
            char_in_token += 1
            
            # For subword tokens, you might need more sophisticated logic here
            if char_in_token >= len(current_token):
                token_idx += 1
                char_in_token = 0
        else:
            encoded_offset['char'] = char_offset['char']
            encoded_offset['token_id'] = None
        
        encoded_char_offsets.append(encoded_offset)
    
    return encoded_char_offsets


import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_raw_cost_matrix(attention_matrix):
    """
    Simple visualization of cost matrix to see alignment patterns.
    
    Args:
        attention_matrix: Tensor [batch_size, decoder_steps, encoder_steps] or [decoder_steps, encoder_steps]
        cost_function: 'neg_log' or 'invert' 
    """
    import matplotlib.pyplot as plt
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.cpu().float().numpy()
    
    # Handle both 2D and 3D arrays
    if len(attention_matrix.shape) == 2:
        # 2D case: [decoder_steps, encoder_steps]
        plt.figure(figsize=(10, 6))
        plt.imshow(attention_matrix, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(label='Cost')
        plt.xlabel('Encoder Steps (Audio Frames)')
        plt.ylabel('Decoder Steps (Text Tokens)')
        plt.title('Raw Cost Matrix')
        plt.savefig('cost_matrix.png')
        plt.show()
    else:
        # 3D case: [batch_size, decoder_steps, encoder_steps]
        batch_size = attention_matrix.shape[0]
        
        for b in range(batch_size):
            plt.figure(figsize=(10, 6))
            plt.imshow(attention_matrix[b], cmap='viridis', aspect='auto', origin='lower')
            plt.colorbar(label='Cost')
            plt.xlabel('Encoder Steps (Audio Frames)')
            plt.ylabel('Decoder Steps (Text Tokens)')
            plt.title(f'Raw Cost Matrix - Batch {b}')
            plt.savefig(f'cost_matrix_batch_{b}.png')
            plt.show()
        