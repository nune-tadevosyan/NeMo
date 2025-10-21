def get_utt_obj_tdt(
    text: str,
    T: int,
    model: ASRModel,
    durations: List[int],  # TDT-specific: list of possible durations [0, 1, 2, 4, 8, ...]
    segment_separators: Union[str, List[str]] = ['.', '?', '!', '...'],
    word_separator: Optional[str] = " ",
    audio_filepath: Optional[str] = None,
    utt_id: Optional[str] = None,
):
    """
    TDT-specific version of get_utt_obj for Viterbi alignment.
    
    Key differences from CTC:
    1. TDT predicts both tokens AND durations
    2. Blank token placement is different 
    3. Sequence structure accounts for duration predictions
    4. Blank ID position: len(vocabulary) (not len(vocabulary) + extra_outputs)
    
    Args:
        text: Text from another model that needs to be aligned to TDT logits
        T: Number of frames in the TDT logits
        model: TDT ASR model
        durations: List of possible duration values that TDT can predict
        segment_separators: Segment delimiter tokens
        word_separator: Word delimiter token
        audio_filepath: Path to audio file
        utt_id: Utterance ID
        
    Returns:
        Utterance object with TDT-specific tokenization
    """
    from nemo.collections.asr.parts.utils.aligner_utils import (
        Utterance, Segment, Word, Token, 
        BLANK_TOKEN, SPACE_TOKEN, get_char_tokens, restore_token_case
    )
    
    utt = Utterance(
        text=text,
        audio_filepath=audio_filepath,
        utt_id=utt_id,
    )

    # TDT-specific blank ID calculation
    if hasattr(model, 'tokenizer'):
        if hasattr(model, 'blank_id'):
            BLANK_ID = model.blank_id
        else:
            # For TDT: blank_id = len(vocabulary) (not len(vocabulary) + extra_outputs)
            BLANK_ID = model.tokenizer.vocab_size
            
        if hasattr(model.tokenizer, 'unk_id'):
            UNK_ID = model.tokenizer.unk_id
        else:
            UNK_ID = 0
            
        UNK_WORD = model.tokenizer.ids_to_text([UNK_ID]).strip()
        UNK_TOKEN = model.tokenizer.ids_to_tokens([UNK_ID])[0]
        
    elif hasattr(model.decoder, "vocabulary"):  # Character-based
        BLANK_ID = len(model.decoder.vocabulary)
        SPACE_ID = model.decoder.vocabulary.index(" ")
    else:
        raise RuntimeError(
            "Cannot get tokens of this model as it does not have a `tokenizer` or `vocabulary` attribute."
        )

    # Handle empty text
    if len(text) == 0:
        # TDT: Start with blank token
        utt.token_ids_with_blanks = [BLANK_ID]
        return utt

    # Parse segments
    if not segment_separators:
        segments = [text.strip()]
    else:
        segment_separators = [segment_separators] if isinstance(segment_separators, str) else segment_separators
        segments = []
        last_sep_idx = -1
        
        for i, letter in enumerate(text):
            next_letter = text[i + 1] if i + 1 < len(text) else ""
            if letter in segment_separators and next_letter == " ":
                segments.append(text[last_sep_idx + 1 : i + 1].strip())
                last_sep_idx = i + 1
                
        if last_sep_idx < len(text):
            segments.append(text[last_sep_idx + 1 :].strip())
    
    segments = [seg for seg in segments if len(seg) > 0]

    # TDT-specific tokenization
    if hasattr(model, 'tokenizer'):
        # **KEY DIFFERENCE**: TDT tokenization for Viterbi alignment
        
        # Check if sequence will be too long
        all_tokens = model.tokenizer.text_to_ids(text)
        if len(all_tokens) > T:
            logging.info(
                f"Utterance with ID: {utt_id} has too many tokens compared to the audio file duration."
                " Will not generate output alignment files for this utterance."
            )
            return utt

        # TDT-specific: Initialize with blank, but structure differently
        utt.token_ids_with_blanks = [BLANK_ID]
        
        # Build TDT token sequence
        utt.segments_and_tokens.append(
            Token(
                text=BLANK_TOKEN,
                text_cased=BLANK_TOKEN,
                token_id=BLANK_ID,
                s_start=0,
                s_end=0,
            )
        )

        segment_s_pointer = 1
        word_s_pointer = 1

        for segment in segments:
            segment_tokens = []
            sub_segments = segment.split(UNK_WORD)
            for i, sub_segment in enumerate(sub_segments):
                sub_segment_tokens = model.tokenizer.text_to_tokens(sub_segment.strip())
                segment_tokens.extend(sub_segment_tokens)
                if i < len(sub_segments) - 1:
                    segment_tokens.append(UNK_ID)

            # TDT-specific: Different calculation for segment boundaries
            # In TDT, we don't need to multiply by 2 since we're not inserting blanks between every token
            s_end = segment_s_pointer + len(segment_tokens) - 1
            
            utt.segments_and_tokens.append(
                Segment(
                    text=segment,
                    s_start=segment_s_pointer,
                    s_end=s_end,
                )
            )
            segment_s_pointer = s_end + 2  # Account for blank after segment

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

                # TDT-specific: Different word boundary calculation
                word_s_end = word_s_pointer + len(word_tokens) - 1
                utt.segments_and_tokens[-1].words_and_tokens.append(
                    Word(text=word, s_start=word_s_pointer, s_end=word_s_end)
                )
                word_s_pointer = word_s_end + 2  # Account for blank after word

                for token_i, (token, token_id, token_cased) in enumerate(
                    zip(word_tokens, word_token_ids, word_tokens_cased)
                ):
                    # TDT-specific: Add tokens without blanks between them initially
                    # The Viterbi algorithm will handle the duration aspect
                    utt.token_ids_with_blanks.append(token_id)
                    
                    # Add Token object for non-blank token
                    utt.segments_and_tokens[-1].words_and_tokens[-1].tokens.append(
                        Token(
                            text=token,
                            text_cased=token_cased,
                            token_id=token_id,
                            s_start=len(utt.token_ids_with_blanks) - 1,
                            s_end=len(utt.token_ids_with_blanks) - 1,
                        )
                    )

                # Add blank between words (TDT-specific placement)
                if word_i < len(words) - 1:
                    utt.token_ids_with_blanks.append(BLANK_ID)
                    utt.segments_and_tokens[-1].words_and_tokens.append(
                        Token(
                            text=BLANK_TOKEN,
                            text_cased=BLANK_TOKEN,
                            token_id=BLANK_ID,
                            s_start=len(utt.token_ids_with_blanks) - 1,
                            s_end=len(utt.token_ids_with_blanks) - 1,
                        )
                    )

            # Add blank between segments
            utt.token_ids_with_blanks.append(BLANK_ID)
            utt.segments_and_tokens.append(
                Token(
                    text=BLANK_TOKEN,
                    text_cased=BLANK_TOKEN,
                    token_id=BLANK_ID,
                    s_start=len(utt.token_ids_with_blanks) - 1,
                    s_end=len(utt.token_ids_with_blanks) - 1,
                )
            )

    elif hasattr(model.decoder, "vocabulary"):  # Character-based TDT
        # Similar logic but for character-based models
        # (Implementation similar to above but with character tokens)
        pass

    return utt


# Usage in get_batch_variables_tdt:
def get_batch_variables_tdt_fixed(
    hypotheses: Union[str, List[str], np.ndarray, DataLoader, Hypothesis],
    model: ASRModel,
    durations: List[int],  # TDT-specific durations
    segment_separators: Union[str, List[str]] = ['.', '?', '!', '...'],
    word_separator: Optional[str] = " ",
    audio_filepath_parts_in_utt_id: int = 1,
    gt_text_batch: Union[List[str], str] = None,
    output_timestep_duration: Optional[float] = None,
    padding_value: float = -3.4e38,
):
    """
    Fixed version of get_batch_variables_tdt that uses TDT-specific tokenization.
    """
    # ... (same setup code as original) ...
    
    for idx, sample in enumerate(hypotheses):
        gt_text_for_alignment = gt_text_batch[idx]
        gt_text_for_alignment = " ".join(gt_text_for_alignment.split())

        # **KEY CHANGE**: Use TDT-specific tokenization
        utt_obj = get_utt_obj_tdt(  # <-- Use TDT version instead of regular get_utt_obj
            text=gt_text_for_alignment,
            model=model,
            durations=durations,  # Pass TDT durations
            segment_separators=segment_separators,
            word_separator=word_separator,
            T=T_list_batch[idx],
            audio_filepath=sample if isinstance(sample, str) else f"audio_{idx}",
            utt_id=_get_utt_id(sample if isinstance(sample, str) else f"audio_{idx}", audio_filepath_parts_in_utt_id),
        )
        
        # ... (rest of the function remains the same) ...
