from typing import List, Optional, Tuple, Union, Dict, Any

import argparse
# import jiwer
import json
import logging
import numpy as np
import os
import re
import string

def word_error_rate_detail(
    hypotheses: List[str], references: List[str], use_cer=False, has_timestamp=True, use_jiwer=False, time_diff_threshold=3, remove_space_and_handle_special_tokens_for_hyp=True, hyp_only_has_start=False, hyp_only_has_end=False, hyp_every_n=-1, compute_wer_only=False
    ) -> Tuple[float, int, float, float, float]:
    """
    Computes Average Word Error Rate with details (insertion rate, deletion rate, substitution rate)
    between two texts represented as corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer (float): average word error rate
        words (int):  Total number of words/charactors of given reference texts
        ins_rate (float): average insertion error rate
        del_rate (float): average deletion error rate
        sub_rate (float): average substitution error rate
    """
    scores = 0
    words = 0
    ops_count = {'substitutions': 0, 'insertions': 0, 'deletions': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'time_mismatches': 0,
                 'avg_start_diff': 0, 'avg_end_diff': 0}
    # print("word_error_rate_detail")
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )

    # print(f"{len(hypotheses)}")
    # print(f"{len(references)}")
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()

        def remove_special_patterns(text):
            # Remove patterns like <|word|>, <word>, or <|6|>
            text = re.sub(r'<\|?\w+\|?>', ' ', text)
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text

        if not has_timestamp:
            r = remove_special_patterns(r)
            h = remove_special_patterns(h)

        #logging.debug('evaluate')
        # logging.info(f'hyp: {h}')
        # logging.info(f'ref: {r}')

        def levenshtein_distance_with_details(s1, s2, hyp_only_has_start, hyp_only_has_end, compute_wer_only=False):
            # s1: ref, s2: hyp
            # Create a matrix to store distances
            d = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
            insertions = deletions = substitutions = tp = fp = fn = time_mismatches = 0
            time_differences_m = time_differences_n = time_diff_count = 0
            
            # Initialize the first row and column of the matrix
            for i in range(len(s1) + 1):
                d[i][0] = i
            for j in range(len(s2) + 1):
                d[0][j] = j

            def parse_token(token, only_has_start=False, only_has_end=False):
                # The time token looks like <|1|>with<|2|>
                # print(f'parse: token:{token}')
                if only_has_start:
                    match = re.match(r"<\|(\d+)\|>(.*)", token)
                elif only_has_end:
                    match = re.match(r"(.*)<\|(\d+)\|>", token)
                else:
                    match = re.match(r"<\|(\d+)\|>(.*)<\|(\d+)\|>", token)
                # Only use for every N word
                # match = re.match(r"(?:<\|(\d+)\|>(.*)|(.+?)<\|(\d+)\|>)", token)
                if match:
                    if only_has_start:
                        m, word = match.groups()
                        return word, int(m), 0
                    elif only_has_end:
                        word, n = match.groups()
                        return word, 0, int(n)
                    else:
                        m, word, n = match.groups()
                        return word, int(m), int(n)
                else:
                    # Only word
                    return token, 0, 0
                
            def normalize_word(word):
                return word.lower().translate(str.maketrans('', '', string.punctuation))
    
            def tokens_equal(token1, token2, is_first_word=False, is_last_word=False, hyp_only_has_start=False, hyp_only_has_end=False, compute_wer_only=False):
                #logging.debug(f'token1: {token1}, token2: {token2}')
                word1, m1, n1 = parse_token(token1, hyp_only_has_start, hyp_only_has_end)
                word2, m2, n2 = parse_token(token2, hyp_only_has_start, hyp_only_has_end)

                word1 = normalize_word(word1)
                word2 = normalize_word(word2)
                # print(f'matching norm word1:{word1}, word2:{word2}, m1:{m1}, m2:{m2}, n1:{n1}, n2:{n2}')

                # if is_first_word:
                #     time_equal = abs(n1 - n2) <= time_diff_threshold
                #     print('first word, only consider end time!')
                if is_last_word:
                    time_equal = abs(m1 - m2) <= time_diff_threshold
                    #logging.debug('last word, only consider start time!')
                else:
                    time_equal = abs(m1 - m2) <= time_diff_threshold and abs(n1 - n2) <= time_diff_threshold

                if word1 == word2 and time_equal:
                    #logging.debug('word match, time match!')
                    return True, True, abs(m1-m2), abs(n1-n2), word1, word2
                elif word1 == word2:
                    #logging.debug('word match, time mismatch!')
                    time_equal = False if not compute_wer_only else True
                    return True, time_equal, abs(m1-m2), abs(n1-n2), word1, word2
                else:
                    return False, False, abs(m1-m2), abs(n1-n2), word1, word2
            
            # Fill the matrix
            for i in range(1, len(s1) + 1):
                for j in range(1, len(s2) + 1):
                    is_first_word = (i == 1 and j == 1)
                    is_last_word = (i == len(s1) and j == len(s2))
                    word_equal, time_equal, _, _, word1, word2 = tokens_equal(
                        s1[i - 1], s2[j - 1], is_first_word, is_last_word, hyp_only_has_start, hyp_only_has_end, compute_wer_only=compute_wer_only)
                    if word_equal:
                        cost = 0 if time_equal else 1
                    else:
                        cost = 1
                    d[i][j] = min(d[i - 1][j] + 1,      # Deletion
                                d[i][j - 1] + 1,      # Insertion
                                d[i - 1][j - 1] + cost)  # Substitution
            
            # Backtrack to find the number of insertions, deletions, and substitutions
            i, j = len(s1), len(s2)
            while i > 0 or j > 0:
                is_first_word = (i == 1 and j == 1)
                is_last_word = (i == len(s1) and j == len(s2))

                if i > 0 and d[i][j] == d[i-1][j] + 1:
                    #logging.debug(f'i: {i}, j: {j}, deletion')
                    deletions += 1
                    i -= 1
                    fn += 1
                elif j > 0 and d[i][j] == d[i][j-1] + 1:
                    #logging.debug(f'i: {i}, j: {j}, inseration')
                    insertions += 1
                    j -= 1
                    fp += 1
                elif i > 0 and j > 0:
                    word_equal, time_equal, m_diff, n_diff, word1, word2 = tokens_equal(
                        s1[i - 1], s2[j - 1], hyp_only_has_start=hyp_only_has_start, hyp_only_has_end=hyp_only_has_end, compute_wer_only=compute_wer_only)
                    # print(f'i: {i}, j:{j} --> {s1[i - 1]}, {s2[j - 1]}, word_equal: {word_equal}, time_equal: {time_equal}, '
                    #       f'm_diff: {m_diff}, n_diff: {n_diff}')
                    if d[i][j] == d[i-1][j-1] + 1:
                        #logging.debug('substition')
                        # word or time, at least one does not match
                        substitutions += 1
                        assert (not word_equal or not time_equal), 'at least word or time not match'                            
                        if word_equal and not time_equal:
                            # word match, time not match --> count diff
                            time_mismatches += 1
                            i -= 1
                            j -= 1
                            # if not is_first_word:
                            time_differences_m += m_diff
                            if not is_last_word:
                                time_differences_n += n_diff
                            # This is a both a False Positive and False Negative, i.e.,
                            # A ref alignment does not have a matching hyp alignment, and 
                            # a hyp alignment does not have a matching ref alignment either
                            fp += 1
                            fn += 1
                            time_diff_count += 1
                        elif not word_equal and not time_equal:
                            # word does not match --> both a fp and fn
                            fp += 1
                            fn += 1
                            # word does not match --> do not count time diff
                            i -= 1
                            j -= 1
                    else:
                        #logging.debug('match')
                        # word match, time match within the collar --> still count time diff
                        i -= 1
                        j -= 1        
                        # if not is_first_word:
                        time_differences_m += m_diff
                        if not is_last_word:                            
                            time_differences_n += n_diff
                        # This is a True Positive: A ref word matches a hyp word, and times match
                        tp += 1
                        time_diff_count += 1
                    # tp + fp = # hyp words, fn + tp = # ref words
            
            return d[len(s1)][len(s2)], insertions, deletions, substitutions, tp, fp, fn, time_mismatches, time_differences_m, time_differences_n, time_diff_count

        def remove_spaces_and_add_between_special_tokens(s, only_has_start_time=False, only_has_end_time=False,
                                                         remove_token_before_word=False, remove_token_after_word=False, hyp_every_n=-1):
            # First format the sequence to has a space between each time token and word. E.g.: <|start|> word <|end|> <|start|> word <|end|>, or <|start|> word <|start|> word, word <|end|> word <|end|>
            if only_has_start_time or only_has_end_time:
                s = re.sub(r'(<\|\d+\|>)(\w)', r'\1 \2', s)  # Add space after time token
                s = re.sub(r'(\w)(<\|\d+\|>)', r'\1 \2', s)  # Add space before time token

            # This is specifically designed for ref to remove extra time tokens
            if remove_token_before_word:
                # Remove the time token before the word
                s = re.sub(r'<\|\d+\|>\s*(\w)', r'\1', s)
            if remove_token_after_word:
                # Remove the time token after the word
                s = re.sub(r'(\w)\s*<\|\d+\|>', r'\1', s)
            
            if only_has_end_time:
                # Remove spaces before '<' if the previous non-space character is not '>'
                s = re.sub(r'(?<!>)\s+<', '<', s)
            elif only_has_start_time:
                # Remove spaces after '>' if the following non-space character is not '<'
                s = re.sub(r'>\s+(?!<)', '>', s)
            else:
                # Remove both spaces
                s = re.sub(r'(?<!>)\s+<', '<', s)
                s = re.sub(r'>\s+(?!<)', '>', s)
            
            # Add a space between two special tokens
            s = re.sub(r'(?<=[<>])(?=[<>])', ' ', s)

            s = re.sub(r'(?:<\|\d+\|>\s*){3,}', '', s)
            
            return s

        def word_error_rate(reference, hypothesis, remove_space_and_handle_special_tokens_for_hyp=True, hyp_only_has_start=True, hyp_only_has_end=False,
                            hyp_every_n=-1):
            # Split the strings into words
            #logging.debug(f'word_error_rate, ref: {reference}, hyp: {hypothesis}')
            
            # Ref has both start and end, so first format to ref to: 1) both start and end, 2) only start, or 3) only end
            ref = remove_spaces_and_add_between_special_tokens(
                reference, hyp_only_has_start, hyp_only_has_end,
                remove_token_after_word=hyp_only_has_start, remove_token_before_word=hyp_only_has_end,
                hyp_every_n=hyp_every_n)
            #logging.debug(f'ref after remove_spaces_and_add_between_special_tokens: {ref}')
            
            # Then format hyp
            if remove_space_and_handle_special_tokens_for_hyp:
                hyp = remove_spaces_and_add_between_special_tokens(hypothesis, hyp_only_has_start, hyp_only_has_end)
            else:
                hyp = hypothesis
            #logging.debug(f'hyp after remove_spaces_and_add_between_special_tokens: {hyp}')

            # Remove prompt tokens like <|startoftranscript|><|en|><|transcribe|><|en|>... but keep timestamps like <|123|>
            hyp = re.sub(r'<[^>0-9]+>', '', hyp)
            # Deal with situation like <unk>123<unk>
            hyp = re.sub(r'(\d+)\s+([\w\']+)\s+(\d+)', r'<|\1|>\2<|\3|>', hyp)
            hyp = re.sub(r'\|(\d+)\|\s+([\w\'\-.]+)\s+\|(\d+)\|', r'<|\1|>\2<|\3|>', hyp)
            # hyp = re.sub(r'<unk>\|(\d+)\|<unk>\s+([\w\']+)\s+<unk>\|(\d+)\|<unk>', r'<|\1|>\2<|\3|>', hyp)
            #logging.debug(f'hyp after <[^>0-9]+> --> "": {hyp}')

            def modify_text(text):
                # Regular expression to match time tokens
                time_token_pattern = re.compile(r"(<\|\d+\|>)")                
                # Remove spaces between a time token and a word
                text = re.sub(r"\s*(<\|\d+\|>)\s*", r"\1", text)
                # Insert a space between adjacent time tokens
                text = re.sub(r"(<\|\d+\|>)(?=<\|\d+\|>)", r"\1 ", text)
                return text
            if not hyp_only_has_start and not hyp_only_has_end:
                hyp = modify_text(hyp)

            #logging.debug(f'ref: {ref}, hyp: {hyp}')

            ref_words = ref.split()
            hyp_words = hyp.split()

            logging.debug(f'ref_words: {ref_words}')
            logging.debug(f'hyp_words: {hyp_words}')

            # Merge any timestamps like <|m|>(punc)<|n|> into neighboring time tokens
            def merge_punctuation(hyp_words):
                merged_words = []
                if hyp_only_has_start:
                    pattern = re.compile(r'(<\|\d+\|>)(\D+)')
                elif hyp_only_has_end:
                    pattern = re.compile(r'(\D+)(<\|\d+\|>)')
                else:
                    pattern = re.compile(r'(<\|\d+\|>)(\D+)(<\|\d+\|>)')
                
                # If no time tokens are found, return the input as is
                if not any(re.match(r'<\|\d+\|>', word) for word in hyp_words):
                    return hyp_words
                
                for i, word in enumerate(hyp_words):
                    # print(f'word: {word}')
                    match = pattern.match(word)
                    if not match:
                        continue
                    
                    if hyp_only_has_start:
                        start_ts, text = match.groups()
                    elif hyp_only_has_end:
                        text, end_ts = match.groups()
                    else:
                        start_ts, text, end_ts = match.groups()
            
                    # print(f'start_ts: {start_ts}, text:{text}, end_ts:{end_ts}')
                    if text in {'.', ',', '!', '?', ':', ';'}:
                        if merged_words:
                            prev_start, prev_text, _ = merged_words.pop()
                            new_word = f"{prev_start}{prev_text}{text}{end_ts}"
                            if hyp_only_has_start:
                                merged_words.append((prev_start, prev_text + text))
                            elif hyp_only_has_end:
                                merged_words.append((prev_text + text, end_ts))
                            else:
                                merged_words.append((prev_start, prev_text + text, end_ts))
                    else:
                        if hyp_only_has_start:
                            merged_words.append((start_ts, text))
                        elif hyp_only_has_end:
                            merged_words.append((text, end_ts))
                        else:
                            merged_words.append((start_ts, text, end_ts))
                        
                if hyp_only_has_start:
                    result = [f"{start}{text}" for start, text in merged_words]
                elif hyp_only_has_end:
                    result = [f"{text}{end}" for text, end in merged_words]
                else:
                    result = [f"{start}{text}{end}" for start, text, end in merged_words]
                return result
            
            hyp_words = merge_punctuation(hyp_words)
            #logging.debug(f'hyp_words after merge punctuation: {hyp_words}')
            
            # Compute the Levenshtein distance and the detailed errors
            # import pdb; pdb.set_trace()
            distance, insertions, deletions, substitutions, tp, fp, fn, time_mismatches, start_diff, end_diff, time_diff_count = (
                levenshtein_distance_with_details(ref_words, hyp_words, hyp_only_has_start, hyp_only_has_end, compute_wer_only=compute_wer_only))
            
            # Compute WER
            wer = distance / len(ref_words)

            start_diff_avg = start_diff / float(time_diff_count + 0.000001)
            end_diff_avg = end_diff / float(time_diff_count + 0.000001)
            #logging.info(
            #    f'hyp: {hyp}, '
            #    f'per-utt wer: {wer}, distance:{distance}, ins:{insertions}, del:{deletions}, sub:{substitutions}, '
            #    f'tp, fp, fn: {tp}, {fp}, {fn}, time_mismatches: {time_mismatches}, precision: {tp/(tp+fp+1e-6)}, recall: {tp/(tp+fn+1e-6)}, '
            #    f'start_diff: {start_diff}, end_diff: {end_diff}, '
            #    f'start_diff_avg: {start_diff_avg}, end_diff_avg: {end_diff_avg}')

            return {
                "wer": wer,
                "insertions": insertions,
                "deletions": deletions,
                "substitutions": substitutions,
                "words": len(ref_words),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "time_mismatches": time_mismatches,
                "avg_start_diff": start_diff_avg,
                "avg_end_diff": end_diff_avg,
            }

        # To get rid of the issue that jiwer does not allow empty string
        # print(r_list)
        # print(h_list)
        #import pdb; pdb.set_trace()
        if len(r_list) == 0:
            if len(h_list) != 0:
                errors = len(h_list)
                ops_count['insertions'] += errors
            else:
                errors = 0
        else:
            if use_cer:
                measures = jiwer.cer(r, h, return_dict=True)
            else:
                if use_jiwer:
                    measures = jiwer.compute_measures(r, h)
                else:
                    measures = word_error_rate(r, h, remove_space_and_handle_special_tokens_for_hyp, hyp_only_has_start, hyp_only_has_end,
                                               hyp_every_n)
            if measures['wer'] >= 0.8:
                print(f"*************************************************************")
                print(f"wer: {measures['wer']}")
                print(f"hyp: {h}")
                print(f"ref: {r}")
                print()
                print()
            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']
            ops_count['insertions'] += measures['insertions']
            ops_count['deletions'] += measures['deletions']
            ops_count['substitutions'] += measures['substitutions']
            if not use_jiwer:
                ops_count['tp'] += measures['tp']
                ops_count['fp'] += measures['fp']
                ops_count['fn'] += measures['fn']
                ops_count['avg_start_diff'] += measures['avg_start_diff']
                ops_count['avg_end_diff'] += measures['avg_end_diff']
                ops_count['time_mismatches'] += measures['time_mismatches']

        scores += errors
        # words += len(r_list)
        # print(f'use_jiwer: {use_jiwer}')
        # print(f'measures: {measures}')
        if not use_jiwer:
            words += measures['words']
        # break

    if use_jiwer:
        return measures['wer'], 0, 0, 0, 0, 0, 0, 0

    if words != 0:
        wer = 1.0 * scores / words
        ins_rate = 1.0 * ops_count['insertions'] / words
        del_rate = 1.0 * ops_count['deletions'] / words
        sub_rate = 1.0 * ops_count['substitutions'] / words
        # timestamp error rate
        ter = 1.0 * ops_count['time_mismatches'] / words
        logging.info(f'TER: {ter:.4f}, total words: {words}')
        # precision and recall
        total_hyp_words = ops_count['tp'] + ops_count['fp']
        precision = 1.0 * ops_count['tp'] / (total_hyp_words + 1e-6)
        total_ref_words = ops_count['tp'] + ops_count['fn']
        recall = 1.0 * ops_count['tp'] / (total_ref_words + 1e-6)
        logging.info(f'overall avg precision: {precision:.4f}, total hyp words: {total_hyp_words}')
        logging.info(f'overall avg recall: {recall:.4f}, total ref words: {total_ref_words}')
        avg_start_diff = ops_count['avg_start_diff'] / len(hypotheses) * 80
        avg_end_diff = ops_count['avg_end_diff'] / len(hypotheses) * 80
        logging.info(f'overall avg start_diffs: {avg_start_diff:.2f} ms')
        logging.info(f"overall avg end_diffs: {avg_end_diff:.2f} ms")
    else:
        wer, ins_rate, del_rate, sub_rate = float('inf'), float('inf'), float('inf'), float('inf')

    return wer, words, ins_rate, del_rate, sub_rate, precision, recall, avg_start_diff, avg_end_diff


def convert_json_timestamps_to_token_format(timestamp_data: Union[List[Dict[str, Any]], Dict[str, Any]], use_start_offset: bool = True, use_end_offset: bool = True) -> str:
    """
    Converts JSON timestamp format to the token format expected by the evaluation script.
    
    Args:
        timestamp_data: Either a list of word dictionaries or a dict containing a 'word' key with the list
        use_start_offset: Whether to use start_offset for start timestamps
        use_end_offset: Whether to use end_offset for end timestamps
    
    Returns:
        String in the format expected by the evaluation script
    """
    # print(f"*******************************************************************")
    # print(f"*******************************************************************")
    # print(f"*******************************************************************")
    # print(f"*******************************************************************")
    # print(f"*******************************************************************")
    # print(f"*******************************************************************")
    # print(f"timestamp_data {timestamp_data}")
    # Handle nested structure where word list is under 'word' key
    if isinstance(timestamp_data, dict) and 'word' in timestamp_data:
        word_list = timestamp_data['word']
    elif isinstance(timestamp_data, list):
        word_list = timestamp_data
    else:
        raise ValueError(f"Unexpected timestamp_data format: {type(timestamp_data)}")
    
    result_parts = []
    
    for word_info in word_list:
        word = word_info["word"]
        
        if use_start_offset and use_end_offset:
            start_frame = word_info["start_offset"]
            end_frame = word_info["end_offset"]
            result_parts.append(f"<|{start_frame}|>{word}<|{end_frame}|>")
        elif use_start_offset:
            start_frame = word_info["start_offset"]
            result_parts.append(f"<|{start_frame}|>{word}")
        elif use_end_offset:
            end_frame = word_info["end_offset"]
            result_parts.append(f"{word}<|{end_frame}|>")
        else:
            result_parts.append(word)
    
    return " ".join(result_parts)


def load_json_and_match(hyp_json_path: str, ref_json_path: str, hyp_field: str, ref_field: str, 
                       use_jiwer: bool = False, hyp_timestamp_field: str = None, ref_timestamp_field: str = None,
                       use_start_offset: bool = True, use_end_offset: bool = True, 
                       hyp_audio_id_field: str = "audio_filepath", ref_audio_id_field: str = "audio_filepath"):
    """
    Loads two JSON files (newline-delimited JSON), keeps the order of the reference file, 
    and extracts the text fields specified by 'hyp_field' and 'ref_field'. Optionally removes 
    text like <|x|> where x is an integer in both hypotheses and references.
    
    Args:
        hyp_json_path (str): Path to the hypothesis JSON file.
        ref_json_path (str): Path to the reference JSON file.
        hyp_field (str): Field name in the hypothesis JSON that contains the text to evaluate.
        ref_field (str): Field name in the reference JSON that contains the text to evaluate.
        use_jiwer (bool): If True, remove text like <|x|> in both hypotheses and references.
        hyp_timestamp_field (str): Field name containing timestamp information in hypothesis JSON.
        ref_timestamp_field (str): Field name containing timestamp information in reference JSON.
        use_start_offset (bool): Whether to use start_offset for timestamps.
        use_end_offset (bool): Whether to use end_offset for timestamps.
        hyp_audio_id_field (str): Field name containing the audio identifier in hypothesis JSON.
        ref_audio_id_field (str): Field name containing the audio identifier in reference JSON.

    Returns:
        Tuple[List[str], List[str]]: Matched lists of hypotheses and reference texts.
    """

    # Create a function to remove <|x|> patterns
    def remove_timestamp_tokens(text: str) -> str:
        return re.sub(r"<\|.*?\|>", "", text)

    # Load reference file as a list and remove optional timestamp tokens
    refs = []
    with open(ref_json_path, 'r') as ref_file:
        for line in ref_file:
            ref = json.loads(line.strip())  # Parse each line as a separate JSON record
            
            # Handle timestamp conversion for reference
             
            if ref_timestamp_field and ref_timestamp_field in ref:
                ref_text = convert_json_timestamps_to_token_format(
                    ref[ref_timestamp_field], use_start_offset, use_end_offset)
            else:
                ref_text = ref[ref_field]
                
            if use_jiwer:
                ref_text = remove_timestamp_tokens(ref_text)
            
            # Use the specified audio ID field, with fallback to basename if it's a path
            audio_id = ref[ref_audio_id_field]
            if '/' in audio_id:
                # Extract basename and remove extension for matching
                audio_id = os.path.basename(audio_id)
                if '.' in audio_id:
                    audio_id = os.path.splitext(audio_id)[0]  # Remove file extension
            refs.append((audio_id, ref_text))  # Keep order

    # Initialize list to hold hypotheses in the same order as references
    hyps = [''] * len(refs)

    # Open and read the hypothesis JSON file line by line (NDJSON format)
    hyp_data = {}
    with open(hyp_json_path, 'r') as hyp_file:
        for line in hyp_file:
            hyp_record = json.loads(line.strip())
            
            # Use the specified audio ID field, with fallback to basename if it's a path
            audio_id = hyp_record[hyp_audio_id_field]
            if '/' in audio_id:
                # Extract basename and remove extension for matching
                audio_id = os.path.basename(audio_id)
                if '.' in audio_id:
                    audio_id = os.path.splitext(audio_id)[0]  # Remove file extension
            
            # Handle timestamp conversion for hypothesis
           # print(f"hyp_timestamp_field {hyp_timestamp_field}")
            if hyp_timestamp_field and hyp_timestamp_field in hyp_record:
                hyp_text = convert_json_timestamps_to_token_format(
                    hyp_record[hyp_timestamp_field], use_start_offset, use_end_offset)
            else:
                #hyp_text = hyp_record[hyp_field]
                hyp_text = convert_json_timestamps_to_token_format(hyp_record, use_start_offset, use_end_offset)
                
            hyp_data[audio_id] = hyp_text

    # Match hypotheses to references in the same order
    for idx, (ref_basename, _) in enumerate(refs):
        #print(f"********************************************************************************************")
        #print(f"********************************************************************************************")
        #print(f"********************************************************************************************")
        #print(f"********************************************************************************************")
        #print(f"********************************************************************************************")
        #print(f"ref_basename {ref_basename}")
        if ref_basename in hyp_data:
            #print(f"ref_basename in hyp_dat")
            hyp_text = hyp_data[ref_basename]
            if use_jiwer:
                hyp_text = remove_timestamp_tokens(hyp_text)
            hyps[idx] = hyp_text
        else:
            print(f"Warning: No matching hypothesis found for {ref_basename}")

    # Extract references in order to return
    refs_ordered = [ref_text for _, ref_text in refs]



    return hyps, refs_ordered


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate hypotheses and references for WER calculation")
    
    # Command line arguments for JSON files and text fields
    parser.add_argument("--hyp_json", required=True, help="Path to the hypotheses JSON file")
    parser.add_argument("--ref_json", required=True, help="Path to the references JSON file")
    parser.add_argument("--hyp_field", required=True, help="Field name in the hypothesis JSON containing the text")
    parser.add_argument("--ref_field", required=True, help="Field name in the reference JSON containing the text")
    parser.add_argument("--hyp_timestamp_field", required=False, help="Field name in the hypothesis JSON containing timestamp information")
    parser.add_argument("--ref_timestamp_field", required=False, help="Field name in the reference JSON containing timestamp information")
    parser.add_argument("--use_start_offset", type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help="Use start_offset for timestamps")
    parser.add_argument("--use_end_offset", type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help="Use end_offset for timestamps")
    parser.add_argument("--hyp_audio_id_field", required=False, default="audio_filepath", help="Field name containing the audio identifier in hypothesis JSON (default: audio_filepath)")
    parser.add_argument("--ref_audio_id_field", required=False, default="audio_filepath", help="Field name containing the audio identifier in reference JSON (default: audio_filepath)")
    parser.add_argument("--only_start", required=False, help="Only predicting start timestamp")
    parser.add_argument("--only_end", required=False, help="Only predicting end timestamp") 
    parser.add_argument("--has_timestamp", type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help="Ref or hyp has timestamps like <||>")
    parser.add_argument("--compute_wer_only", required=False, help="Only compute WER and not consider timestamps.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    use_jiwer = False
    time_diff_threshold = 3
    # basically not considering time difference
    # time_diff_threshold = 100000

    hyp_json_path = args.hyp_json
    ref_json_path = args.ref_json
    hyp_field = args.hyp_field
    ref_field = args.ref_field
    hyp_timestamp_field = args.hyp_timestamp_field
    ref_timestamp_field = args.ref_timestamp_field
    use_start_offset = args.use_start_offset
    use_end_offset = args.use_end_offset
    hyp_audio_id_field = args.hyp_audio_id_field
    ref_audio_id_field = args.ref_audio_id_field
    only_start = args.only_start
    only_end = args.only_end
    has_timestamp = args.has_timestamp
    compute_wer_only = args.compute_wer_only

    # Load and match hypotheses and references based on audio_filepath
    hyps, refs = load_json_and_match(hyp_json_path, ref_json_path, hyp_field, ref_field, 
                                   use_jiwer=use_jiwer, hyp_timestamp_field=hyp_timestamp_field, 
                                   ref_timestamp_field=ref_timestamp_field, 
                                   use_start_offset=use_start_offset, use_end_offset=use_end_offset,
                                   hyp_audio_id_field=hyp_audio_id_field, ref_audio_id_field=ref_audio_id_field)

    # Compute WER and other metrics
    remove_space_and_handle_special_tokens_for_hyp = True #if only_start or only_end else False
    wer, total_words, ins_rate, del_rate, sub_rate, precision, recall, avg_start_diff, avg_end_diff = word_error_rate_detail(
        hyps, refs, has_timestamp=has_timestamp, use_jiwer=use_jiwer, time_diff_threshold=time_diff_threshold, 
        remove_space_and_handle_special_tokens_for_hyp=remove_space_and_handle_special_tokens_for_hyp, 
        hyp_only_has_start=only_start, hyp_only_has_end=only_end, compute_wer_only=compute_wer_only)

    # Output the results
    # print(f"=========Summary=========")
    # print(f"WER: {wer}")
    # print(f"precision: {precision:.2f}")
    # print(f"recall: {recall:.2f}")
    
    # print(f"Total words: {total_words}")
    # print(f"Insertion rate: {ins_rate:.2f}")
    # print(f"Deletion rate: {del_rate:.2f}")
    # print(f"Substitution rate: {sub_rate:.2f}")

    # print(f'overall avg start_diffs: {avg_start_diff:.2f} ms')
    # print(f"overall avg end_diffs: {avg_end_diff:.2f} ms")

    print("\n========= Summary =========\n")
    print(f"Word Error Rate (%):         {wer*100:.2f}")
    print("\n------- Word Analysis -------")
    print(f"Total Words:                 {total_words}")
    print(f"Insertion Rate (%):          {ins_rate*100:.2f}")
    print(f"Deletion Rate (%):           {del_rate*100:.2f}")
    print(f"Substitution Rate (%):       {sub_rate*100:.2f}")
    print("\n------ Timing Analysis ------")
    print(f"Precision (%):               {precision*100:.2f}")
    print(f"Recall (%):                  {recall*100:.2f}")
    print(f"Average Start Diff:          {avg_start_diff:.2f} ms")
    print(f"Average End Diff:            {avg_end_diff:.2f} ms")
    print("\n=============================")
