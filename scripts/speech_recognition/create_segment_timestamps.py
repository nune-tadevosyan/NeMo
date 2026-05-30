#!/usr/bin/env python3
"""
Create segment-level timestamps from word-level timestamps in generations.jsonl.

Input: JSONL file where each line has:
  - id, duration, pred_text, word (list of {word, start, end, start_offset, end_offset})

Output: JSONL file with:
  - audio_filepath, text, timestamped_text, offset_text, duration
"""

import json
import argparse
import re


def split_into_sentences(words):
    """Group words into sentence segments based on sentence-ending punctuation."""
    sentences = []
    current = []

    for word_info in words:
        current.append(word_info)
        stripped = word_info["word"].strip()
        if stripped and stripped[-1] in ".?!":
            sentences.append(current)
            current = []

    if current:
        sentences.append(current)

    return sentences


def sentence_text(words):
    """Reconstruct sentence text from word entries, stripping leading whitespace."""
    text = "".join(w["word"] for w in words)
    return text.strip()


def fmt_ts(val):
    return f"{val:.2f}"


def process_entry(entry):
    words = entry.get("word", [])
    if not words:
        return None

    entry_id = entry["id"]
    duration = entry["duration"]
    pred_text = entry.get("pred_text", "")
    audio_filepath = f"wavs/{entry_id}.wav"

    sentences = split_into_sentences(words)

    ts_parts = []
    off_parts = []

    for sent_words in sentences:
        text = sentence_text(sent_words)
        start_time = sent_words[0]["start"]
        end_time = sent_words[-1]["end"]
        start_off = sent_words[0]["start_offset"]
        end_off = sent_words[-1]["end_offset"]

        ts_parts.append(f"<|{fmt_ts(start_time)}|>{text}<|{fmt_ts(end_time)}|>")
        off_parts.append(f"<|{start_off}|>{text}<|{end_off}|>")

    return {
        "audio_filepath": audio_filepath,
        "text": pred_text,
        "timestamped_text": " ".join(ts_parts),
        "offset_text": " ".join(off_parts),
        "duration": duration,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create segment-level timestamps from word-level timestamps"
    )
    parser.add_argument("--input", "-i", required=True, help="Input generations.jsonl")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    args = parser.parse_args()

    count = 0
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            result = process_entry(entry)
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1

    print(f"Processed {count} entries -> {args.output}")


if __name__ == "__main__":
    main()
