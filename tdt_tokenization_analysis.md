# TDT Tokenization for Viterbi Alignment - Key Insights

## The Core Problem
You need to align **text predictions from another model** with **TDT logits** using Viterbi decoding.

## Key Differences: CTC vs TDT Tokenization

### CTC Tokenization (Current `get_utt_obj`)
```python
# CTC Structure: [BLANK, token1, BLANK, token2, BLANK, ...]
# Length = 2 * num_tokens + 1
token_ids_with_blanks = [BLANK_ID]
for token in tokens:
    token_ids_with_blanks.extend([token_id, BLANK_ID])
```

### TDT Tokenization (What You Need)
```python
# TDT Structure: Tokens with explicit durations
# Key insight: TDT logits contain BOTH token predictions AND duration predictions
# The logits shape is typically [T, V + num_durations] where:
# - V = vocabulary size (including blank)
# - num_durations = number of possible duration values

# TDT doesn't need blanks between EVERY token like CTC
# Instead, it predicts how long each token lasts
```

## Critical TDT-Specific Details

### 1. Blank ID Position
```python
# CTC: blank_id = len(vocabulary) + extra_outputs  
# TDT: blank_id = len(vocabulary)  # No extra outputs for blank
```

### 2. Logits Structure
```python
# TDT logits shape: [T, V + num_durations]
# Last `num_durations` dimensions are duration predictions
# First `V` dimensions are token predictions (including blank)
```

### 3. Duration Handling
```python
# TDT durations are typically: [0, 1, 2, 4, 8, ...] or similar
# Each token gets assigned a duration that determines how many frames it spans
```

## Practical Implementation Strategy

### Option 1: Minimal Changes (Recommended)
Modify the existing `get_utt_obj` to handle TDT by:
1. Adjusting blank_id calculation
2. Changing token sequence structure
3. Accounting for duration predictions in logits

### Option 2: Separate TDT Function
Create `get_utt_obj_tdt` with TDT-specific logic.

## Key Questions to Resolve

1. **What are the possible duration values** in your TDT model? (e.g., [0,1,2,4,8])
2. **What's the exact shape** of your TDT logits? (T x V+durations)
3. **How should blanks be placed** for optimal Viterbi alignment?
4. **Do you want to modify existing code** or create new TDT-specific functions?

## Recommended Next Steps

1. **Check your TDT model's duration configuration**:
   ```python
   durations = model.cfg.get('durations', [0, 1, 2, 4, 8])  # Example
   ```

2. **Verify logits shape**:
   ```python
   print(f"TDT logits shape: {logits.shape}")  # Should be [T, V + len(durations)]
   ```

3. **Implement TDT tokenization** with proper blank placement for your specific use case.
