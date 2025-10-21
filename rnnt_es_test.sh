#!/bin/bash

#!/bin/bash

# List of directories to search
search_dirs=(
    "/home/ntadevosyan/models/150_default_ASR3_l_4k"
    "/home/ntadevosyan/models/150_mamba_ASR3_l_4k"
)

# List of manifest file paths
manifest_filepaths=(
    "/home/ntadevosyan/Documents/librispeech/test_clean.json"
    "/home/ntadevosyan/Documents/librispeech/test_other_manifest.json"
    "/home/ntadevosyan/data/earnings_align/test_earnings21_5h.json"
    "/home/ntadevosyan/data/SPGI/audio_manifest_test_clean.json"
    # Add more manifest file paths here if needed
)

# Iterate over each search directory
for search_dir in "${search_dirs[@]}"; do
    # Find the .nemo file in the current search directory (assuming only one .nemo file)
    model_file=$(find "$search_dir" -name "*.nemo" | head -n 1)
    
    # Check if a .nemo file was found
    if [ -n "$model_file" ]; then
        # Define the output file for the current model
        output_file="$(dirname "$model_file")/eval.txt"
        echo "" > "$output_file"  # Clear the output file if it exists
        
        # Iterate over each manifest file path
        for manifest_filepath in "${manifest_filepaths[@]}"; do
            # Extract manifest filename without extension
            manifest_filename=$(basename "$manifest_filepath" .json)
            
            # Run the Python script and filter the output
            python_output=$(python3 /home/ntadevosyan/code/NeMo/examples/asr/speech_to_text_eval.py \
                model_path="$model_file" \
                dataset_manifest="$manifest_filepath" | grep 'Dataset WER')
            
            # Check if there are any "best" lines and append them to the output file
            if [ -n "$python_output" ]; then
                echo "greedy for ${manifest_filename}:" >> "$output_file"
                echo "$python_output" >> "$output_file"
                echo "" >> "$output_file"
            fi
        done
    else
        echo "No .nemo file found in $search_dir"
    fi
done

