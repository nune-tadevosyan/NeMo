#!/usr/bin/env bash

NEMO_DIR=/lustre/fsw/portfolios/convai/users/ntadevosyan/code/dcc_uni/NeMo
export PYTHONPATH="$NEMO_DIR:$PYTHONPATH"

python $NEMO_DIR/scripts/speech_recognition/estimate_duration_bins_2d.py \
    -b 8 \
    -s 2 \
    -n 200000 \
    -t /lustre/fsw/portfolios/convai/users/ntadevosyan/data/asr3/tokenizer/tokenizer_spe_bpe_v1024/tokenizer.model \
    --text-field text \
    input_cfg.yaml 
    #'[["/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/NeMo_ASR_SET/English/v3.0/train_bucketed//bucket1/sharded_manifests/manifest__OP_0..8191_CL_.json"],["/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/NeMo_ASR_SET/English/v3.0/train_bucketed//bucket2/sharded_manifests/manifest__OP_0..8191_CL_.json"],["/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/NeMo_ASR_SET/English/v3.0/train_bucketed//bucket3/sharded_manifests/manifest__OP_0..8191_CL_.json"], ["/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket4/sharded_manifests/manifest__OP_0..8191_CL_.json"], ["/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket5/sharded_manifests/manifest__OP_0..8191_CL_.json"], ["/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket6/sharded_manifests/manifest__OP_0..8191_CL_.json"], ["/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket7/sharded_manifests/manifest__OP_0..8191_CL_.json"], ["/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket8/sharded_manifests/manifest__OP_0..8191_CL_.json"]]'

    #-t /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/canary/canary_v0/tokenizers/unkfix/spe_bpe_v1024/tokenizer_{en,de,fr,es}_1024/tokenizer_spe_bpe_v1024_max_4/tokenizer.model \
    #-a en de fr es \

