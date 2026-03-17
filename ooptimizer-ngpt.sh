#!/usr/bin/env bash

NEMO_DIR=/lustre/fsw/portfolios/convai/users/ntadevosyan/code/dcc_unified/NeMo/
export PYTHONPATH="$NEMO_DIR:$PYTHONPATH"

BUCKETS='[[5.760,62],[7.120,77],[8.320,83],[9.440,92],[10.500,103],[11.680,111],[12.880,117],[14.080,130],[15.440,138],[17.200,156],[19.360,158],[22.400,189],[26.640,217],[32.800,272],[40.100,352]]'

# Note: --no-ddp can be set to remove ddp overhead from computation due to gradient_as_bucket_view=True in YAML config
python $NEMO_DIR/scripts/speech_recognition/oomptimizer.py \
    -m nemo.collections.asr.models.EncDecRNNTBPEModel\
    -c /lustre/fsw/portfolios/convai/users/ntadevosyan/projects/mamba/conf/andrei_fc-xl_rnnt_pnc_r-data_init-uni-best_0.5-sym_conv31.yaml \
    --no-ddp \
    -b "$BUCKETS"

