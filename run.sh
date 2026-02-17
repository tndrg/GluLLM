#!/bin/bash

# Train and test GluLLM
python main.py \
    --ds REPLACE-BG \
    --mn llama1 \
    --seq_len 288 \
    --seq_len 72 \
    --token_len 12 \
    --test_pred_len 72 \
    --train_epochs 20 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --weight_decay 0.0 \
    --patience 5 \
    --use_amp \
    --checkpoint_dir ./checkpoints \
    --results_dir ./test_results \
    --mode train_test \
    --seed 2026 \
    --num_workers 8 \
    --data_base /novo/projects/departments/nnrco/genetic_department/share/ZTZU_share/storage/glullm/data/ \
    --cache_dir /novo/projects/departments/nnrco/genetic_department/share/ZTZU_share/storage/glullm/llm_cache \