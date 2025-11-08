#!/bin/bash

DATA_FILE="./data/ABIDE/gnn_preprocessed/ABIDE_gnn_data_*.pkl"

for FC in pearson ledoit_wolf; do
    for FEAT in statistical temporal; do
        for MODEL in gcn gat linear; do
            echo "Running: FC=$FC, Feature=$FEAT, Model=$MODEL"
            python gnn_validation.py \
                --data_file $DATA_FILE \
                --fc_method $FC \
                --feature_method $FEAT \
                --model $MODEL \
                --n_folds 5 \
                --n_repeats 10 \
                --device cuda
        done
    done
done

echo "All experiments completed!"
