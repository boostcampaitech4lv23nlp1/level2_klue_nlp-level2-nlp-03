#!/bin/bash
CONFIGS=("T5/T5_large_GRU" "T5/T5_large_GRU3e-5" "T5/T5_large_LSTM" "T5/T5_large_LSTM3e-5")
WANDB=("init" "init" "init" "init")

for (( i=0; i<4; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]} -- wandb ${WANDB[$i]}
done