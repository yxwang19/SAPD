#!/bin/bash

for dataset in OECDSuzhou; do
  for seed in $(seq 0 4); do
    echo "Running dataset=$dataset, seed=$seed"
    if [ "$dataset" = "EQSQ" ]; then
      option_num=4
    else
      option_num=5
    fi
    python main.py \
      --method=sapd \
      --datatype=$dataset \
      --test_size=0.2 \
      --seed=$seed \
      --device=cuda:0 \
      --epoch=10 \
      --batch_size=1024 \
      --lr=0.003 \
      --option_num=$option_num \
      --gnn_type=LightGCN
  done
done
