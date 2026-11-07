#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw data/processed checkpoints results outputs figures cache

python fetch_real_data.py                      # UniProt -> data/processed/sequences.tsv (+ plant_seeds.fasta)
python cluster_and_split.py                    # 40% identity clusters -> data/processed/dataset.tsv
python baselines.py --seeds 5                  # rung 1 + charge-only floor, cluster vs random split

for neg in random hard; do                     # rungs 2-4, both negative sets
  for m in cnn_lstm esm_mean esm_attn; do
    python train.py --role scorer --model $m --neg $neg --seeds 5
  done
done
python train.py --role scorer --model esm_attn --neg hard --mode random --seeds 5   # leakage demo (no checkpoints saved)

python train.py --role judge --model esm_attn --neg hard --seeds 5                  # judge: disjoint clusters
python train.py --role judge --model cnn_lstm --neg hard --seeds 5                  # second judge: raw-sequence model

python search.py --scorer checkpoints/scorer_esm_attn_hard_s0.pt --n_seeds 25
python evaluate_candidates.py --judge_glob "checkpoints/judge_esm_attn_hard_s*.pt"
python evaluate_candidates.py --judge_glob "checkpoints/judge_cnn_lstm_hard_s*.pt" --tag _cnn
python summarize.py
