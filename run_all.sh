#!/usr/bin/env bash
# Runs the complete DeepAMP pipeline, from raw UniProt download to figures.
# Usage:  bash run_all.sh        (override the interpreter with  PY=python3.13 bash run_all.sh)
set -euo pipefail
PY="${PY:-python3}"
mkdir -p data/raw data/processed checkpoints results outputs figures cache

$PY fetch_real_data.py                      # 1. UniProt -> data/processed/sequences.tsv (+ plant_seeds.fasta)
$PY cluster_and_split.py                    # 2. 40% identity clusters -> data/processed/dataset.tsv
$PY baselines.py --seeds 5                  # 3. charge-only, logistic regression, random forest

for neg in random hard; do                  # 4. neural ladder, both negative sets
  for m in cnn_lstm esm_mean esm_attn; do
    $PY train.py --role scorer --model $m --neg $neg --seeds 5
  done
done
$PY train.py --role scorer --model esm_attn --neg hard --mode random --seeds 5   # 5. leakage check (no checkpoints saved)

$PY train.py --role judge --model esm_attn --neg hard --seeds 5                  # 6. judges (disjoint clusters)
$PY train.py --role judge --model cnn_lstm --neg hard --seeds 5

$PY search.py --scorer checkpoints/scorer_esm_attn_hard_s0.pt --n_seeds 40       # 7. simulated annealing
$PY evaluate_candidates.py --judge_glob "checkpoints/judge_esm_attn_hard_s*.pt"  # 8. audit with judge 1
$PY evaluate_candidates.py --judge_glob "checkpoints/judge_cnn_lstm_hard_s*.pt" --tag _cnn   # judge 2
$PY summarize.py                            # 9. results/summary.md + figures/
