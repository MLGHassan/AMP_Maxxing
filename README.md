# DeepAMP: learning to score and search for plant-derived antimicrobial peptides

EthnoHack 2026 · Track: Natural products and drug discovery · CS/CE 351/363 Artificial Intelligence, Fall 2026 · Group 34
Team: Syed Hassan Askari Zaidi (sz10164) · Saqib Rehman (sr10128) · Miqdad Yusuf Hayati (mh10011)

> **All outputs are computational hypotheses. Nothing here has been tested in a laboratory.**
> A predicted probability is not evidence of antimicrobial activity, selectivity or safety.

## 1. What this project does

1. **Scorer**: predicts P(antimicrobial) for a 10-50 residue peptide. Six models are compared under one identical,
   homology-aware split: charge-only, logistic regression and random forest on physicochemical descriptors, a CNN-BiLSTM on
   raw sequence, and frozen ESM-2 (8M) embeddings with a mean-pool or an attention-pool head.
2. **Searcher**: simulated annealing from real plant AMPs, BLOSUM62-weighted single-residue substitutions (max 5 per peptide),
   fitness = scorer probability minus a charge/hydrophobicity penalty.
3. **Independent judge**: models trained on protein clusters the scorer never saw, never loaded by the search. Candidates are
   compared with two controls: the unmodified plant seeds and random mutants with the same number of substitutions.

## 2. Quick start (step by step)

### 2.0 Requirements
- macOS or Linux, **Python 3.10 or newer** (tested on 3.13), about 2 GB free disk, internet access
  (UniProt REST API, and Hugging Face for the first ESM-2 download of about 30 MB).
- Optional: [CD-HIT](https://github.com/weizhongli/cdhit) on your PATH. If present it is used automatically; otherwise a built-in
  clustering routine with the same identity definition is used (slower, fine at this scale).
- A GPU is not needed. Apple Silicon (MPS) and CUDA are used automatically if available; CPU works.

### 2.1 Set up
```bash
git clone <YOUR-REPO-URL> deepamp && cd deepamp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Check the install (about 1 minute, no internet needed)
```bash
python tests/smoke_test.py
```
It must end with `SMOKE TEST PASSED`. It runs every stage on tiny stand-in data with a stub in place of ESM-2, so its numbers
are meaningless; it only proves the code paths work on your machine.

### 2.3 Run everything with one command
```bash
bash run_all.sh                    # or:  PY=python3.13 bash run_all.sh
```
This runs steps 2.4 to 2.12 in order. If a step fails, fix it and run that step by hand; earlier outputs are kept.

### 2.4 Download real data
```bash
python fetch_real_data.py
```
Pulls reviewed UniProt peptides and cytoplasmic proteins, builds both negative sets and writes
`data/processed/sequences.tsv`, `data/processed/plant_seeds.fasta` and `data/processed/DATA_SUMMARY.json` (queries, UniProt
release, date, counts). **Expect** about 1,000 unique positives, about 120 plant seeds, and `n_pos_unmatched_in_hard` close to 0.
Counts change slightly with each UniProt release; our runs used release 2026_04.

### 2.5 Cluster and split
```bash
python cluster_and_split.py
```
Clusters everything at 40% identity and assigns whole clusters to `test` (15%), `scorer` (42.5%) and `judge` (42.5%), balanced on
both rows and positives. Writes `data/processed/dataset.tsv`. **Expect** about 1,170 clusters and, in the printed table, roughly
150 positives in `test` and about 435 in each of `scorer` and `judge`.

### 2.6 Descriptor baselines
```bash
python baselines.py --seeds 5
```
Writes `results/baselines.json` and `results/baselines.csv` (both negative sets, cluster split and size-matched random split).

### 2.7 Neural scorer ladder (rungs 2-4)
```bash
for neg in random hard; do for m in cnn_lstm esm_mean esm_attn; do
  python train.py --role scorer --model $m --neg $neg --seeds 5
done; done
```
Each run prints per-seed metrics and a `MEAN±SD` line, writes `results/scorer_<model>_<neg>_cluster.json` and five checkpoints
`checkpoints/scorer_<model>_<neg>_s0..s4.pt`. The first ESM run downloads the weights and caches embeddings in `cache/`.
The Hugging Face warnings about `lm_head` / `pooler` weights and about unauthenticated requests are harmless.

### 2.8 Leakage check
```bash
python train.py --role scorer --model esm_attn --neg hard --mode random --seeds 5
```
Same model, but the split ignores clusters while keeping the same train/val/test sizes. Saves no checkpoints.

### 2.9 Train the independent judges
```bash
python train.py --role judge --model esm_attn --neg hard --seeds 5
python train.py --role judge --model cnn_lstm --neg hard --seeds 5
```
Judges are trained on the `judge` clusters, which the scorer never sees.

### 2.10 Search for candidates
```bash
python search.py --scorer checkpoints/scorer_esm_attn_hard_s0.pt --n_seeds 40
```
Seeds are plant AMPs from clusters the judge never saw. The scorer is ESM attention-pool, seed 0, chosen in advance, not by
test score. Writes `outputs/candidates.csv` and `outputs/candidates.fasta`.

### 2.11 Audit candidates with the judges
```bash
python evaluate_candidates.py --judge_glob "checkpoints/judge_esm_attn_hard_s*.pt"
python evaluate_candidates.py --judge_glob "checkpoints/judge_cnn_lstm_hard_s*.pt" --tag _cnn
```
Prints mean judge probability for seeds, random mutants and candidates, plus bootstrap 95% intervals over seeds for the
differences. Writes `results/candidate_audit*.csv` and the ranked library `outputs/candidate_library_ranked*.fasta`.

### 2.12 Tables and figures
```bash
python summarize.py
```
Writes `results/summary.md` and `figures/ladder_mcc.png`, `figures/candidate_audit.png`, `figures/candidate_audit_cnn.png`.

## 3. Outputs

| file | content |
|---|---|
| `results/summary.md` | ladder tables (both negative sets) and leakage check |
| `results/candidate_audit.csv`, `_cnn.csv` | each candidate: scorer P, judge P, controls, charge, hydrophobic moment, edit distance to nearest known AMP |
| `outputs/candidate_library_ranked.fasta` | candidate sequences with all of the above in the header |
| `data/processed/DATA_SUMMARY.json` | exact queries, UniProt release, retrieval date, counts |
| `figures/*.png` | model-ladder bar chart and judge-score box plots |

## 4. Results from our run (UniProt release 2026_04, Apple Silicon)
Numbers vary slightly with UniProt release, hardware and library versions; expect the same ordering, not identical digits.

**Hard (charge-matched) negatives, held-out clusters, mean ± sd over 5 seeds**

| model | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|
| charge only | 0.568 | 0.543 | 0.066 ± 0.042 |
| logistic regression | 0.818 | 0.814 | 0.480 ± 0.019 |
| random forest | 0.870 | 0.880 | 0.544 ± 0.035 |
| CNN-BiLSTM | 0.914 | 0.922 | 0.696 ± 0.053 |
| ESM-2 + attention pool | 0.924 | 0.936 | 0.748 ± 0.035 |
| ESM-2 + mean pool | 0.941 | 0.950 | 0.781 ± 0.039 |

**Random negatives:** charge only MCC 0.485 · logistic regression 0.607 · random forest 0.644 · CNN-BiLSTM 0.712 · ESM attention 0.768 · ESM mean 0.779.

**Search audit** (mean judge probability; differences with 95% bootstrap interval over seeds)

| judge | plant seeds | random mutants | candidates | candidates − seeds | candidates − random mutants |
|---|---|---|---|---|---|
| ESM-attention ensemble | 0.784 | 0.742 | 0.823 | +0.039 [+0.008, +0.075] | +0.081 [+0.040, +0.128] |
| CNN-BiLSTM ensemble | 0.806 | 0.760 | 0.780 | −0.027 [−0.066, +0.002] | +0.020 [−0.019, +0.058] |

How to read this: annealing beats random mutation and slightly beats the seeds under the ESM judge, but the raw-sequence
judge does not reproduce that gain. We report both.

## 5. Evaluation protocol
- All sequences (positives and both negative sets) are clustered together at 40% identity; whole clusters are assigned to groups,
  so near-duplicates never straddle train and test.
- Within a pool, clusters are split train/validation (85/15) with a different partition per seed. The epoch is chosen on validation
  ROC-AUC and the decision threshold on validation MCC; the test group plays no part in selection.
- Metrics: ROC-AUC, PR-AUC, MCC, sensitivity, specificity. The ± values reflect re-partitioning of train/validation and
  initialisation only; the test group is the same for every seed, so they understate total uncertainty.
- Leakage check: a size-matched random split. On this data it inflated scores only modestly (for example random forest MCC 0.544 → 0.602 on hard negatives).

## 6. Data
| item | source | notes |
|---|---|---|
| positives | UniProtKB/Swiss-Prot, reviewed, 10-50 aa, keyword Antimicrobial (KW-0929) or Antibiotic (KW-0044) | CC BY 4.0; curated annotation, not guaranteed experimental measurement |
| plant seeds | same query, Viridiplantae (taxonomy 33090) | accession and species in FASTA header |
| random negatives | length-matched fragments of reviewed cytoplasmic proteins, not annotated antimicrobial or secreted | Veltri et al. 2018 recipe |
| hard negatives | as above, also matched on net charge at pH 7 (±1, widened to ±3 if needed) | unmatched positives are counted in `DATA_SUMMARY.json` |

More validated sequences (for example DRAMP or DBAASP, check their licences) can be added with
`python fetch_real_data.py --extra_pos_fasta your_file.fasta`.

## 7. Limitations
- Labels are database annotations; negatives are assumed inactive, not measured inactive.
- Neural scores barely dropped from random to hard negatives. Charge matching removes the charge shortcut, but the models can
  probably still separate mature peptides from internal fragments of globular proteins. Negatives made of non-antimicrobial peptides would be a tougher test.
- The ESM judge shares frozen ESM-2 features with the scorer: independent in training data, not in every inductive bias.
- Cysteine-rich plant peptides (defensins, thionins, cyclotides) depend on disulfide bonds or cyclisation that a linear sequence model cannot represent.
- Novelty is edit distance to known AMPs in our data (median 5, equal to the mutation cap), not BLAST against all of UniProt.
- Toxicity, haemolysis and selectivity are not modelled. Deviation from the written proposal: repeated cluster-grouped hold-out (5 seeds) instead of 5-fold cross-validation.

## 8. Repository layout
```
fetch_real_data.py        UniProt download, negative sets
cluster_and_split.py      40% identity clustering, cluster-level group assignment
baselines.py              charge-only, logistic regression, random forest
train.py                  CNN-BiLSTM, ESM mean-pool, ESM attention-pool (scorer or judge)
models.py, esm_embed.py   model heads, frozen ESM-2 embedder with cache
search.py                 simulated annealing
evaluate_candidates.py    judge audit with controls and bootstrap intervals
summarize.py              tables and figures
common.py                 descriptors, metrics, partitioning
tests/smoke_test.py       end-to-end check on stand-in data
run_all.sh                whole pipeline
```

## 9. Troubleshooting
- **UniProt download fails or times out:** rerun `python fetch_real_data.py`; the script retries each request. Check your connection.
- **`no judge checkpoints match ...`:** run step 2.9 first. **Scorer checkpoint not found:** run step 2.7 first.
- **MPS errors on a Mac:** run `PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py ...`, or force CPU with `CUDA_VISIBLE_DEVICES=""` on Linux.
- **Clustering is slow:** install CD-HIT, or ignore; the built-in method finishes in a few minutes for about 3,000 sequences.
- **Different numbers from ours:** expected if the UniProt release differs; compare `DATA_SUMMARY.json`.

## 10. Licences, credits and AI disclosure
- Code: MIT (see `LICENSE`). Data: UniProt, CC BY 4.0; please cite UniProt. Model: ESM-2 `facebook/esm2_t6_8M_UR50D` (MIT).
- Libraries: PyTorch, Hugging Face Transformers, scikit-learn, Biopython, pandas, matplotlib, rapidfuzz, CD-HIT (optional).
- Method inspiration: Veltri, Kamath and Shehu 2018 (Bioinformatics); Lin et al. 2023 (Science, ESM-2); Porto et al. 2018 (Nature Communications).
- AI assistance: parts of the code were written with an AI assistant (Claude), then run, tested and reviewed by the team. <!-- edit to state exactly what your team did yourselves -->