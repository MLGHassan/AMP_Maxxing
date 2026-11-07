# DeepAMP — learning to score and search for plant-derived antimicrobial peptides

EthnoHack 2026 · Track 2 (Natural products and drug discovery) · CS/CE 351 Artificial Intelligence, Group 34

> **Everything this project outputs is a computational hypothesis. Nothing has been tested in a lab.**
> Predicted "AMP probability" is not evidence of antimicrobial activity, selectivity or safety.

## What it does
1. **Scorer** – given a 10–50 aa peptide, predict P(antimicrobial). A four-rung ladder, all evaluated under the
   same homology-aware split: (1) descriptor logistic regression / random forest (+ a charge-only floor),
   (2) CNN-BiLSTM on raw sequence (after Veltri et al. 2018), (3) frozen ESM-2 (8M) mean-pool + MLP,
   (4) frozen ESM-2 + attention pooling.
2. **Searcher** – simulated annealing starting from real plant AMPs, BLOSUM62-weighted single substitutions
   (≤5 per peptide), fitness = scorer probability − λ·(charge/hydrophobicity penalty).
3. **Independent judge** – an ensemble trained on clusters the scorer never saw, never loaded by the search.
   Candidates are compared with two controls: the unmodified seeds and random mutants with the same number of
   substitutions. Annealing only "works" if it beats both under the judge.

## Data (all real, all public)
| item | source | notes |
|---|---|---|
| positives | UniProtKB/Swiss-Prot, reviewed, 10–50 aa, keyword *Antimicrobial* (KW-0929) or *Antibiotic* (KW-0044) | CC BY 4.0. Curated **annotation**, not a guarantee of experimentally measured activity |
| plant seeds | same query restricted to Viridiplantae (taxonomy 33090) | accession + species kept in the FASTA header |
| negatives (random) | length-matched random fragments of reviewed cytoplasmic proteins, not annotated antimicrobial/secreted | Veltri et al. recipe |
| negatives (hard) | as above, additionally matched on net charge at pH 7 (±1, widened to ±3 if needed) | counts of unmatched positives are logged in `data/processed/DATA_SUMMARY.json` |

Exact queries, UniProt release and retrieval date are written to `data/processed/DATA_SUMMARY.json`.
Further validated sequences (DRAMP, DBAASP – check their licences) can be added with `--extra_pos_fasta`.

## Evaluation protocol
* All sequences (positives and both negative sets) are clustered together at **40 % identity** (CD-HIT if installed,
  otherwise an equivalent built-in greedy clusterer) and **whole clusters** are assigned to: test (15 %),
  scorer pool (42.5 %), judge pool (42.5 %).
* Inside a pool, clusters are split train/validation (85/15) with a different partition per seed → mean ± sd over 5 seeds.
  The epoch is chosen on validation ROC-AUC and the decision threshold on validation MCC; the test set plays no part in selection.
* Metrics: ROC-AUC, PR-AUC, MCC, sensitivity, specificity — reported for **both** negative sets.
* Leakage check: the same models re-run with a naive random split (`--mode random`) to show how much homology inflates scores.
* The scorer used for search is seed 0, fixed in advance (not picked by test score).

## Reproduce
```bash
pip install -r requirements.txt
bash run_all.sh            # data -> clusters -> baselines -> ladder -> judge -> search -> audit -> figures
python tests/smoke_test.py # 1-minute check of the code paths on tiny stand-in data (no network)
```
Results land in `results/summary.md`, `results/candidate_audit.csv`, `figures/`, `outputs/candidate_library_ranked.fasta`.

## Honest limitations
* Positive labels are database annotations; negatives are assumed-inactive protein fragments, not measured inactives.
* The judge shares its label source and (for the ESM judge) the frozen ESM-2 features with the scorer, so it is *independent
  in data, not in every inductive bias*. A second judge on raw sequence (CNN-BiLSTM) is also reported.
* Cysteine-rich plant peptides (defensins, thionins, cyclotides) depend on disulfide bonds/cyclisation that a linear
  sequence model cannot represent; candidates derived from them are the least trustworthy.
* Novelty is measured by edit distance to known AMPs in our data, not by BLAST against all of UniProt.
* Deviation from the written proposal: repeated cluster-grouped hold-out (5 seeds) instead of 5-fold CV.
* Toxicity, haemolysis and selectivity are not modelled.

## Third-party resources
UniProt (CC BY 4.0) · ESM-2 `facebook/esm2_t6_8M_UR50D` (MIT) · PyTorch · Hugging Face Transformers · Biopython ·
scikit-learn · rapidfuzz · CD-HIT (optional). Method inspiration: Veltri et al. 2018; Lin et al. 2023; Porto et al. 2018.

## AI-assistance disclosure
Parts of the code were written with an AI assistant (Claude) and reviewed/run by the team. <!-- edit to state exactly what you did yourselves -->
