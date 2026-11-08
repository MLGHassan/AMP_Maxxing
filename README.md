# DeepAMP: learning to score and search for plant-derived antimicrobial peptides

EthnoHack 2026 · Track: Natural products and drug discovery · CS/CE 351/363 Artificial Intelligence, Fall 2026 · Group 34
Team: Syed Hassan Askari Zaidi (sz10164) · Saqib Rehman (sr10128) · Miqdad Yusuf Hayati (mh10011)

> **All outputs are computational hypotheses. Nothing here has been tested in a laboratory.**
> A predicted probability is not evidence of antimicrobial activity, selectivity or safety.
> Confirming any candidate would need wet-lab synthesis and MIC assays, which is outside this project.

## 1. What this project does

Positives come from UniProtKB/Swiss-Prot **and DRAMP 3.0**, merged and cleaned under one set of chemistry
rules: 6,681 sequences, of which 142 are plant-derived seeds for the search.

1. **Scorer** — predicts P(antimicrobial) for a 10–50 residue peptide. Seven models are compared under one
   identical, homology-aware protocol: charge-only (a deliberate sanity floor), logistic regression and random
   forest on physicochemical descriptors, a CNN-BiLSTM on raw sequence (after Veltri et al. 2018), and frozen
   ESM-2 (8M) embeddings with mean pooling, single-head attention pooling, or multi-head attention pooling.
2. **Searcher** — simulated annealing from real plant AMP seeds, BLOSUM62-weighted single-residue substitutions
   (at most 5 per peptide, length never changes), fitness = scorer probability − λ·(charge/hydrophobicity penalty).
3. **Independent judge** — classifiers trained on protein clusters the scorer never saw and never loaded by the
   search. Candidates are compared against two controls: the unmodified plant seeds, and random BLOSUM mutants
   with the same number of substitutions.

Three negative sets are reported, from easiest to hardest: random length-matched protein fragments,
charge-matched protein fragments, and **real reviewed non-antimicrobial peptides** of matched length.

`PROPOSAL_FIDELITY.md` maps every claim in our written proposal to the code that delivers it, including the
three things that are still open.

## 2. Running it on Windows

Everything below also works on macOS and Linux (see §2.6). The pipeline needs **no GPU**. On the machine we
verified it on (Windows 11, 4 CPU cores, no CUDA) a full run took about **1 hour 40 minutes**: ~6 min to
download and clean the data, **26 min to cluster** 24,287 sequences at 40% identity, and ~65 min for the
ladder, the search and the audits. Clustering is by far the slowest step; installing CD-HIT replaces it with
a few seconds. Budget about **1.5 GB of disk**: the cached ESM-2 embeddings alone are ~390 MB and the
checkpoints ~31 MB (both gitignored and regenerable).

### 2.1 Install Python

Install **Python 3.10 or newer** from [python.org](https://www.python.org/downloads/windows/) and tick
**"Add python.exe to PATH"** during setup. Then open **PowerShell** and check it:

```powershell
python --version
```

> Do not use `python3` on Windows. It is a Microsoft Store stub that opens the Store instead of running anything.

### 2.2 Get the code and install the dependencies

```powershell
git clone <YOUR-REPO-URL> deepamp
```

```powershell
cd deepamp
```

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, allow signed local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

```powershell
pip install -r requirements.txt
```

To reproduce the exact numbers in §4 instead, use `pip install -r requirements-lock.txt`.

### 2.3 Check the install first (about 3 minutes, no internet needed)

```powershell
python tests/test_descriptors.py
```

```powershell
python tests/smoke_test.py
```

The first must end with `DESCRIPTOR TEST PASSED`, the second with `SMOKE TEST PASSED`. The smoke test runs
**every** stage on tiny stand-in data with a stub in place of ESM-2, so its numbers are meaningless — it only
proves the code paths work on your machine.

### 2.4 Run the pipeline

```powershell
.\run_all.ps1
```

That is 21 steps: download and clean the data, cluster it, train the ladder, run the search, audit the
candidates, write the tables and figures, and verify the run. It prints each step with its runtime, and skips
any step whose output already exists, so if something fails you can fix it and run `.\run_all.ps1` again
without losing work. `.\run_all.ps1 -Force` redoes everything.

To see the whole thing work in well under an hour first, use fewer folds, epochs and negative sets:

```powershell
.\run_all.ps1 -Quick
```

A `-Quick` run is for checking that the pipeline works end to end. Do not report its numbers: 2 folds and 20
epochs are not the protocol.

The first ESM-2 run downloads about 30 MB from Hugging Face and caches the embeddings in `cache/`. Two warnings
during that download are expected and harmless: one about `lm_head` / `pooler` weights, and one about symlinks
needing Developer Mode.

### 2.5 Read the results

```powershell
type results\summary.md
```

Open `figures\ladder_mcc.png` and `figures\candidate_audit.png`, and
`outputs\candidate_library_ranked.fasta` for the ranked library.

### 2.6 macOS and Linux

Identical, with `bash run_all.sh` in place of `.\run_all.ps1`:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

```bash
python tests/smoke_test.py && bash run_all.sh
```

Both wrappers just call `run_all.py`, so all three operating systems execute exactly the same steps. Apple
Silicon (MPS) and CUDA are used automatically when present.

### 2.7 Running the steps by hand

`python run_all.py --list` prints every step with its exact command. The main ones:

```bash
python fetch_real_data.py          # UniProt download + cleaning -> data/processed/sequences.tsv
python cluster_and_split.py        # 40% identity clusters -> data/processed/dataset.tsv
python baselines.py                # rung 1: charge-only, logistic regression, random forest
python train.py --role scorer --model esm_attn --neg hard    # rungs 2-4
python train.py --role judge  --model esm_attn --neg hard    # the independent judge
python search.py --scorer checkpoints/scorer_esm_attn_hard_s0.pt --n_seeds 40
python evaluate_candidates.py --judge_glob "checkpoints/judge_esm_attn_hard_s*.pt"
python summarize.py
python tests/verify_results.py
```

Optional: install [CD-HIT](https://github.com/weizhongli/cdhit) and it is used for clustering automatically.
It is awkward to build on Windows, so the built-in greedy clusterer (same identity definition) is the default
there and finishes in about two minutes. Optional: install BLAST+ and `evaluate_candidates.py` will report real
BLASTP identities instead of the equivalent local-alignment identity.

## 3. How do I know the results are correct?

Three layers, in increasing strength.

### 3.1 The code does what it says — `python tests/smoke_test.py`

Runs all nine stages on synthetic data and asserts the outputs exist and have the right columns. It also tests
the cleaning rules directly: that a D-amino-acid entry is dropped, a cyclopeptide is dropped, a non-standard
residue is dropped, and a disulfide-bonded entry is *kept and flagged*. Ends in `SMOKE TEST PASSED`.

### 3.2 The descriptors are right — `python tests/test_descriptors.py`

Checks net charge, hydrophobicity and hydrophobic moment against an independent re-implementation of the
published formulae, against hand-computed values (K₁₀ at pH 7 = 10×0.9998 + 0.9755 − 0.9996 = +9.974), and
against modlAMP itself if it is installed. This is what justifies not using modlAMP, which no longer builds on
current Python.

### 3.3 The run is sound — `python tests/verify_results.py`

This is the one that matters. It reads a finished run and checks the claims that would be **silently wrong** if
the project were broken:

| check | why it matters |
|---|---|
| every 40%-identity cluster sits in exactly one group | if a cluster straddled train and test, near-duplicate peptides would be on both sides and every score would be inflated |
| no sequence appears in two groups; no duplicates anywhere | same reason, at the sequence level |
| no search seed came from a cluster the judge trained on | otherwise the "independent" judge is not independent and the audit proves nothing |
| no candidate is identical to a known AMP, or to its own seed | otherwise the search is returning its input |
| every candidate is within the 5-substitution cap and unchanged in length | the search is doing what it is documented to do |
| all CV folds completed | a partial run would report a mean over fewer folds than it claims |
| every released candidate header carries `COMPUTATIONAL_HYPOTHESIS` | nothing leaves this repo looking like a validated hit |
| the cleaning phase actually removed non-linear chemistry | the proposal's stated main risk |

It then prints the ladder and the search audit and tells you **which claims your own numbers support**. It
exits non-zero only on integrity failures; ordering results are `WARN`/`info`, because they legitimately shift
with the UniProt release.

### 3.4 What "correct" looks like

Integrity checks: all `PASS`. Then compare against §4 below — you should see the same **ordering and rough
magnitudes**, not identical digits. Digits move with the UniProt release, your library versions, and CPU
floating-point order. Compare `data/processed/DATA_SUMMARY.json` first: if your UniProt release differs from
ours, expect every number to differ slightly.

Specifically, a healthy run shows:

- `positives_by_source` in `DATA_SUMMARY.json` reading roughly `{"dramp3": 5684, "uniprot_pos": 997}`. **Check
  this first.** If DRAMP failed to download you have 997 positives, not 6,681, and nothing below applies
- **every neural rung beats the best descriptor baseline** on MCC, on all three negative sets — the proposal's
  stated target, and what `verify_results.py` checks
- charge-only collapses from ~0.53 MCC on random fragments to ~0.11 (ROC-AUC ~0.62) on the charge-matched set.
  If charge alone still scored well there, the matching is broken
- all three comparisons are balanced, and **length alone gives ROC-AUC 0.500** in each. A value much above 0.55
  means the negatives are not length-matched and the models are being scored partly on length
- both judges find the candidates better than random mutants, with 95% intervals excluding zero

Things you should **not** expect, because they are not true in our run:

- the cluster-grouped split does **not** beat a naive random split here (−0.001 MCC). That is a property of
  this dataset, not a bug: the negatives are nearly all singleton clusters
- the candidates are **not** novel — ~79% median identity to a known AMP

A large leakage gap, or candidates with low identity to known AMPs, means something changed in the data rather
than in the models — check `DATA_SUMMARY.json` and `CLEANING_REPORT.tsv` first.

## 4. Results from our run

<!-- RESULTS_START -->
UniProt release 2026_04 + DRAMP 3.0, **6,681 positives** after cleaning, Windows 11 / Python 3.14 / CPU only.
Full tables, including the CV columns, are in `results/summary.md`.

**MCC on the held-out cluster test group (mean +/- sd over the 5 CV folds)**

| model | random fragments | charge-matched fragments | **real non-AMP peptides** |
|---|---|---|---|
| charge only (floor) | 0.532 +/- 0.015 | **0.112 +/- 0.075** | 0.441 +/- 0.003 |
| logistic regression | 0.588 +/- 0.023 | 0.436 +/- 0.041 | 0.362 +/- 0.037 |
| random forest | 0.659 +/- 0.009 | 0.559 +/- 0.028 | 0.493 +/- 0.013 |
| CNN-BiLSTM (Veltri) | 0.718 +/- 0.013 | 0.705 +/- 0.015 | 0.659 +/- 0.018 |
| ESM-2 + mean pool | **0.763 +/- 0.022** | **0.726 +/- 0.005** | **0.666 +/- 0.027** |
| ESM-2 + 1-head attention | - | 0.721 +/- 0.009 | - |
| ESM-2 + multi-head attention | 0.740 +/- 0.027 | 0.723 +/- 0.007 | 0.664 +/- 0.023 |

1. **The charge-only floor behaves exactly as the proposal intended.** MCC 0.532 against random fragments
   collapses to **0.112** (ROC-AUC 0.622) once the negatives are charge-matched. The hard column is measuring
   something other than "is this peptide cationic".
2. **The ladder is monotonic on all three negative sets**, and every neural rung beats every descriptor
   baseline. Frozen ESM-2 with mean pooling is the best model everywhere.
3. **Rung 4 does not beat rung 3, but it is no longer behind it.** Multi-head attention (0.723) sits level with
   mean pooling (0.726) and single-head (0.721) on charge-matched negatives. Extra pooling parameters buy
   nothing here; they no longer cost anything either.
4. **The hardest negative set now separates the models clearly.** Against real reviewed non-AMP peptides, the
   best neural model (0.666) beats random forest (0.493) by 0.17 MCC. Note that logistic regression (0.362)
   scores *below* the single-feature charge-only floor (0.441) on this set: a linear model over six descriptors
   is the wrong tool once the negatives are real peptides.

**Leakage check** (ESM-2 multi-head, charge-matched negatives): cluster-grouped split MCC 0.723 vs
cluster-blind random split 0.722, a difference of **-0.001**. On this dataset the homology-aware split makes
essentially no difference, because the negatives are nearly all singleton clusters and clustering therefore
only constrains the positives. We report the cluster split because it is the defensible choice, not because it
rescued us from inflated scores.

**Search audit** - mean independent-judge P(AMP) over 120 candidates from 40 plant seeds, 95% bootstrap
intervals over seeds:

| judge | plant seeds | random mutants | candidates | cand - seeds | cand - random mutants |
|---|---|---|---|---|---|
| ESM multi-head ensemble (5 heads) | 0.638 | 0.614 | 0.926 | **+0.288 [+0.192, +0.392]** | **+0.312 [+0.218, +0.411]** |
| CNN-BiLSTM ensemble (5 heads) | 0.638 | 0.611 | 0.879 | **+0.242 [+0.148, +0.343]** | **+0.268 [+0.180, +0.363]** |

Both judges agree, including the CNN-BiLSTM one, which shares no features with the ESM scorer that drove the
search.

**The candidates are still not very novel.** Median edit distance to the nearest known AMP is 5 - exactly the
mutation cap - and median local-alignment identity is **79.2%**. The search polishes known plant peptides; it
does not find new families.

### What changed when DRAMP was added

The earlier Swiss-Prot-only run had 997 positives. Two conclusions from it did **not** survive the larger
dataset, and are retracted here rather than quietly dropped:

| claim from the 997-positive run | at 6,681 positives |
|---|---|
| "the ladder is not monotonic; CNN-BiLSTM beats ESM-2 on two of three negative sets" | false - ESM-2 mean pool is best on all three |
| "multi-head attention is *worse* than mean pooling (0.695 vs 0.727)" | false - they are level (0.723 vs 0.726) |
| "on real peptides the neural gain over random forest is only 0.06 MCC" | the gain is 0.17 MCC |

Both were small-data artifacts. The lesson for the report is that at ~1,000 labelled peptides the ranking
between these architectures was mostly noise.
<!-- RESULTS_END -->

## 5. Evaluation protocol

- All sequences (positives and all three negative sets) are clustered together at 40% identity. Whole clusters
  are assigned to `test` (15%), `scorer` (42.5%) and `judge` (42.5%), balancing both total rows and positives,
  so near-duplicates never straddle groups.
- **Nested 5-fold cross-validation over the clusters of a pool.** For each of the 5 outer folds, the 4 training
  folds are split again by cluster into inner-train and inner-validation; the epoch is chosen on
  inner-validation ROC-AUC and the decision threshold on inner-validation MCC. The outer fold is then predicted
  out-of-fold, and the fixed `test` group is scored once.
  - **CV** columns pool the out-of-fold predictions, so the threshold never saw the data it is applied to. This
    is the cross-validated estimate the proposal asked for.
  - **test** columns are the fixed held-out cluster group, mean ± sd over the 5 folds. Because every fold scores
    the *same* test group, that sd reflects only re-partitioning and initialisation — it understates total
    uncertainty. Quote the CV figure when you want one number.
- Metrics: ROC-AUC, PR-AUC, MCC, sensitivity, specificity.
- **Leakage check**: the same model under a size-matched split that ignores clusters entirely.
- `--protocol holdout` reproduces the earlier repeated cluster-grouped hold-out, for comparison.

## 6. Data

| item | source | notes |
|---|---|---|
| positives (Swiss-Prot) | UniProtKB/Swiss-Prot, reviewed, 10–50 aa, keyword Antimicrobial (KW-0929) or Antibiotic (KW-0044) | CC BY 4.0. **Curated annotation, not measured activity.** |
| positives (DRAMP 3.0) | DRAMP's `general_amps.txt`, entries with antimicrobial/antibacterial activity | downloaded at run time into `data/raw/` (gitignored). Free for academic use; **not redistributed here**. Adds 5,684 sequences beyond Swiss-Prot. |
| plant seeds | Viridiplantae (taxonomy 33090) Swiss-Prot entries, plus DRAMP entries whose `Swiss_Prot_Entry` resolves to a reviewed plant accession | 142 seeds. DRAMP's own `Source` column is free text, so it is not used for this. |
| random negatives | length-matched fragments of reviewed cytoplasmic proteins not annotated antimicrobial or secreted | Veltri et al. 2018 recipe |
| hard negatives | as above, also matched on net charge at pH 7 (±1, widened to ±3 if needed) | unmatched positives counted in `DATA_SUMMARY.json` |
| peptide negatives | reviewed peptides 10–50 aa **not** annotated antimicrobial, matched on **exact** length | real peptides, so a model cannot win by spotting "mature peptide vs internal fragment" |

Each positive records, in the `sets` column, which comparisons it actually has a partner in. A positive with no
charge-matched fragment, or no non-AMP peptide of exactly its length, is left out of **that** comparison rather
than facing a missing counterpart. This matters: reviewed non-AMP peptides are much longer than AMPs (median 31
vs 20 residues), and an earlier version that allowed a ±10 length tolerance let **length alone** reach ROC-AUC
0.665 on the peptide set. Matching exactly costs 2,410 positives from that one comparison and brings length-alone
ROC-AUC to 0.500 — verified for all three sets by `tests/verify_results.py`.

### Cleaning

The proposal called data cleaning the project's main risk, because the peptide databases mix cyclic peptides,
D-amino acids and chemically modified entries. `fetch_real_data.py` implements that phase and logs every
decision to `data/processed/CLEANING_REPORT.tsv`:

- **dropped**, because the sequence string misrepresents the molecule — D-amino acids (KW-0208), thioether /
  sactipeptide cross-links (KW-0883), head-to-tail cyclopeptides, non-standard residues, out-of-range lengths,
  duplicates
- **kept and flagged**, because the backbone stays linear — disulfide bonds, C-terminal amidation, lipidation,
  lantibiotic annotation

On UniProt release 2026_04 this drops 136 of 1,133 fetched positives (57 duplicates, 49 non-standard residues,
15 thioether cross-links, 8 D-amino-acid entries, 7 cyclopeptides), leaving 997. Of those, 651 carry a
modification flag (355 disulfide, 313 amidation, 1 lantibiotic). Dropping the disulfide entries too would
remove most plant defensins and much of the plant seed set, so they are kept and the limitation is stated
instead (see §7).

DRAMP is merged by default. To control it:

```bash
python fetch_real_data.py --dramp none          # Swiss-Prot only (the older, smaller dataset)
```

```bash
python fetch_real_data.py --dramp_strict        # also drop DRAMP entries with uncurated chemistry
```

```bash
python fetch_real_data.py --extra_pos_fasta your_DBAASP_or_APD3_export.fasta
```

If the DRAMP server is unreachable the pipeline prints a warning, records `dramp_error` in
`DATA_SUMMARY.json`, and continues with Swiss-Prot only — it does not fail. Check each database's licence before
redistributing anything (DBAASP v3 is CC BY-NC-SA, so it cannot be committed to this repository).

## 7. Limitations — read this before quoting any number

- **Labels are database annotations.** Positives are peptides a curator annotated as antimicrobial; negatives
  are *assumed* inactive, not measured inactive.
- **Two of the three negative sets are protein fragments.** A model can score well on them by recognising
  "mature secreted peptide" rather than "antimicrobial". The peptide-level negative set exists to close that
  gap and is the number to trust.
- **Cysteine-rich plant peptides** (defensins, thionins, cyclotides) depend on disulfide topology or
  cyclisation that a linear sequence model cannot represent at all, and they are a large share of the
  positives.
- **The ESM judge is not fully independent of the ESM scorer.** They are trained on disjoint clusters, but they
  share the same frozen ESM-2 features, so they share inductive bias. That is exactly why a second judge with a
  different architecture (CNN-BiLSTM) is also reported.
- **The candidates are near-relatives of their seeds**, at 83% median identity to a known AMP and exactly the
  5-substitution cap in edit distance. The search optimises; it does not explore.
- **The search gain is measured by models, not by experiments.** Two judges agreeing says the annealing found
  sequences that classifiers like. It says nothing about activity.
- **Novelty is measured against our own positives**, not against all of UniProt.
- **Toxicity, haemolysis and selectivity are not modelled at all.**
- **39% of the DRAMP positives (2,223 of 5,684) have uncurated chemistry.** DRAMP records "Not included yet" for their
  stereochemistry or backbone shape, so we cannot confirm they are linear L-peptides. They are kept and flagged
  `chemistry_unverified`; `--dramp_strict` drops them, and running both is a sensitivity analysis worth
  reporting.
- **Still open against the proposal:** DBAASP v3 and APD3 are not fetched automatically (no bulk endpoint /
  no verifiable TLS certificate from here), and the AMP Scanner v2 reference digits in
  `external_reference.json` have not been checked against the paper. Both are tracked in
  `PROPOSAL_FIDELITY.md`.

## 8. Repository layout

```
fetch_real_data.py        UniProt download, cleaning phase, three negative sets
external_db.py            DRAMP 3.0 download + its own chemistry cleaning; generic FASTA
cluster_and_split.py      40% identity clustering, cluster-level group assignment
baselines.py              rung 1: charge-only, logistic regression, random forest
train.py                  rungs 2-4 (scorer or judge), nested 5-fold cluster CV
models.py, esm_embed.py   model heads, frozen ESM-2 embedder with on-disk cache
search.py                 simulated annealing
evaluate_candidates.py    judge audit with controls, bootstrap intervals, BLAST identity
summarize.py              results/summary.md + figures
common.py                 descriptors, metrics, clustering identity, CV splitters
external_reference.json   published AMP Scanner v2 figures (needs verifying)
run_all.py                the pipeline; run_all.ps1 / run_all.sh are thin wrappers
tests/smoke_test.py       end-to-end check on stand-in data
tests/test_descriptors.py descriptor checks, incl. against modlAMP when installed
tests/verify_results.py   integrity + claim checks on a finished run
PROPOSAL_FIDELITY.md      every proposal claim mapped to the code
```

## 9. Troubleshooting

| symptom | fix |
|---|---|
| `python3` opens the Microsoft Store | use `python`, not `python3`, on Windows |
| `.\run_all.ps1` is "not digitally signed" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or run `python run_all.py` directly |
| `Activate.ps1` cannot be loaded | same as above |
| UniProt download fails or times out | rerun; each request already retries 4 times. Check your connection, then `python run_all.py --from 1` |
| `WARNING: DRAMP unavailable` | the run continues with Swiss-Prot only and records `dramp_error` in `DATA_SUMMARY.json`. Rerun later, or download `general_amps.txt` from [DRAMP](http://dramp.cpu-bioinfor.org/downloads/) by hand and pass `--dramp path/to/general_amps.txt` |
| results differ a lot from §4 | check `positives_by_source` in `DATA_SUMMARY.json` — a run without DRAMP has 997 positives instead of 6,681 and is not comparable |
| `no judge checkpoints match ...` | run the judge steps first: `python run_all.py --from 1` will skip what already exists |
| `Scorer checkpoint not found` | the scorer training step did not finish; rerun it |
| symlink / Developer Mode warning from Hugging Face | harmless, ignore |
| `lm_head` / `pooler` weight warnings | harmless: we use ESM-2 as a frozen encoder, so its language-model head is unused |
| clustering seems slow | expected: ~26 min for 24,287 sequences, and it is single-threaded. Install CD-HIT to cut it to seconds, or use `--dramp none` for the smaller Swiss-Prot-only dataset |
| MPS errors on a Mac | `PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py ...` |
| different numbers from §4 | expected. Compare `data/processed/DATA_SUMMARY.json`; a different UniProt release changes everything slightly |
| out of memory | `python train.py ... --bs 32` |

## 10. Licences, credits and AI disclosure

- Code: MIT (see `LICENSE`). Model: ESM-2 `facebook/esm2_t6_8M_UR50D` (MIT).
- Data: UniProtKB/Swiss-Prot, CC BY 4.0 — please cite UniProt. DRAMP 3.0 (Shi et al., *Nucleic Acids Research*
  2022), free for academic use — downloaded at run time, **not redistributed in this repository**; please cite
  DRAMP and check their terms before any other use.
- Libraries: PyTorch, Hugging Face Transformers, scikit-learn, Biopython, pandas, matplotlib, rapidfuzz;
  CD-HIT and BLAST+ optional.
- Method inspiration: Veltri, Kamath and Shehu 2018 (*Bioinformatics*); Lin et al. 2023 (*Science*, ESM-2);
  Porto et al. 2018 (*Nature Communications*).
- AI assistance: parts of the code were written with an AI assistant (Claude), then run, tested and reviewed by
  the team. <!-- edit to state exactly what your team did yourselves -->
