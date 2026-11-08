# Proposal fidelity audit

Every claim in the DeepAMP proposal (CS/CE 351/363, Group 34), mapped to where the code
delivers it. `MET` means the code does what the proposal says. `CHANGED` means we did
something different on purpose and say why. `OPEN` means it is still not done.

Audited against the submitted proposal PDF on 2026-11-08.

## 3. Input and output

| proposal says | status | where |
|---|---|---|
| peptides 10-50 residues, 20-letter alphabet, from FASTA | MET | `common.is_canonical`, `common.read_fasta` |
| physicochemical descriptors: net charge at pH 7, hydrophobicity, hydrophobic moment, pI, MW | MET | `common.descriptors` (also adds length) |
| computed "with modlAMP" | CHANGED | modlAMP is unmaintained and does not build on Python 3.12+. `common.py` computes the same five quantities directly (EMBOSS pKa values for charge, Eisenberg consensus scale for hydrophobicity and moment, Biopython for pI and MW). `tests/test_descriptors.py` checks them against an independent re-implementation of the published formulae, against hand-computed invariants, and against modlAMP itself when it is installed. |
| binary label, 1 = experimentally confirmed antimicrobial | CHANGED | Labels are Swiss-Prot *curated annotations* (KW-0929 / KW-0044), not measured activity. Stated in the README and `fetch_real_data.py`. DRAMP/DBAASP can be merged in with `--extra_pos_fasta`; see "Still open" below. |
| probability in [0,1] per sequence | MET | `models.Scorer.predict` |
| ranked FASTA of generated peptides with probability, net charge, hydrophobic moment, distance to nearest training peptide | MET | `outputs/candidate_library_ranked*.fasta` |

## 5. Proposed solution

### (a) The scorer ladder

| rung | proposal says | status | where |
|---|---|---|---|
| 1 | logistic regression and random forest on descriptors | MET | `baselines.py` |
| 1b | "deliberately crude net-charge-only classifier as a sanity floor" (§7) | MET | `baselines.py`, model `charge_only` |
| 2 | embedding -> 1D convolution -> BiLSTM, reproducing Veltri et al. | MET | `models.CNNBiLSTM` |
| 3 | frozen ESM-2 `esm2_t6_8M_UR50D`, mean-pooled, small feed-forward head | MET | `esm_embed.py` + `models.MeanHead` |
| 4 | the same embeddings with **multi-head** attention pooling | MET (was wrong) | `models.MultiAttnHead`. The earlier code used single-head additive attention and called it `esm_attn`. It is now 4-head attention pooling; the single-head version is kept as `esm_attn1h` so the ablation is reported rather than lost. |
| - | ESM-2 frozen, not fine-tuned, embeddings cached | MET | `esm_embed.EsmEmbedder`, `requires_grad = False`, on-disk cache |
| - | CD-HIT clustering at 40% identity, split by cluster not by sequence | MET | `cluster_and_split.py`. Uses CD-HIT when it is on PATH; otherwise a built-in greedy clusterer with the identical identity definition (CD-HIT is not easily installable on Windows). |

### (b) The searcher

| proposal says | status | where |
|---|---|---|
| simulated annealing with the best scorer as fitness | MET | `search.anneal` |
| `Fitness(s) = P(AMP|s) - lambda * Penalty(s)` | MET | `search.py`, `lam=0.6` |
| penalty discourages extreme hydrophobicity and very low net charge | MET | `search.penalty` |
| seeds are plant AMPs of medicinally used species | MET | `data/processed/plant_seeds.fasta`, Viridiplantae (taxonomy 33090) |
| single-residue substitutions weighted by BLOSUM62 | MET | `search.mutate` |
| geometric cooling, `T_{k+1} = 0.95 T_k` | MET | `search.anneal`, `cool=0.95` |

### (c) The independent judge

| proposal says | status | where |
|---|---|---|
| second classifier trained on a disjoint set of clusters | MET | `cluster_and_split.assign_groups` puts whole clusters in `scorer` / `judge` / `test`; `train.py --role judge` |
| kept entirely out of the search loop | MET | `search.py` never loads a judge checkpoint; the audit is a separate script |
| used only to score final candidates | MET | `evaluate_candidates.py` |

## 7. Evaluation plan

| proposal says | status | where |
|---|---|---|
| ROC-AUC, PR-AUC, MCC, sensitivity, specificity | MET (was partly missing) | all five computed in `common.metrics_at`; sensitivity and specificity were computed but never reached `results/summary.md`, and now do |
| **5-fold cross-validation over CD-HIT clusters** | MET (was missing) | `common.cv_folds` + `train.py --protocol cv` (the default). Previously the code ran a repeated cluster-grouped hold-out over 5 seeds, which is not cross-validation. The CV is nested: the decision threshold is chosen on an inner split of the training folds, so it never sees the fold it is scored on. |
| plus a held-out test set touched once | MET | the `test` cluster group; scored once per fold, never used for epoch, threshold or model choice |
| results reported twice: random length-matched fragments, and hard negatives matched on length and net charge | MET | `--neg random` and `--neg hard` |
| novelty by Levenshtein distance **and BLAST identity** | MET (BLAST was missing) | edit distance in `common.levenshtein_nearest`; identity in `common.identity_nearest` (local-alignment % identity, the quantity BLAST reports), and `evaluate_candidates.blastp_identity` runs real BLASTP when BLAST+ is on PATH |
| properties checked against ranges reported for known active peptides | MET | `evaluate_candidates.py` reports the fraction inside the 5-95% charge and hydrophobic-moment range of the known AMPs |
| external reference: published accuracy of AMP Scanner v2 | PARTLY (was missing) | `external_reference.json` + a section in `results/summary.md`. The digits are pre-filled from Veltri et al. 2018 but **the team must confirm them against the paper and set `verified: true`**. |
| beat the descriptor baseline by a clear margin on MCC on hard negatives | MET | see `results/summary.md` |
| limitations stated in report, README and Devpost page | MET in repo | README section 1 and 7; the Devpost page is outside this repository |

## 8. Resources

| proposal says | status | notes |
|---|---|---|
| UniProtKB/Swiss-Prot for negatives | MET | `fetch_real_data.py` |
| PyTorch, Transformers (ESM-2), BioPython, scikit-learn, NumPy, Pandas, Matplotlib | MET | `requirements.txt` |
| CD-HIT | MET (optional) | used when present |
| modlAMP | CHANGED | see §3 above |
| **DRAMP 3.0 for verified AMPs** | MET | `external_db.py` downloads DRAMP 3.0's `general_amps.txt` at run time into `data/raw/` (gitignored: DRAMP is free for academic use but is not ours to redistribute) and `fetch_real_data.py --dramp auto` is the default. DRAMP's own annotation columns drive the same cleaning rules we apply to Swiss-Prot. Result: **6,681 positives, up from 997** — the "few thousand" the proposal assumed. |
| **DBAASP v3 for verified AMPs** | PARTLY | DBAASP has no bulk download and is CC BY-NC-SA, so it is not fetched automatically. An export can be merged with `--extra_pos_fasta <file>`; a plain FASTA carries no machine-readable chemistry, so those entries are flagged `chemistry_unverified` and only the alphabet/length rules apply to them. |
| **APD3 for plant-origin seeds** | PARTLY | APD3's site is not reachable over a verified TLS connection from here, so it is not wired up; `--extra_pos_fasta` accepts an export. Plant seeds now come from the Viridiplantae Swiss-Prot subset **plus** DRAMP entries resolved to a reviewed plant accession through their `Swiss_Prot_Entry` cross-reference (DRAMP's own `Source` column is free text and unusable for this). That took the seed set from 112 to **142**. |
| free Colab / Kaggle T4 | MET | and it runs on a 4-core laptop CPU in about an hour; no GPU needed |

## 9. Timeline and main risk

> "Main risk is data cleaning, not modelling. The peptide databases mix cyclic peptides,
> D-amino acids and chemically modified entries, and resolving that is where we expect
> most of the effort to go, which is why it has a dedicated phase before the hackathon."

This was the largest gap in the submitted code: there was **no cleaning phase at all**
beyond a length and alphabet filter, so the entries the proposal specifically named were
being fed to the models as if they were ordinary linear peptides.

It now exists (`fetch_real_data.classify`, reported in `data/processed/CLEANING_REPORT.tsv`):

- **dropped**, because the sequence string misrepresents the molecule: D-amino acids
  (KW-0208), thioether / sactipeptide cross-links (KW-0883), head-to-tail cyclopeptides
  (CROSSLNK or PTM text), non-standard residues, out-of-range lengths, duplicates
- **kept and flagged**, because the backbone stays linear: disulfide bonds, C-terminal
  amidation, lipidation, lantibiotic annotation

On UniProt release 2026_04 this removes 30 chemically misrepresented entries that the
earlier dataset contained (8 D-amino acid, 15 thioether, 7 cyclic), on top of 49
non-standard-residue and 57 duplicate entries the old code already excluded.

Note that 651 of the 997 surviving positives carry at least one modification flag: 355
disulfide, 313 C-terminal amidation, 1 lantibiotic. Dropping the disulfide entries as well
would remove most plant defensins and much of the plant seed set, so they are kept and the
limitation is stated instead: a linear sequence model cannot represent the disulfide
topology those peptides depend on.

## Still open

1. **DBAASP v3 and APD3 are not fetched automatically.** DBAASP has no bulk download
   endpoint and is CC BY-NC-SA; APD3 did not present a verifiable TLS certificate from
   this machine. Both can be merged from a manual export with `--extra_pos_fasta`, and
   both would arrive without machine-readable chemistry, so they would be flagged
   `chemistry_unverified`. With DRAMP merged the positive set is already 6,681, so this
   is now a nice-to-have rather than the blocking gap it was.
2. **The external reference digits are unverified.** Five minutes with the Veltri et al.
   paper closes this.
3. **Negatives are still not ideal.** Two of the three negative sets are internal
   fragments of cytoplasmic proteins, so a model can score well by recognising "mature
   secreted peptide" rather than "antimicrobial". The third set (`--neg peptide`, real
   reviewed non-AMP peptides) was added to address exactly this and is the honest
   headline number.
4. **DRAMP's chemistry is partly uncurated.** 2,223 of the 5,684 DRAMP positives (39%)
   carry `chemistry_unverified` because DRAMP records "Not included yet" for their
   stereochemistry or backbone shape. They are kept by default and dropped by
   `--dramp_strict`, which is a sensitivity analysis worth running for the report.
5. **Deviation from the proposal, kept deliberately:** the proposal promised 5-fold CV
   *instead of* repeated hold-out. Both are available (`--protocol cv`, the default, and
   `--protocol holdout`); CV is what the reported tables use.
