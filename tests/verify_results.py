"""Check a finished pipeline run: are these results CORRECT, not just present?

Run this after `python run_all.py`. It does not retrain anything. It reads what the
pipeline wrote and checks the things that would be silently wrong if the project were
broken or if the headline claims were overstated:

  A. integrity  - the split really is homology-aware, the judge really is independent,
                  the test group really was held out, candidates really are novel
  B. data       - the cleaning phase removed the chemistry the proposal named
  C. ordering   - the model ladder behaves the way the README says it does
  D. honesty    - claims that are NOT supported are flagged, so nobody reports them

Exit code 0 means every integrity check passed. Ordering checks are reported as
INFO/WARN rather than failures, because they depend on the UniProt release.

Usage:  python tests/verify_results.py
"""
import glob, json, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

fails, warns = [], []


def ok(cond, msg, hard=True):
    print(f"  {'PASS' if cond else ('FAIL' if hard else 'WARN')}  {msg}")
    if not cond:
        (fails if hard else warns).append(msg)
    return cond


def info(msg):
    print(f"  info  {msg}")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def mcc_of(model, neg, split="cluster"):
    """Mean test MCC for one cell of the ladder, from whichever file holds it."""
    if model in ("charge_only", "logreg", "random_forest"):
        if not os.path.exists("results/baselines.json"):
            return None
        b = jload("results/baselines.json").get(f"{neg}_negatives|{split}_split", {})
        return b.get(model, {}).get("test", {}).get("mcc", [None])[0]
    p = f"results/scorer_{model}_{neg}_{split}.json"
    return jload(p)["summary"]["mcc"][0] if os.path.exists(p) else None


print("=" * 78)
print("A. INTEGRITY  (these must all pass; they are what makes the numbers meaningful)")
print("=" * 78)

ok(os.path.exists("data/processed/dataset.tsv"),
   "data/processed/dataset.tsv exists (run run_all.py first)")
if fails:
    raise SystemExit("\nnothing to verify yet\n")

d = pd.read_csv("data/processed/dataset.tsv", sep="\t")

# A1. the whole point of the clustering: a cluster must live in exactly one group
spans = d.groupby("cluster").group.nunique()
ok((spans == 1).all(),
   f"every 40%-identity cluster sits in exactly one group "
   f"({int((spans > 1).sum())} clusters span groups; must be 0)")

# A2. the three groups must be disjoint in sequence, not just in cluster id
by_group = {g: set(v.seq) for g, v in d.groupby("group")}
for a, b in [("test", "scorer"), ("test", "judge"), ("scorer", "judge")]:
    if a in by_group and b in by_group:
        ok(not (by_group[a] & by_group[b]),
           f"no sequence appears in both '{a}' and '{b}' ({len(by_group[a] & by_group[b])} shared)")

# A3. no duplicate sequences at all
ok(d.seq.duplicated().sum() == 0, f"no duplicate sequences ({d.seq.duplicated().sum()} found)")

# A4. every sequence is a legal peptide
from common import is_canonical
bad = [s for s in d.seq if not is_canonical(s)]
ok(not bad, f"every sequence is 10-50 standard residues ({len(bad)} violations)")

# A5. the judge must have been trained on the judge pool, the scorer on the scorer pool
for f in sorted(glob.glob("results/judge_*.json")):
    ok(jload(f)["args"]["role"] == "judge", f"{os.path.basename(f)} really was trained as a judge")
for f in sorted(glob.glob("results/scorer_*.json")):
    ok(jload(f)["args"]["role"] == "scorer", f"{os.path.basename(f)} really was trained as a scorer")

# A6. the search must not have seeded from anything the judge trained on
if os.path.exists("outputs/candidates.csv"):
    c = pd.read_csv("outputs/candidates.csv")
    grp = d.set_index("id").group.to_dict()
    leaked = sorted({s for s in c.seed_id if grp.get(s) == "judge"})
    ok(not leaked,
       f"no search seed came from a cluster the judge trained on ({len(leaked)} leaked)")
    unknown = sorted({s for s in c.seed_id if s not in grp})
    ok(not unknown, f"every search seed is traceable to the dataset ({len(unknown)} unknown)")

    # A7. candidates must be new sequences, not copies of training peptides
    known = set(d.query("label == 1").seq)
    copies = [s for s in c.seq if s in known]
    ok(not copies, f"no candidate is identical to a known AMP ({len(copies)} copies)")
    ok((c.seq != c.seed_seq).all(), "no candidate is identical to its own seed")
    ok((c.n_mut <= 5).all(), f"every candidate is within the 5-substitution cap "
                             f"(max {int(c.n_mut.max())})")
    ok((c.seq.str.len() == c.seed_seq.str.len()).all(),
       "the search never changed a peptide's length, as designed")

# A8. the test group must not have been used for selection
for f in sorted(glob.glob("results/scorer_*_cluster.json")) + \
         sorted(glob.glob("results/judge_*_cluster.json")):
    j = jload(f)
    if j["args"].get("protocol") == "cv":
        n = len(j.get("per_fold", []))
        ok(n == j["args"]["folds"],
           f"{os.path.basename(f)}: all {j['args']['folds']} CV folds completed ({n})")

# A9. the hypothesis warning must be on every released candidate
for p in glob.glob("outputs/candidate_library_ranked*.fasta"):
    heads = [l for l in open(p, encoding="utf-8") if l.startswith(">")]
    ok(heads and all("COMPUTATIONAL_HYPOTHESIS" in h for h in heads),
       f"{os.path.basename(p)}: every header carries the COMPUTATIONAL_HYPOTHESIS warning")

print()
print("=" * 78)
print("B. DATA CLEANING  (the proposal's stated main risk)")
print("=" * 78)
if os.path.exists("data/processed/DATA_SUMMARY.json"):
    s = jload("data/processed/DATA_SUMMARY.json")
    r = s.get("uniprot_drop_reasons", {})
    info(f"UniProt release {s.get('uniprot_release')} retrieved {s.get('date')}")
    info(f"UniProt: {s.get('n_uniprot_pos_fetched')} fetched -> "
         f"{s.get('n_uniprot_pos_kept')} kept ({s.get('n_uniprot_pos_dropped')} dropped: {r})")
    info(f"positives by source: {s.get('positives_by_source')}")
    ok(s.get("n_pos_kept", 0) > 500, f"enough positives survived cleaning ({s.get('n_pos_kept')})")
    ok(any(k in r for k in ("d_amino_acid", "cyclic_backbone", "thioether_bond")),
       "the cleaning phase actually removed non-linear chemistry (the proposal's main risk)")
    if "dramp_drop_reasons" in s:
        dr = s["dramp_drop_reasons"]
        info(f"DRAMP: {s.get('dramp_rows')} rows -> {s.get('dramp_kept')} kept "
             f"({s.get('dramp_plant')} plant); dropped {dr}")
        ok(any(k in dr for k in ("d_amino_acid", "cyclic_backbone", "branched_backbone",
                                 "chemical_label")),
           "DRAMP's own chemistry annotations were used to drop non-linear entries")
        ok(s.get("n_external_pos_added", 0) > 0,
           f"DRAMP actually added positives ({s.get('n_external_pos_added')} new, "
           f"{s.get('n_external_pos_duplicate_of_uniprot')} were duplicates of UniProt)")
    elif "dramp_error" in s:
        warns.append(f"DRAMP was not merged: {s['dramp_error']}")
        print(f"  WARN  DRAMP was not merged ({s['dramp_error']}); results are Swiss-Prot only")
    if s.get("peptide_len_matching") == "exact":
        info(f"peptide negatives are matched on exact length; "
             f"{s.get('n_pos_unmatched_in_peptide', 0)} positives had no peptide of their "
             f"length and sit out that comparison (they stay in the other two)")
    elif s.get("n_pos_unmatched_in_peptide"):
        warns.append("peptide negatives are not matched on exact length")
        print("  WARN  peptide negatives are not exact-length matched; check that length "
              "alone cannot separate the classes")
    for k in ("n_neg_random", "n_neg_hard"):
        ok(s.get(k, 0) > 0, f"{k} = {s.get(k)}")
    if not s.get("n_neg_peptide"):
        warns.append("the peptide-level negative set is missing")
        print("  WARN  the peptide-level negative set is missing (rerun fetch_real_data.py)")

# --- B2. each comparison is a matched, balanced pairing ---------------------
# A positive with no partner is excluded from THAT comparison via its `sets` entry,
# so what matters is not "did every positive match" but "is each loaded comparison
# balanced, and does length carry no signal in it".
print()
from sklearn.metrics import roc_auc_score  # noqa: E402

from common import load_dataset  # noqa: E402

for neg in ["random", "hard", "peptide"]:
    try:
        sub = load_dataset("data/processed/dataset.tsv", neg)
    except Exception:
        continue
    if not len(sub):
        continue
    npos, nneg = int((sub.label == 1).sum()), int((sub.label == 0).sum())
    ok(npos == nneg,
       f"[{neg}] the comparison is balanced: {npos} positives vs {nneg} negatives")
    lens = sub.seq.str.len().values
    y = (sub.label == 1).astype(int).values
    auc = max(roc_auc_score(y, lens), roc_auc_score(y, -lens))
    ok(auc < 0.55,
       f"[{neg}] peptide LENGTH alone cannot separate the classes (ROC-AUC {auc:.3f}); "
       f"otherwise the models would be scored partly on length")
    excluded = int((d.label == 1).sum()) - npos
    if excluded:
        info(f"[{neg}] {excluded} positives are excluded from this comparison "
             f"(no partner could be matched), by design")
if os.path.exists("data/processed/CLEANING_REPORT.tsv"):
    cr = pd.read_csv("data/processed/CLEANING_REPORT.tsv", sep="\t")
    # NB: cr["flags"], never cr.flags - DataFrame.flags is a pandas property.
    # Count flags only on KEPT rows: a dropped duplicate can carry flags too.
    kept = cr[cr["decision"] == "kept"]
    n_flag = int(kept["flags"].notna().sum())
    info(f"cleaning report: {len(cr)} rows, "
         f"{int((cr['decision'] == 'dropped').sum())} dropped, "
         f"{n_flag} of {len(kept)} kept entries carry a modification flag")
    info("drop reasons: " + ", ".join(f"{k}={v}" for k, v in
                                      cr.loc[cr["decision"] == "dropped", "reason"]
                                      .value_counts().items()))

print()
print("=" * 78)
print("C. THE LADDER  (does it behave the way the README claims?)")
print("=" * 78)
for neg in ["random", "hard", "peptide"]:
    vals = {m: mcc_of(m, neg) for m in
            ["charge_only", "logreg", "random_forest", "cnn_lstm", "esm_mean", "esm_attn"]}
    vals = {k: v for k, v in vals.items() if v is not None}
    if not vals:
        continue
    print(f"\n  {neg} negatives, test MCC:")
    for k, v in vals.items():
        print(f"      {k:<15} {v:.3f}")
    best_neural = max((v for k, v in vals.items() if k.startswith(("esm", "cnn"))), default=None)
    best_desc = max((v for k, v in vals.items() if k in ("logreg", "random_forest")), default=None)
    if best_neural is not None and best_desc is not None:
        ok(best_neural > best_desc,
           f"[{neg}] the best neural rung beats the best descriptor baseline "
           f"({best_neural:.3f} vs {best_desc:.3f}) - the proposal's stated target",
           hard=False)
    if "charge_only" in vals and best_desc is not None:
        ok(vals["charge_only"] < best_desc,
           f"[{neg}] the charge-only floor is below the descriptor models "
           f"({vals['charge_only']:.3f} vs {best_desc:.3f})", hard=False)

ch_r, ch_h = mcc_of("charge_only", "random"), mcc_of("charge_only", "hard")
if ch_r is not None and ch_h is not None:
    ok(ch_h < ch_r - 0.15,
       f"charge matching really did remove the charge shortcut "
       f"(charge-only MCC {ch_r:.3f} on random negatives -> {ch_h:.3f} on hard)", hard=False)

cl, rd = mcc_of("esm_attn", "hard", "cluster"), mcc_of("esm_attn", "hard", "random")
if cl is not None and rd is not None:
    info(f"leakage check: cluster split MCC {cl:.3f} vs cluster-blind random split {rd:.3f} "
         f"({rd - cl:+.3f})")
    if rd - cl > 0.10:
        print("  info  the random split inflates scores noticeably, which is why the "
              "cluster split is the one reported")
    else:
        print("  info  the two splits are close here: the negatives are mostly singleton "
              "clusters, so homology mainly affects the positives. Say this rather than "
              "claiming the cluster split was decisive.")

print()
print("=" * 78)
print("D. THE SEARCH  (is the generated library actually better, under a model that")
print("   never saw the search?)")
print("=" * 78)
verdicts = []
for f in sorted(glob.glob("results/candidate_audit*.csv")):
    c = pd.read_csv(f)
    name = os.path.basename(f).replace("candidate_audit", "judge").replace(".csv", "") or "judge"
    rs = np.random.RandomState(0)
    print(f"\n  {name}: {len(c)} candidates")
    print(f"      plant seeds      {c.seed_judge_p.mean():.3f}")
    print(f"      random mutants   {c.random_mutant_judge_p.mean():.3f}")
    print(f"      SA candidates    {c.judge_p.mean():.3f}")
    for label, other in [("seeds", c.seed_judge_p), ("random mutants", c.random_mutant_judge_p)]:
        diff = (c.judge_p - other).groupby(c.seed_id).mean().values
        boot = [rs.choice(diff, len(diff)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = lo > 0
        print(f"      vs {label:<15} {diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"  {'significant' if sig else 'NOT significant'}")
        verdicts.append((name, label, sig, diff.mean()))
    if "pct_identity_nearest_known" in c:
        print(f"      novelty: median edit distance {c.edit_dist_nearest_known.median():.0f}, "
              f"median identity to nearest known AMP "
              f"{c.pct_identity_nearest_known.median():.1f}%")

if verdicts:
    vs_rand = [v for v in verdicts if v[1] == "random mutants"]
    n_sig = sum(1 for v in vs_rand if v[2])
    print()
    if n_sig == len(vs_rand) and vs_rand:
        info("every judge agrees the annealed candidates beat random mutation. "
             "That claim is safe to report.")
    elif n_sig:
        print(f"  WARN  only {n_sig} of {len(vs_rand)} judges find the candidates better than "
              f"random mutation. Report BOTH judges, and do not claim the search works in "
              f"general.")
        warns.append("the search gain is not reproduced by every judge")
    else:
        print("  WARN  no judge finds the candidates better than random mutation. The search "
              "is not demonstrably doing anything; report that honestly.")
        warns.append("no judge supports the search gain")

print()
print("=" * 78)
if fails:
    print(f"INTEGRITY FAILURES ({len(fails)}):")
    for m in fails:
        print(f"  - {m}")
    print("\nVERIFICATION FAILED: do not report these results until the above are fixed.")
    raise SystemExit(1)
print("ALL INTEGRITY CHECKS PASSED.")
if warns:
    print(f"\n{len(warns)} thing(s) to be careful about when writing this up:")
    for m in warns:
        print(f"  - {m}")
print("\nReminder: every sequence this pipeline produces is a computational prediction.")
print("No candidate here has been synthesised or assayed.")
