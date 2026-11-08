"""Collect results/*.json into results/summary.md and figures/*.png."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ENC

#: config file that ships with the code, so it is found whatever the working directory is
EXT_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_reference.json")

METRICS = ["roc_auc", "pr_auc", "mcc", "sensitivity", "specificity"]
ORDER = ["charge_only", "logreg", "random_forest", "cnn_lstm", "esm_mean", "esm_attn1h", "esm_attn"]
NICE = {"charge_only": "charge only (floor)", "logreg": "logistic regression",
        "random_forest": "random forest", "cnn_lstm": "CNN-BiLSTM (Veltri)",
        "esm_mean": "ESM-2 + mean pool", "esm_attn1h": "ESM-2 + 1-head attention",
        "esm_attn": "ESM-2 + multi-head attention"}
NEG_LABEL = {"random": "random fragment negatives",
             "hard": "charge-matched fragment negatives",
             "peptide": "real non-AMP peptide negatives"}


def load_rows():
    """-> DataFrame with one row per (model, negatives, split) and both estimates."""
    rows = []
    if os.path.exists("results/baselines.json"):
        with open("results/baselines.json", encoding=ENC) as fh:
            for k, d in json.load(fh).items():
                neg, split = k.split("|")
                neg, split = neg.split("_")[0], split.split("_")[0]
                for m, v in d.items():
                    rows.append(dict(model=m, negatives=neg, split=split,
                                     **{n: tuple(v["test"][n]) for n in METRICS},
                                     **{f"cv_{n}": v["cv_pooled"].get(n) for n in METRICS}))
    for f in sorted(glob.glob("results/scorer_*.json")):
        if f.endswith("_holdout.json"):        # the alternative protocol, reported separately
            continue
        with open(f, encoding=ENC) as fh:
            j = json.load(fh)
        a = j["args"]
        if a["role"] != "scorer":
            continue
        cv = j.get("cv_pooled", {})
        rows.append(dict(model=a["model"], negatives=a["neg"], split=a["mode"],
                         **{n: tuple(j["summary"][n]) for n in METRICS},
                         **{f"cv_{n}": cv.get(n) for n in METRICS}))
    return pd.DataFrame(rows)


def fmt(t):
    return "n/a" if t is None or (isinstance(t, tuple) and t[0] is None) else f"{t[0]:.3f} +/- {t[1]:.3f}"


def fmt1(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
R = load_rows()
if R.empty:
    raise SystemExit("no results found - run baselines.py and train.py first")

negs = [n for n in ["random", "hard", "peptide"] if (R.negatives == n).any()]
out = []
out.append("# DeepAMP results\n")
out.append("All numbers are computational predictions. Nothing here was tested in a laboratory.\n")
out.append("Protocol: nested 5-fold cross-validation over 40%-identity clusters of the scorer pool.\n"
           "`test` columns are the fixed held-out cluster group, scored once per fold (mean +/- sd over\n"
           "the 5 folds, so they share one test set and understate total uncertainty). `CV` columns are\n"
           "the pooled out-of-fold predictions, where the decision threshold never saw the fold it is\n"
           "applied to.\n")

for neg in negs:
    out.append(f"\n## {NEG_LABEL[neg]}, held-out cluster test group\n")
    out.append("| model | ROC-AUC | PR-AUC | MCC | sensitivity | specificity | CV MCC | CV ROC-AUC |\n"
               "|---|---|---|---|---|---|---|---|\n")
    for m in ORDER:
        r = R[(R.model == m) & (R.negatives == neg) & (R.split == "cluster")]
        if not len(r):
            continue
        r = r.iloc[0]
        out.append(f"| {NICE.get(m, m)} | {fmt(r.roc_auc)} | {fmt(r.pr_auc)} | {fmt(r.mcc)} | "
                   f"{fmt(r.sensitivity)} | {fmt(r.specificity)} | {fmt1(r.cv_mcc)} | "
                   f"{fmt1(r.cv_roc_auc)} |\n")

out.append("\n## Leakage check: cluster-grouped split vs a naive random split (MCC on the test group)\n\n")
out.append("| model | negatives | cluster split | random split | inflation |\n|---|---|---|---|---|\n")
for (m, neg), g in R.groupby(["model", "negatives"]):
    if set(g.split) >= {"cluster", "random"}:
        c = g[g.split == "cluster"].iloc[0].mcc
        r = g[g.split == "random"].iloc[0].mcc
        out.append(f"| {NICE.get(m, m)} | {neg} | {fmt(c)} | {fmt(r)} | {r[0] - c[0]:+.3f} |\n")

# ---- protocol comparison, if anyone ran the alternative ------------------
hold = sorted(glob.glob("results/scorer_*_holdout.json"))
if hold:
    out.append("\n## Protocol check: 5-fold cluster CV vs repeated cluster-grouped hold-out\n\n")
    out.append("The proposal specified cross-validation; the hold-out is kept so the difference\n"
               "is visible rather than assumed.\n\n")
    out.append("| model | negatives | 5-fold CV (test MCC) | repeated hold-out (test MCC) |\n"
               "|---|---|---|---|\n")
    for f in hold:
        with open(f, encoding=ENC) as fh:
            j = json.load(fh)
        a = j["args"]
        cvr = R[(R.model == a["model"]) & (R.negatives == a["neg"]) & (R.split == a["mode"])]
        cvs = fmt(tuple(cvr.iloc[0].mcc)) if len(cvr) else "n/a"
        out.append(f"| {NICE.get(a['model'], a['model'])} | {a['neg']} | {cvs} | "
                   f"{fmt(tuple(j['summary']['mcc']))} |\n")

# ---- external reference (the proposal's stand-in for a leaderboard) --------
if os.path.exists(EXT_REF):
    with open(EXT_REF, encoding=ENC) as fh:
        ext = json.load(fh)
    best = None
    for m in reversed(ORDER):
        r = R[(R.model == m) & (R.negatives == "hard") & (R.split == "cluster")]
        if len(r):
            best = (m, r.iloc[0]); break
    out.append(f"\n## External reference: {ext['name']}\n\n")
    out.append("EthnoHack has no numeric leaderboard, so the proposal's external reference is the\n"
               "published accuracy of AMP Scanner v2. **The datasets are not the same** (different\n"
               "positives, different negatives, different split), so this is a sanity check on the\n"
               "order of magnitude, not a head-to-head comparison.\n\n")
    if not ext.get("verified"):
        out.append("> The reference digits below have NOT yet been checked against the paper by the\n"
                   "> team. Confirm them and set `verified: true` in `external_reference.json`.\n\n")
    out.append("| | sensitivity | specificity | MCC | ROC-AUC |\n|---|---|---|---|---|\n")
    e = ext["metrics"]
    out.append(f"| {ext['name']} ({ext['dataset']}) | {e['sensitivity']:.3f} | {e['specificity']:.3f} "
               f"| {e['mcc']:.3f} | {e['roc_auc']:.3f} |\n")
    if best:
        m, r = best
        out.append(f"| ours: {NICE.get(m, m)}, charge-matched negatives | {r.sensitivity[0]:.3f} | "
                   f"{r.specificity[0]:.3f} | {r.mcc[0]:.3f} | {r.roc_auc[0]:.3f} |\n")

# ---- candidate audit ------------------------------------------------------
for f in sorted(glob.glob("results/candidate_audit*.csv")):
    c = pd.read_csv(f)
    tag = os.path.basename(f).replace("candidate_audit", "").replace(".csv", "") or " (ESM judge)"
    out.append(f"\n## Search audit{tag}: judge scores for {len(c)} candidates\n\n")
    out.append("| group | mean judge P |\n|---|---|\n")
    out.append(f"| plant seeds (unmodified) | {c.seed_judge_p.mean():.3f} |\n")
    out.append(f"| random BLOSUM mutants (same edit count) | {c.random_mutant_judge_p.mean():.3f} |\n")
    out.append(f"| annealed candidates | {c.judge_p.mean():.3f} |\n")
    rs = np.random.RandomState(0)
    for name, other in [("plant seeds", c.seed_judge_p), ("random mutants", c.random_mutant_judge_p)]:
        d = (c.judge_p - other).groupby(c.seed_id).mean().values
        b = [rs.choice(d, len(d)).mean() for _ in range(5000)]
        out.append(f"\ncandidates minus {name}: {d.mean():+.3f} "
                   f"(95% bootstrap CI over seeds [{np.percentile(b, 2.5):+.3f}, "
                   f"{np.percentile(b, 97.5):+.3f}])\n")
    if "pct_identity_nearest_known" in c:
        out.append(f"\nnovelty: median edit distance {c.edit_dist_nearest_known.median():.0f}, "
                   f"median local-alignment identity to the nearest known AMP "
                   f"{c.pct_identity_nearest_known.median():.1f}%\n")

with open("results/summary.md", "w", encoding=ENC, newline="\n") as f:
    f.writelines(out)
print("".join(out))

# ---- figures --------------------------------------------------------------
SHORT = {"charge_only": "charge only", "logreg": "logistic reg.", "random_forest": "random forest",
         "cnn_lstm": "CNN-BiLSTM", "esm_mean": "ESM mean", "esm_attn1h": "ESM attn x1",
         "esm_attn": "ESM attn x4"}
fig, ax = plt.subplots(1, len(negs), figsize=(3.9 * len(negs), 4.0), sharey=True, squeeze=False)
for i, (a_, neg) in enumerate(zip(ax[0], negs)):
    ms = [m for m in ORDER if len(R[(R.model == m) & (R.negatives == neg) & (R.split == "cluster")])]
    vals = [R[(R.model == m) & (R.negatives == neg) & (R.split == "cluster")].iloc[0].mcc for m in ms]
    colours = ["#b0b0b0" if m in ("charge_only", "logreg", "random_forest") else "#4c78a8"
               for m in ms]
    a_.bar([SHORT.get(m, m) for m in ms], [v[0] for v in vals], yerr=[v[1] for v in vals],
           color=colours, capsize=3)
    a_.axhline(0, color="black", lw=0.8)
    a_.set_title(NEG_LABEL[neg], fontsize=10)
    if i == 0:
        a_.set_ylabel("MCC (held-out clusters)")
    a_.tick_params(axis="x", rotation=45, labelsize=8)
    for lb in a_.get_xticklabels():
        lb.set_horizontalalignment("right")
fig.suptitle("Model ladder: grey = descriptor baselines, blue = neural", fontsize=9, y=0.015)
plt.tight_layout(); plt.savefig("figures/ladder_mcc.png", dpi=160, bbox_inches="tight"); plt.close()

for f in sorted(glob.glob("results/candidate_audit*.csv")):
    c = pd.read_csv(f); plt.figure(figsize=(5, 3.8))
    plt.boxplot([c.seed_judge_p, c.random_mutant_judge_p, c.judge_p],
                tick_labels=["plant seeds", "random mutants", "SA candidates"])
    plt.ylabel("independent judge P(AMP)"); plt.title("Judge scores (never seen by search)")
    plt.tight_layout()
    plt.savefig("figures/" + os.path.basename(f).replace(".csv", ".png"), dpi=160); plt.close()
print("wrote results/summary.md and figures/*.png")
