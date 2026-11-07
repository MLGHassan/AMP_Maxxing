"""Collect results/*.json into results/summary.md and figures/*.png."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
rows = []
if os.path.exists("results/baselines.json"):
    for k, d in json.load(open("results/baselines.json")).items():
        neg, split = k.split("|"); neg = neg.split("_")[0]; split = split.split("_")[0]
        for m, v in d.items():
            rows.append(dict(model=m, negatives=neg, split=split, **{n: v[n] for n in ["roc_auc", "pr_auc", "mcc"]}))
for f in glob.glob("results/scorer_*_*_*.json"):
    j = json.load(open(f)); a = j["args"]
    rows.append(dict(model=a["model"], negatives=a["neg"], split=a["mode"], **{n: j["summary"][n] for n in ["roc_auc", "pr_auc", "mcc"]}))
R = pd.DataFrame(rows)
fmt = lambda t: f"{t[0]:.3f} ± {t[1]:.3f}"
order = ["charge_only", "logreg", "random_forest", "cnn_lstm", "esm_mean", "esm_attn"]
with open("results/summary.md", "w") as f:
    for neg in ["random", "hard"]:
        f.write(f"\n### {neg}-negatives, cluster-held-out test set (mean ± sd over seeds)\n\n| model | ROC-AUC | PR-AUC | MCC |\n|---|---|---|---|\n")
        for m in order:
            r = R[(R.model == m) & (R.negatives == neg) & (R.split == "cluster")]
            if len(r):
                r = r.iloc[0]; f.write(f"| {m} | {fmt(r.roc_auc)} | {fmt(r.pr_auc)} | {fmt(r.mcc)} |\n")
    f.write("\n### Leakage check: same model, cluster split vs naive random split (MCC)\n\n| model | negatives | cluster split | random split |\n|---|---|---|---|\n")
    for (m, neg), g in R.groupby(["model", "negatives"]):
        if set(g.split) >= {"cluster", "random"}:
            c = g[g.split == "cluster"].iloc[0].mcc; r = g[g.split == "random"].iloc[0].mcc
            f.write(f"| {m} | {neg} | {fmt(c)} | {fmt(r)} |\n")
print(open("results/summary.md").read())

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a_, neg in zip(ax, ["random", "hard"]):
    ms = [m for m in order if len(R[(R.model == m) & (R.negatives == neg) & (R.split == "cluster")])]
    vals = [R[(R.model == m) & (R.negatives == neg) & (R.split == "cluster")].iloc[0].mcc for m in ms]
    a_.bar(ms, [v[0] for v in vals], yerr=[v[1] for v in vals], color="#4c78a8")
    a_.set_title(f"{neg} negatives"); a_.set_ylabel("MCC (held-out clusters)"); a_.tick_params(axis="x", rotation=40)
plt.tight_layout(); plt.savefig("figures/ladder_mcc.png", dpi=160); plt.close()

for f in glob.glob("results/candidate_audit*.csv"):
    c = pd.read_csv(f); plt.figure(figsize=(5, 3.8))
    plt.boxplot([c.seed_judge_p, c.random_mutant_judge_p, c.judge_p], tick_labels=["plant seeds", "random mutants", "SA candidates"])
    plt.ylabel("independent judge P(AMP)"); plt.title("Judge scores (never seen by search)"); plt.tight_layout()
    plt.savefig("figures/" + os.path.basename(f).replace(".csv", ".png"), dpi=160); plt.close()
