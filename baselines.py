"""Non-neural baselines on physicochemical descriptors, with the SAME cluster split as the
neural models.  Also shows how much a naive random split inflates scores (leakage check).

Models : charge-only (sanity floor), logistic regression, random forest.
Usage  : python baselines.py            # runs both negative sets
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import descriptors, DESC_NAMES, load_dataset, partition, metrics_at, best_mcc_threshold


def featurize(df):
    return np.array([descriptors(s) for s in df.seq])


def run(neg, mode, seeds, role="scorer", path="data/processed/dataset.tsv"):
    df = load_dataset(path, neg)
    X = featurize(df); y = df.label.values
    res = {}
    for seed in range(seeds):
        tr, va, te = partition(df, role, seed, mode)
        models = {
            "charge_only": None,
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            "random_forest": RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=seed),
        }
        for name, m in models.items():
            if m is None:                                   # score = net charge, rescaled to [0,1]
                f = lambda idx: 1 / (1 + np.exp(-X[idx, 0]))
            else:
                m.fit(X[tr], y[tr]); f = lambda idx, m=m: m.predict_proba(X[idx])[:, 1]
            thr = best_mcc_threshold(y[va], f(va))          # threshold from validation only
            res.setdefault(name, []).append(metrics_at(y[te], f(te), thr))
    return {k: {m: (float(np.mean([r[m] for r in v])), float(np.std([r[m] for r in v])))
                for m in ["roc_auc", "pr_auc", "mcc", "sensitivity", "specificity"]} for k, v in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", default="data/processed/dataset.tsv")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    out = {}
    for neg in ["random", "hard"]:
        for mode in ["cluster", "random"]:
            out[f"{neg}_negatives|{mode}_split"] = run(neg, mode, a.seeds, path=a.data)
    json.dump(out, open("results/baselines.json", "w"), indent=2)
    rows = [dict(negatives=k.split("|")[0].split("_")[0], split=k.split("|")[1].split("_")[0], model=m,
                 **{n: f"{v[n][0]:.3f}±{v[n][1]:.3f}" for n in ["roc_auc", "pr_auc", "mcc"]})
            for k, d in out.items() for m, v in d.items()]
    t = pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv("results/baselines.csv", index=False)
