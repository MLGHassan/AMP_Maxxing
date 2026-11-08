"""Rung 1 of the ladder: non-neural baselines on physicochemical descriptors.

Run under EXACTLY the protocol train.py uses (nested 5-fold cross-validation over
clusters + a fixed held-out test group), so the ladder comparison is like-for-like:
the threshold is always chosen on an inner validation split, never on the data it
is scored against.

Models : charge-only (the proposal's sanity floor), logistic regression, random forest.
Usage  : python baselines.py          # every negative set x cluster/random split
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import (ENC, descriptors, DESC_NAMES, load_dataset, partition, cv_folds,
                    split_by_cluster, metrics_at, best_mcc_threshold)

METRICS = ["roc_auc", "pr_auc", "mcc", "sensitivity", "specificity"]


def featurize(df):
    return np.array([descriptors(s) for s in df.seq])


def make_models(seed):
    return {
        "charge_only": None,
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(300, class_weight="balanced",
                                                n_jobs=-1, random_state=seed),
    }


def run(neg, mode, folds, role="scorer", path="data/processed/dataset.tsv"):
    df = load_dataset(path, neg)
    X = featurize(df); y = df.label.values
    test_rows, oof = {}, {}
    for k, (tr_all, ho, te) in enumerate(cv_folds(df, role, folds, mode=mode)):
        tr, va = split_by_cluster(df, tr_all, seed=k) if mode == "cluster" else \
                 (tr_all[: int(len(tr_all) * .85)], tr_all[int(len(tr_all) * .85):])
        for name, m in make_models(k).items():
            if m is None:                                   # score = net charge, squashed to [0,1]
                f = lambda idx: 1 / (1 + np.exp(-X[idx, 0]))
            else:
                m.fit(X[tr], y[tr]); f = lambda idx, m=m: m.predict_proba(X[idx])[:, 1]
            thr = best_mcc_threshold(y[va], f(va))          # inner validation only
            test_rows.setdefault(name, []).append(metrics_at(y[te], f(te), thr))
            oof.setdefault(name, []).append((y[ho], f(ho), thr))
    out = {}
    for name, rows in test_rows.items():
        yy = np.concatenate([a for a, _, _ in oof[name]])
        pp = np.concatenate([b for _, b, _ in oof[name]])
        thr = float(np.mean([t for _, _, t in oof[name]]))
        out[name] = {"test": {m: (float(np.mean([r[m] for r in rows])),
                                  float(np.std([r[m] for r in rows]))) for m in METRICS},
                     "cv_pooled": metrics_at(yy, pp, thr)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--negatives", nargs="+", default=["random", "hard", "peptide"])
    ap.add_argument("--data", default="data/processed/dataset.tsv")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    avail = set(pd.read_csv(a.data, sep="\t").sets.str.cat(sep=",").split(","))
    out = {}
    for neg in a.negatives:
        if neg not in avail:
            print(f"  skipping '{neg}' negatives: not present in {a.data}")
            continue
        for mode in ["cluster", "random"]:
            print(f"  {neg} negatives, {mode} split ...", flush=True)
            out[f"{neg}_negatives|{mode}_split"] = run(neg, mode, a.folds, path=a.data)
    with open("results/baselines.json", "w", encoding=ENC, newline="\n") as f:
        json.dump(out, f, indent=2)
    rows = [dict(negatives=k.split("|")[0].split("_")[0], split=k.split("|")[1].split("_")[0],
                 model=m,
                 **{n: f"{v['test'][n][0]:.3f}+/-{v['test'][n][1]:.3f}"
                    for n in ["roc_auc", "pr_auc", "mcc"]},
                 cv_mcc=f"{v['cv_pooled']['mcc']:.3f}")
            for k, d in out.items() for m, v in d.items()]
    t = pd.DataFrame(rows); print(t.to_string(index=False))
    t.to_csv("results/baselines.csv", index=False, lineterminator="\n", encoding=ENC)
