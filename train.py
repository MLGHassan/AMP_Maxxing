"""Train one rung of the ladder on one pool (scorer | judge) and one negative set.

Protocol (the proposal's evaluation plan, --protocol cv, the default)
--------------------------------------------------------------------
5-fold cross-validation over the clusters of the role's pool, PLUS a held-out
test group that is touched once per fold and never used for any selection:

  for each of the 5 outer folds
      inner split of the 4 training folds into inner-train / inner-val BY CLUSTER
      epoch chosen on inner-val ROC-AUC, threshold chosen on inner-val MCC
      predict the outer held-out fold (out-of-fold) and the fixed test group

  cv  = metrics on the POOLED out-of-fold predictions (the cross-validated
        estimate; the threshold never saw the fold it is applied to)
  test = per-fold metrics on the fixed test group, reported as mean +/- sd

--protocol holdout reproduces the earlier repeated cluster-grouped hold-out
(`--seeds n` re-partitions train/val n times) for comparison.

Usage: python train.py --role scorer --model esm_attn --neg hard
"""
import argparse, json, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from common import (ENC, load_dataset, partition, cv_folds, split_by_cluster,
                    metrics_at, best_mcc_threshold)
from models import HEADS, encode_ids

METRICS = ["roc_auc", "pr_auc", "mcc", "sensitivity", "specificity"]


def _x(X, b, M, device):
    x = X[b]
    return (x.float() if M is not None else x).to(device)   # ESM features -> float, token ids stay long


def predict(head, kind, X, M, idx, device, bs=256):
    head.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            logit = head(_x(X, b, M, device), None if M is None else M[b].to(device))
            out.append(torch.sigmoid(logit).squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(kind, X, M, y, tr, va, seed, a, device):
    """Train with early stopping on inner-validation ROC-AUC. -> (head, thr, val_auc, epochs)."""
    torch.manual_seed(seed); np.random.seed(seed)
    head = HEADS[kind]().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-3)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(y, dtype=torch.float32)
    best, best_state, bad, ep = -1.0, None, 0, 0
    for ep in range(a.epochs):
        head.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            logit = head(_x(X, b, M, device), None if M is None else M[b].to(device)).squeeze(-1)
            loss = lossf(logit, yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        auc = roc_auc_score(y[va], predict(head, kind, X, M, va, device))
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    thr = best_mcc_threshold(y[va], predict(head, kind, X, M, va, device))
    return head, thr, float(best), ep + 1


def agg(rows):
    return {m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows])))
            for m in METRICS}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["scorer", "judge"], default="scorer")
    ap.add_argument("--model", choices=list(HEADS), default="esm_attn")
    ap.add_argument("--neg", choices=["random", "hard", "peptide"], default="hard")
    ap.add_argument("--mode", choices=["cluster", "random"], default="cluster",
                    help="'random' ignores clusters: the leakage control")
    ap.add_argument("--protocol", choices=["cv", "holdout"], default="cv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5, help="only used by --protocol holdout")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--data", default="data/processed/dataset.tsv")
    a = ap.parse_args()
    from esm_embed import pick_device
    device = "cpu" if a.model == "cnn_lstm" and not torch.cuda.is_available() else pick_device()
    df = load_dataset(a.data, a.neg); y = df.label.values
    print(f"{a.role}/{a.model}/{a.neg}-negatives/{a.mode}-split/{a.protocol} on {device}: "
          f"{len(df)} sequences ({int(y.sum())} positive)")
    if a.model == "cnn_lstm":
        X, M = encode_ids(df.seq.tolist()), None
    else:
        from esm_embed import EsmEmbedder
        X, M = EsmEmbedder(device=device).embed(df.seq.tolist())
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)

    test_rows, oof_p, oof_y, thrs, out = [], [], [], [], {}
    if a.protocol == "cv":
        for k, (tr_all, ho, te) in enumerate(cv_folds(df, a.role, a.folds, mode=a.mode)):
            tr, va = split_by_cluster(df, tr_all, seed=k) if a.mode == "cluster" else \
                     (tr_all[: int(len(tr_all) * .85)], tr_all[int(len(tr_all) * .85):])
            head, thr, vauc, neps = fit(a.model, X, M, y, tr, va, k, a, device)
            p_ho = predict(head, a.model, X, M, ho, device)
            oof_p.append(p_ho); oof_y.append(y[ho]); thrs.append(thr)
            r = metrics_at(y[te], predict(head, a.model, X, M, te, device), thr)
            r.update(fold=k, val_auc=vauc, epochs_run=neps,
                     holdout_fold_auc=float(roc_auc_score(y[ho], p_ho)))
            test_rows.append(r)
            print(f"  fold {k}: test AUC {r['roc_auc']:.3f} PR {r['pr_auc']:.3f} MCC {r['mcc']:.3f} "
                  f"sens {r['sensitivity']:.2f} spec {r['specificity']:.2f} | "
                  f"out-of-fold AUC {r['holdout_fold_auc']:.3f} ({neps} ep)")
            if a.mode == "cluster":
                torch.save({"kind": a.model, "state_dict": head.cpu().state_dict(), "threshold": thr},
                           f"checkpoints/{a.role}_{a.model}_{a.neg}_s{k}.pt")
        cv = metrics_at(np.concatenate(oof_y), np.concatenate(oof_p), float(np.mean(thrs)))
        out["cv_pooled"] = cv
        print(f"  CV (pooled out-of-fold, {a.folds} folds)  AUC {cv['roc_auc']:.3f}  "
              f"PR-AUC {cv['pr_auc']:.3f}  MCC {cv['mcc']:.3f}  "
              f"sens {cv['sensitivity']:.2f}  spec {cv['specificity']:.2f}")
    else:
        for seed in range(a.seeds):
            tr, va, te = partition(df, a.role, seed, a.mode)
            head, thr, vauc, neps = fit(a.model, X, M, y, tr, va, seed, a, device)
            r = metrics_at(y[te], predict(head, a.model, X, M, te, device), thr)
            r.update(fold=seed, val_auc=vauc, epochs_run=neps)
            test_rows.append(r)
            print(f"  seed {seed}: AUC {r['roc_auc']:.3f} PR-AUC {r['pr_auc']:.3f} "
                  f"MCC {r['mcc']:.3f} sens {r['sensitivity']:.2f} spec {r['specificity']:.2f}")
            if a.mode == "cluster":
                torch.save({"kind": a.model, "state_dict": head.cpu().state_dict(), "threshold": thr},
                           f"checkpoints/{a.role}_{a.model}_{a.neg}_s{seed}.pt")

    summ = agg(test_rows)
    out.update(args=vars(a), per_fold=test_rows, summary=summ)
    print("  TEST MEAN+/-SD  " + "  ".join(f"{k} {v[0]:.3f}+/-{v[1]:.3f}" for k, v in summ.items()))
    # the holdout protocol gets its own filename so it cannot overwrite the
    # cross-validated result for the same model/negatives/split
    suffix = "" if a.protocol == "cv" else f"_{a.protocol}"
    path = f"results/{a.role}_{a.model}_{a.neg}_{a.mode}{suffix}.json"
    with open(path, "w", encoding=ENC, newline="\n") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")
