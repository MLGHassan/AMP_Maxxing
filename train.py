"""Train one rung of the ladder on one pool (scorer | judge) and one negative set.

* train / val / test are split BY CLUSTER (see common.partition).
* the epoch is chosen on validation ROC-AUC, the decision threshold on validation MCC;
  the test set is only evaluated once per seed, after training has finished.
* each seed re-partitions the pool's clusters into train/val, giving mean +- sd.

Usage: python train.py --role scorer --model esm_attn --neg hard --seeds 5
"""
import argparse, json, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from common import load_dataset, partition, metrics_at, best_mcc_threshold
from models import HEADS, encode_ids


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
    return np.concatenate(out)


def train_one(kind, X, M, y, tr, va, te, seed, a, device):
    torch.manual_seed(seed); np.random.seed(seed)
    head = HEADS[kind]().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-3)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(y, dtype=torch.float32)
    best, best_state, bad = -1, None, 0
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
    head.load_state_dict(best_state)
    pv = predict(head, kind, X, M, va, device)
    thr = best_mcc_threshold(y[va], pv)
    res = metrics_at(y[te], predict(head, kind, X, M, te, device), thr)
    res.update(val_auc=float(best), epochs_run=ep + 1)
    return head, thr, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["scorer", "judge"], default="scorer")
    ap.add_argument("--model", choices=list(HEADS), default="esm_attn")
    ap.add_argument("--neg", choices=["random", "hard"], default="hard")
    ap.add_argument("--mode", choices=["cluster", "random"], default="cluster")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--data", default="data/processed/dataset.tsv")
    a = ap.parse_args()
    from esm_embed import pick_device
    device = "cpu" if a.model == "cnn_lstm" and not torch.cuda.is_available() else pick_device()
    df = load_dataset(a.data, a.neg); y = df.label.values
    print(f"{a.role}/{a.model}/{a.neg}-negatives/{a.mode}-split on {device}: {len(df)} sequences")
    if a.model == "cnn_lstm":
        X, M = encode_ids(df.seq.tolist()), None
    else:
        from esm_embed import EsmEmbedder
        X, M = EsmEmbedder(device=device).embed(df.seq.tolist())
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)
    allres = []
    for seed in range(a.seeds):
        tr, va, te = partition(df, a.role, seed, a.mode)
        head, thr, r = train_one(a.model, X, M, y, tr, va, te, seed, a, device)
        allres.append(r)
        print(f"  seed {seed}: AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  MCC {r['mcc']:.3f}  "
              f"sens {r['sensitivity']:.2f} spec {r['specificity']:.2f}  (val AUC {r['val_auc']:.3f}, {r['epochs_run']} ep)")
        if a.mode == "cluster":
            torch.save({"kind": a.model, "state_dict": head.cpu().state_dict(), "threshold": thr},
                       f"checkpoints/{a.role}_{a.model}_{a.neg}_s{seed}.pt")
    summ = {m: (float(np.mean([r[m] for r in allres])), float(np.std([r[m] for r in allres])))
            for m in ["roc_auc", "pr_auc", "mcc", "sensitivity", "specificity"]}
    print("  MEAN±SD  " + "  ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in summ.items()))
    json.dump({"args": vars(a), "per_seed": allres, "summary": summ},
              open(f"results/{a.role}_{a.model}_{a.neg}_{a.mode}.json", "w"), indent=2)
