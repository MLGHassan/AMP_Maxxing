"""Shared helpers: alphabet, physicochemical features, metrics, data partitioning."""
import math
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, confusion_matrix)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
AA2ID = {a: i + 1 for i, a in enumerate(AA)}      # 0 = padding
MAX_LEN = 50

# Eisenberg consensus hydrophobicity scale
EISENBERG = {'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
             'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
             'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
             'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08}

# EMBOSS pKa values, used for net charge at a given pH
_POS = {'K': 10.8, 'R': 12.5, 'H': 6.5}
_NEG = {'D': 3.9, 'E': 4.1, 'C': 8.5, 'Y': 10.1}
_NTERM, _CTERM = 8.6, 3.6


def net_charge(seq: str, ph: float = 7.0) -> float:
    q = 1.0 / (1.0 + 10 ** (ph - _NTERM)) - 1.0 / (1.0 + 10 ** (_CTERM - ph))
    for a in seq:
        if a in _POS:
            q += 1.0 / (1.0 + 10 ** (ph - _POS[a]))
        elif a in _NEG:
            q -= 1.0 / (1.0 + 10 ** (_NEG[a] - ph))
    return q


def hydrophobicity(seq: str) -> float:
    return sum(EISENBERG[a] for a in seq) / len(seq)


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    ang = math.radians(angle_deg)
    s = sum(EISENBERG[a] * math.sin(i * ang) for i, a in enumerate(seq))
    c = sum(EISENBERG[a] * math.cos(i * ang) for i, a in enumerate(seq))
    return math.sqrt(s * s + c * c) / len(seq)


def descriptors(seq: str) -> list:
    """[net charge pH7, hydrophobicity, hydrophobic moment, pI, MW, length]."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    pa = ProteinAnalysis(seq)
    return [net_charge(seq), hydrophobicity(seq), hydrophobic_moment(seq),
            pa.isoelectric_point(), pa.molecular_weight(), float(len(seq))]

DESC_NAMES = ["charge_pH7", "hydrophobicity", "hydrophobic_moment", "pI", "MW", "length"]


def is_canonical(seq: str, lo=10, hi=MAX_LEN) -> bool:
    return lo <= len(seq) <= hi and set(seq) <= AA_SET


def read_fasta(path):
    recs, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf)))
                name, buf = line[1:], []
            elif line:
                buf.append(line)
    if name is not None:
        recs.append((name, "".join(buf)))
    return recs


def write_fasta(path, recs):
    with open(path, "w") as f:
        for h, s in recs:
            f.write(f">{h}\n{s}\n")


def metrics_at(y, p, thr=0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"mcc": float(matthews_corrcoef(y, pred)) if len(set(pred)) > 1 else 0.0,
           "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
           "threshold": float(thr), "n": int(len(y)), "n_pos": int(y.sum())}
    if len(set(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return out


def best_mcc_threshold(y, p) -> float:
    """Threshold maximising MCC (chosen on VALIDATION data only)."""
    best, thr = -2.0, 0.5
    for t in np.unique(np.round(p, 4)):
        m = metrics_at(y, p, t)["mcc"]
        if m > best:
            best, thr = m, float(t)
    return thr


def load_dataset(path="data/processed/dataset.tsv", neg="hard") -> pd.DataFrame:
    """Rows = all positives + the chosen negative set ('random' or 'hard')."""
    df = pd.read_csv(path, sep="\t")
    keep = (df.label == 1) | df.sets.str.contains(neg)
    return df[keep].reset_index(drop=True)


def partition(df, role="scorer", seed=0, mode="cluster", val_frac=0.15):
    """Return (train_idx, val_idx, test_idx) as index arrays into df.

    cluster mode: test = the fixed 'test' clusters; the role's pool is split
    into train/val BY CLUSTER (seeded), so no homologous pair straddles a split.
    random mode: ignores clusters entirely (shows how much leakage inflates scores).
    """
    rng = np.random.RandomState(seed)
    if mode == "random":
        idx = rng.permutation(len(df))
        n_te, n_va = int(0.15 * len(df)), int(0.15 * len(df))
        return idx[n_te + n_va:], idx[n_te:n_te + n_va], idx[:n_te]
    test = np.where(df.group == "test")[0]
    pool = df[df.group == role]
    cl = pool.cluster.unique()
    rng.shuffle(cl)
    val_cl = set(cl[:max(1, int(len(cl) * val_frac))])
    is_val = pool.cluster.isin(val_cl).values
    return pool.index.values[~is_val], pool.index.values[is_val], test


def levenshtein_nearest(query: str, refs):
    """(min edit distance, nearest ref) of query against a list of strings."""
    from rapidfuzz.distance import Levenshtein
    best, arg = 10 ** 9, None
    for r in refs:
        d = Levenshtein.distance(query, r)
        if d < best:
            best, arg = d, r
    return best, arg
