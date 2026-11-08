"""Shared helpers: alphabet, physicochemical features, metrics, data partitioning.

Text I/O is forced to UTF-8 everywhere so that runs on Windows (default cp1252)
produce byte-identical files to runs on macOS/Linux.
"""
import math
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, confusion_matrix)

#: every text file this project writes or reads uses this encoding, on every OS
ENC = "utf-8"

# A Windows console is usually cp1252, and the sequence databases contain characters
# it cannot encode (DRAMP has entries with Greek letters). Printing one would raise
# UnicodeEncodeError and kill the run, so degrade to '?' instead of crashing. Files
# are unaffected: those are written as UTF-8 explicitly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):      # not a reconfigurable text stream
        pass


def ascii_safe(text, limit=60):
    """Collapse arbitrary database text to printable ASCII, for log lines and reasons."""
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in str(text))[:limit]

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
    with open(path, encoding=ENC) as f:
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
    # newline="\n" keeps FASTA byte-identical on Windows (no CRLF)
    with open(path, "w", encoding=ENC, newline="\n") as f:
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
    """Rows of one comparison: the positives that have a partner in `neg`, plus that set.

    The `sets` column is filled in for positives too, so a positive that could not be
    matched (no charge-matched fragment, or no peptide of exactly its length) is left
    out of that comparison instead of facing a missing counterpart.
    """
    df = pd.read_csv(path, sep="\t")
    sets = df["sets"].fillna("")
    return df[sets.str.contains(neg)].reset_index(drop=True)


def partition(df, role="scorer", seed=0, mode="cluster", val_frac=0.15):
    """Return (train_idx, val_idx, test_idx) as index arrays into df.

    cluster mode: test = the fixed 'test' clusters; the role's pool is split
    into train/val BY CLUSTER (seeded), so no homologous pair straddles a split.
    random mode: ignores clusters entirely (shows how much leakage inflates scores).
    """
    rng = np.random.RandomState(seed)
    if mode == "random":
        tr_c, va_c, te_c = partition(df, role, seed, "cluster")     # same sizes as the cluster split
        idx = rng.permutation(len(df))
        n_te, n_va, n_tr = len(te_c), len(va_c), len(tr_c)
        return idx[n_te + n_va:n_te + n_va + n_tr], idx[n_te:n_te + n_va], idx[:n_te]
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


# ----------------------------------------------------------------------------
# Local-alignment sequence identity (the quantity BLAST reports as "% identity")
# ----------------------------------------------------------------------------

def make_aligner():
    """Local aligner used for both clustering and novelty reporting."""
    from Bio import Align
    al = Align.PairwiseAligner()
    al.mode = "local"
    al.match_score, al.mismatch_score = 1.0, -1.0
    al.open_gap_score, al.extend_gap_score = -3.0, -1.0
    return al


def alignment_identity(al, a: str, b: str) -> float:
    """Identical residues in the best local alignment / length of the shorter sequence.

    This is the same definition CD-HIT uses at -c, and the same quantity BLASTP
    prints as 'Identities' normalised by the shorter sequence.
    """
    if len(a) < len(b):
        a, b = b, a
    if al.score(a, b) < 0.25 * len(b):          # cheap reject: cannot reach 40%
        return 0.0
    aln = al.align(a, b)[0]
    m = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        m += sum(x == y for x, y in zip(a[s1:e1], b[s2:e2]))
    return m / len(b)


def identity_nearest(query: str, refs, al=None):
    """(max local-alignment identity, nearest ref) of query against a list of strings."""
    al = al or make_aligner()
    best, arg = 0.0, None
    for r in refs:
        v = alignment_identity(al, query, r)
        if v > best:
            best, arg = v, r
    return best, arg


# ----------------------------------------------------------------------------
# Cluster-grouped 5-fold cross-validation (the proposal's evaluation protocol)
# ----------------------------------------------------------------------------

def split_by_cluster(df, idx, seed=0, val_frac=0.15):
    """Split an index array into (train, val) BY CLUSTER, for inner model selection."""
    rng = np.random.RandomState(seed)
    cl = df.cluster.values[idx]
    uniq = np.unique(cl)
    rng.shuffle(uniq)
    val = set(uniq[:max(1, int(len(uniq) * val_frac))])
    is_val = np.array([c in val for c in cl])
    return idx[~is_val], idx[is_val]


def cv_folds(df, role="scorer", n_splits=5, seed=0, mode="cluster"):
    """Yield (train_idx, val_idx, test_idx) for cluster-grouped K-fold CV.

    The role's pool is partitioned into `n_splits` folds BY CLUSTER, so no two
    homologous sequences ever land on opposite sides of a fold boundary.  Each
    fold trains on K-1 parts and selects epoch/threshold on the held-out part.
    `test_idx` is the fixed 'test' group every time: it is scored once per fold
    and never used for any selection decision.

    mode='random' ignores clusters while keeping identical fold sizes, which is
    the leakage control.
    """
    from sklearn.model_selection import GroupKFold, KFold
    test = np.where(df.group == "test")[0]
    pool = df[df.group == role]
    idx = pool.index.values
    if mode == "random":
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(idx)
        for tr, va in KFold(n_splits).split(shuffled):
            yield shuffled[tr], shuffled[va], test
        return
    groups = pool.cluster.values
    # GroupKFold is deterministic; shuffle cluster labels so `seed` still varies folds
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    remap = dict(zip(uniq, rng.permutation(len(uniq))))
    groups = np.array([remap[g] for g in groups])
    for tr, va in GroupKFold(n_splits).split(idx, groups=groups):
        yield idx[tr], idx[va], test
