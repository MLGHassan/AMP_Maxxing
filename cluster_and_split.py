"""Cluster ALL sequences at 40% identity and assign whole clusters to groups.

Clustering uses CD-HIT if it is on PATH (cd-hit -c 0.4 -n 2), otherwise a built-in greedy
algorithm with the same identity definition (identical residues in the best local
alignment / length of the shorter sequence).  All sequences (positives AND both
negative sets) are clustered together, so near-duplicates never straddle groups.

Groups: test (15%), scorer pool (42.5%), judge pool (42.5%).  The judge is trained on
clusters the scorer never sees, and the test clusters are used by nobody for selection.
Output: data/processed/dataset.tsv (sequences.tsv + cluster + group columns)
"""
import argparse, os, shutil, subprocess, tempfile
import numpy as np, pandas as pd
from common import write_fasta, make_aligner, alignment_identity as identity_of

# backwards-compatible aliases (the aligner now lives in common.py so that
# search / candidate novelty can report the same identity measure)
_aligner = make_aligner


def identity(al, a, b):
    """identical aligned residues / length of the shorter sequence."""
    return identity_of(al, a, b)


def greedy_cluster(seqs, thr=0.40):
    """Greedy longest-first clustering at `thr` identity (CD-HIT's definition).

    About a minute for a few thousand peptides: most candidate pairs are thrown
    out by the cheap score bound inside `alignment_identity` before the
    traceback is ever computed.
    """
    al = make_aligner()
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    reps, cid = [], [-1] * len(seqs)
    for n, i in enumerate(order):
        s = seqs[i]
        for c, r in enumerate(reps):
            if identity_of(al, seqs[r], s) >= thr:
                cid[i] = c; break
        else:
            reps.append(i); cid[i] = len(reps) - 1
        if n % 500 == 0:
            print(f"  clustered {n}/{len(seqs)} ({len(reps)} clusters)", flush=True)
    print(f"  done: {len(seqs)} sequences, {len(reps)} clusters", flush=True)
    return cid


def cdhit_cluster(seqs, thr=0.40):
    with tempfile.TemporaryDirectory() as d:
        fa, out = os.path.join(d, "in.fa"), os.path.join(d, "out")
        write_fasta(fa, [(str(i), s) for i, s in enumerate(seqs)])
        word = 2 if thr < 0.5 else 3
        subprocess.run(["cd-hit", "-i", fa, "-o", out, "-c", str(thr), "-n", str(word),
                        "-d", "0", "-M", "4000", "-l", "5"], check=True, stdout=subprocess.DEVNULL)
        cid, cur = [-1] * len(seqs), -1
        for line in open(out + ".clstr", encoding="utf-8"):
            if line.startswith(">Cluster"):
                cur = int(line.split()[1])
            else:
                cid[int(line.split(">")[1].split("...")[0])] = cur
    return cid


def assign_groups(df, seed=0, fracs=(0.15, 0.425, 0.425)):
    """Cluster-level assignment balancing BOTH total rows and positives per group."""
    rng = np.random.RandomState(seed)
    cl = df.groupby("cluster").agg(n=("seq", "size"), pos=("label", "sum")).reset_index()
    cl = cl.sample(frac=1, random_state=rng).sort_values(["pos", "n"], ascending=False, kind="stable")
    names, f = ["test", "scorer", "judge"], np.array(fracs)
    tgt_n, tgt_p = f * len(df), f * df.label.sum()
    got_n, got_p, assign = np.zeros(3), np.zeros(3), {}
    for r in cl.itertuples():
        k = int(np.argmax((tgt_p - got_p) / tgt_p + (tgt_n - got_n) / tgt_n))
        assign[r.cluster] = names[k]; got_n[k] += r.n; got_p[k] += r.pos
    return df.cluster.map(assign)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed/sequences.tsv")
    ap.add_argument("--out", default="data/processed/dataset.tsv")
    ap.add_argument("--identity", type=float, default=0.40)
    ap.add_argument("--method", choices=["auto", "cdhit", "python"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.inp, sep="\t")
    seqs = df.seq.tolist()
    use_cd = a.method == "cdhit" or (a.method == "auto" and shutil.which("cd-hit"))
    print("Clustering with", "CD-HIT" if use_cd else "built-in greedy (slower)", f"at {a.identity:.0%} identity")
    df["cluster"] = cdhit_cluster(seqs, a.identity) if use_cd else greedy_cluster(seqs, a.identity)
    df["group"] = assign_groups(df, a.seed)
    df.to_csv(a.out, sep="\t", index=False, lineterminator="\n", encoding="utf-8")
    sizes = df.cluster.value_counts()
    print(f"{df.cluster.nunique()} clusters; largest = {sizes.iloc[0]} seqs; singletons = {(sizes == 1).sum()}")
    print(df.groupby(["group", "label"]).size().unstack(fill_value=0))
