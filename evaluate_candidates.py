"""Audit the generated candidates with the INDEPENDENT JUDGE (ensemble of heads trained on clusters the
scorer never saw) and compare against two controls:
  (a) the unmodified plant seeds, (b) random BLOSUM mutants with the same number of substitutions.
If annealing is doing real work, candidates should beat BOTH controls under the judge, not just
under the scorer that generated them.
Usage: python evaluate_candidates.py --judge_glob "checkpoints/judge_esm_attn_hard_s*.pt"
"""
import argparse, glob, os, random, shutil, subprocess, tempfile
import numpy as np, pandas as pd
from common import ENC, read_fasta, write_fasta
from search import mutate


def ensemble(paths, embedder):
    from models import load_scorer
    return [load_scorer(p, embedder) for p in paths]


def judge_p(judges, seqs):
    P = np.stack([j.predict(seqs) for j in judges]); return P.mean(0), P.std(0)


def blastp_identity(queries, refs):
    """Best BLASTP % identity of each query against `refs`, or None if BLAST+ is absent.

    The local-alignment identity already in the CSV measures the same quantity;
    this is here because the proposal named BLAST specifically, so when BLAST+ is
    installed we report its own numbers rather than a stand-in.
    """
    if not (shutil.which("blastp") and shutil.which("makeblastdb")):
        return None
    with tempfile.TemporaryDirectory() as d:
        dbf, qf = os.path.join(d, "ref.fasta"), os.path.join(d, "q.fasta")
        write_fasta(dbf, [(f"ref{i}", s) for i, s in enumerate(refs)])
        write_fasta(qf, [(f"q{i}", s) for i, s in enumerate(queries)])
        subprocess.run(["makeblastdb", "-in", dbf, "-dbtype", "prot"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = subprocess.run(["blastp", "-query", qf, "-db", dbf, "-outfmt", "6 qseqid pident",
                              "-max_target_seqs", "5", "-evalue", "1000", "-word_size", "2",
                              "-matrix", "BLOSUM62", "-comp_based_stats", "0"],
                             check=True, capture_output=True, text=True).stdout
    best = {}
    for line in out.splitlines():
        q, pid = line.split("\t")[:2]
        best[q] = max(best.get(q, 0.0), float(pid))
    return [best.get(f"q{i}", float("nan")) for i in range(len(queries))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="outputs/candidates.csv")
    ap.add_argument("--judge_glob", default="checkpoints/judge_esm_attn_hard_s*.pt")
    ap.add_argument("--data", default="data/processed/dataset.tsv")
    ap.add_argument("--n_controls", type=int, default=10)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _cnn")
    a = ap.parse_args()
    from esm_embed import EsmEmbedder, pick_device
    paths = sorted(glob.glob(a.judge_glob)); assert paths, f"no judge checkpoints match {a.judge_glob}"
    emb = EsmEmbedder(device=pick_device()) if "esm" in paths[0] else None
    judges = ensemble(paths, emb)
    thr = float(np.mean([j.threshold for j in judges]))
    c = pd.read_csv(a.cands)
    c["judge_p"], c["judge_sd"] = judge_p(judges, c.seq.tolist())
    c["seed_judge_p"], _ = judge_p(judges, c.seed_seq.tolist())
    rng = random.Random(1); ctrl = []
    for r in c.itertuples():
        vals = []
        for _ in range(a.n_controls):
            s = r.seed_seq
            for _ in range(r.n_mut):
                s = mutate(s, rng, r.seed_seq, 10 ** 6)
            vals.append(s)
        ctrl.append(judge_p(judges, vals)[0].mean())
    c["random_mutant_judge_p"] = ctrl
    c["judge_call"] = np.where(c.judge_p >= thr, "predicted AMP", "below threshold")
    ref_all = pd.read_csv(a.data, sep="\t").query("label == 1")
    bl = blastp_identity(c.seq.tolist(), ref_all.seq.tolist())
    if bl is not None:
        c["blast_pct_identity_nearest_known"] = bl
    os.makedirs("results", exist_ok=True); os.makedirs("outputs", exist_ok=True)
    c.to_csv(f"results/candidate_audit{a.tag}.csv", index=False, lineterminator="\n", encoding=ENC)
    write_fasta(f"outputs/candidate_library_ranked{a.tag}.fasta", [(
        f"{r.cand_id}|seed={r.seed_id}|{r.species}|scorerP={r.scorer_p:.3f}|judgeP={r.judge_p:.3f}|charge={r.charge:.1f}|"
        f"uH={r.hydrophobic_moment:.2f}|dist_to_known={r.edit_dist_nearest_known}|"
        f"id_to_known={r.pct_identity_nearest_known}pct|COMPUTATIONAL_HYPOTHESIS", r.seq)
        for r in c.itertuples()])
    ref = ref_all
    from common import net_charge, hydrophobic_moment
    q = ref.seq.map(net_charge); u = ref.seq.map(hydrophobic_moment)
    inside = ((c.charge.between(*q.quantile([.05, .95]))) & (c.hydrophobic_moment.between(*u.quantile([.05, .95])))).mean()
    print(f"\nJudge ensemble: {len(judges)} heads, mean threshold {thr:.2f}")
    print(f"  mean judge P  | plant seeds {c.seed_judge_p.mean():.3f} | random mutants {c.random_mutant_judge_p.mean():.3f} | SA candidates {c.judge_p.mean():.3f}")
    print(f"  candidates above judge threshold: {(c.judge_p >= thr).mean():.0%}   random-mutant controls: {(c.random_mutant_judge_p >= thr).mean():.0%}")
    print(f"  candidates inside 5-95% charge AND hydrophobic-moment range of known AMPs: {inside:.0%}")
    print(f"  novelty: median edit distance to nearest known AMP "
          f"{c.edit_dist_nearest_known.median():.0f}; median local-alignment identity "
          f"{c.pct_identity_nearest_known.median():.1f}%")
    if bl is not None:
        print(f"           BLASTP median best identity {np.nanmedian(bl):.1f}% "
              f"(max {np.nanmax(bl):.1f}%)")
    else:
        print("           BLAST+ not on PATH, so the alignment identity above is the "
              "reported measure (same definition)")
    rs = np.random.RandomState(0)
    for name, other in [("plant seeds", c.seed_judge_p), ("random mutants", c.random_mutant_judge_p)]:
        d = (c.judge_p - other).groupby(c.seed_id).mean().values
        b = [rs.choice(d, len(d)).mean() for _ in range(5000)]
        print(f"  candidates minus {name}: {d.mean():+.3f}  95% CI [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")