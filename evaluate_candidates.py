"""Audit the generated candidates with the INDEPENDENT JUDGE (ensemble of heads trained on clusters the
scorer never saw) and compare against two controls:
  (a) the unmodified plant seeds, (b) random BLOSUM mutants with the same number of substitutions.
If annealing is doing real work, candidates should beat BOTH controls under the judge, not just
under the scorer that generated them.
Usage: python evaluate_candidates.py --judge_glob "checkpoints/judge_esm_attn_hard_s*.pt"
"""
import argparse, glob, random
import numpy as np, pandas as pd
from common import read_fasta
from search import mutate


def ensemble(paths, embedder):
    from models import load_scorer
    return [load_scorer(p, embedder) for p in paths]


def judge_p(judges, seqs):
    P = np.stack([j.predict(seqs) for j in judges]); return P.mean(0), P.std(0)


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
    c.to_csv(f"results/candidate_audit{a.tag}.csv", index=False)
    from common import write_fasta
    write_fasta(f"outputs/candidate_library_ranked{a.tag}.fasta", [(
        f"{r.cand_id}|seed={r.seed_id}|{r.species}|scorerP={r.scorer_p:.3f}|judgeP={r.judge_p:.3f}|charge={r.charge:.1f}|"
        f"uH={r.hydrophobic_moment:.2f}|dist_to_known={r.edit_dist_nearest_known}|COMPUTATIONAL_HYPOTHESIS", r.seq)
        for r in c.itertuples()])
    ref = pd.read_csv(a.data, sep="\t").query("label == 1")
    from common import net_charge, hydrophobic_moment
    q = ref.seq.map(net_charge); u = ref.seq.map(hydrophobic_moment)
    inside = ((c.charge.between(*q.quantile([.05, .95]))) & (c.hydrophobic_moment.between(*u.quantile([.05, .95])))).mean()
    print(f"\nJudge ensemble: {len(judges)} heads, mean threshold {thr:.2f}")
    print(f"  mean judge P  | plant seeds {c.seed_judge_p.mean():.3f} | random mutants {c.random_mutant_judge_p.mean():.3f} | SA candidates {c.judge_p.mean():.3f}")
    print(f"  candidates above judge threshold: {(c.judge_p >= thr).mean():.0%}   random-mutant controls: {(c.random_mutant_judge_p >= thr).mean():.0%}")
    print(f"  candidates inside 5-95% charge AND hydrophobic-moment range of known AMPs: {inside:.0%}")
    print(f"  median edit distance to nearest known AMP: {c.edit_dist_nearest_known.median():.0f}")
