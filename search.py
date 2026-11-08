"""Simulated annealing over plant AMP seeds, using a trained SCORER as the fitness function.

fitness(s) = P_scorer(AMP | s) - lambda * penalty(s)
moves      : single-residue substitution weighted by BLOSUM62, <= max_mut substitutions from the seed,
             length is never changed
cooling    : T <- 0.95 * T every `steps_per_temp` proposals (geometric)
output     : outputs/candidates.csv and outputs/candidates.fasta (ranked by fitness).  The independent
             JUDGE is never loaded here -- it is used only in evaluate_candidates.py.
Usage      : python search.py --scorer checkpoints/scorer_esm_attn_hard_s0.pt --n_seeds 25
"""
import argparse, math, os, random
import numpy as np, pandas as pd
from Bio.Align import substitution_matrices
from common import (AA, ENC, EISENBERG, read_fasta, write_fasta, net_charge, hydrophobicity,
                    hydrophobic_moment, levenshtein_nearest, identity_nearest, make_aligner)

B62 = substitution_matrices.load("BLOSUM62")


def penalty(s):
    q, h = net_charge(s), hydrophobicity(s)
    return 0.25 * max(0.0, 2.0 - q) + 0.40 * max(0.0, h - 0.65) + 0.10 * max(0.0, q - 10.0)


def mutate(s, rng, seed_seq, max_mut):
    for _ in range(50):
        i = rng.randrange(len(s))
        cands = [a for a in AA if a != s[i]]
        w = [math.exp(B62[s[i]][a] / 2.0) for a in cands]
        new = s[:i] + rng.choices(cands, w)[0] + s[i + 1:]
        if sum(x != y for x, y in zip(new, seed_seq)) <= max_mut:
            return new
    return s


def anneal(seed_seq, scorer, rng, steps=300, t0=0.1, cool=0.95, steps_per_temp=5, lam=0.6, max_mut=5, top_k=3):
    cache = {}
    def fit(s):
        if s not in cache:
            cache[s] = float(scorer.predict([s])[0])
        return cache[s] - lam * penalty(s), cache[s]
    cur = seed_seq; cur_f, _ = fit(cur); T = t0
    for step in range(steps):
        cand = mutate(cur, rng, seed_seq, max_mut)
        if cand == cur:
            continue
        f, _ = fit(cand)
        if f > cur_f or rng.random() < math.exp((f - cur_f) / max(T, 1e-6)):
            cur, cur_f = cand, f
        if (step + 1) % steps_per_temp == 0:
            T *= cool
    ranked = sorted(((s, cache[s] - lam * penalty(s), cache[s]) for s in cache if s != seed_seq),
                    key=lambda x: -x[1])
    return ranked[:top_k], cache.get(seed_seq)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="checkpoints/scorer_esm_attn_hard_s0.pt")
    ap.add_argument("--seeds_fasta", default="data/processed/plant_seeds.fasta")
    ap.add_argument("--data", default="data/processed/dataset.tsv")
    ap.add_argument("--n_seeds", type=int, default=25)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max_mut", type=int, default=5)
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()
    from esm_embed import EsmEmbedder, pick_device
    from models import load_scorer
    dev = pick_device()
    scorer = load_scorer(a.scorer, EsmEmbedder(device=dev) if "esm" in a.scorer else None, "cpu" if dev == "mps" else dev)
    rng = random.Random(a.rng)
    seeds = [(h, s) for h, s in read_fasta(a.seeds_fasta) if set(s) <= set(AA)]
    grp = pd.read_csv(a.data, sep="\t").set_index("id").group.to_dict()
    seeds = [(h, s) for h, s in seeds if grp.get(h.split("|")[0]) in ("scorer", "test")]   # judge never saw these
    print(f"{len(seeds)} eligible plant seeds")
    rng.shuffle(seeds); seeds = seeds[:a.n_seeds]
    train_pos = pd.read_csv(a.data, sep="\t").query("label == 1").seq.tolist()
    al = make_aligner()
    rows = []
    for h, seed_seq in seeds:
        top, p0 = anneal(seed_seq, scorer, rng, a.steps, max_mut=a.max_mut, top_k=a.top_k)
        for s, f, p in top:
            d, nn_ = levenshtein_nearest(s, train_pos)
            ident, ident_hit = identity_nearest(s, train_pos, al)
            rows.append(dict(seed_id=h.split("|")[0], species=h.split("|")[-1].replace("_", " "), seed_seq=seed_seq,
                             seq=s, n_mut=sum(x != y for x, y in zip(s, seed_seq)), seed_scorer_p=p0, scorer_p=p,
                             fitness=f, charge=net_charge(s), hydrophobicity=hydrophobicity(s),
                             hydrophobic_moment=hydrophobic_moment(s), edit_dist_nearest_known=d,
                             pct_identity_nearest_known=round(100 * ident, 1),
                             nearest_known=nn_, nearest_by_identity=ident_hit))
        print(f"{h[:40]:40s} {seed_seq} -> {top[0][0]}  P {p0:.2f} -> {top[0][2]:.2f}" if top else h)
    df = pd.DataFrame(rows).sort_values("fitness", ascending=False).reset_index(drop=True)
    df.insert(0, "cand_id", [f"DeepAMP_{i+1:03d}" for i in range(len(df))])
    os.makedirs("outputs", exist_ok=True); os.makedirs("results", exist_ok=True)
    df.to_csv("outputs/candidates.csv", index=False, lineterminator="\n", encoding=ENC)
    write_fasta("outputs/candidates.fasta", [(f"{r.cand_id}|seed={r.seed_id}|{r.species}|scorerP={r.scorer_p:.3f}|"
                f"charge={r.charge:.1f}|uH={r.hydrophobic_moment:.2f}|dist={r.edit_dist_nearest_known}|"
                f"id={r.pct_identity_nearest_known}pct", r.seq)
                for r in df.itertuples()])
    print(f"wrote {len(df)} candidates -> outputs/candidates.csv, outputs/candidates.fasta")
