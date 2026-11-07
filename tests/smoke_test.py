"""End-to-end smoke test of the whole pipeline on tiny STAND-IN data with a stub embedder.
No network, no ESM download; runs in about a minute.  It checks that the code paths work -- the numbers
it prints mean nothing.   Run:  python tests/smoke_test.py
"""
import os, random, runpy, sys, tempfile
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class StubEmbedder:
    """Fixed random vector per amino acid (+ position signal) in place of ESM-2."""
    def __init__(self, device=None, **kw):
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(21, 320, generator=g); self.dim = 320; self.cache = {}
    def embed(self, seqs, batch=64):
        from common import AA2ID
        H = torch.zeros(len(seqs), 52, 320, dtype=torch.float16); M = torch.zeros(len(seqs), 52, dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids = [0] + [AA2ID[a] for a in s] + [0]
            H[i, :len(ids)] = self.W[ids].half(); M[i, :len(ids)] = True
        return H, M


import esm_embed
esm_embed.EsmEmbedder = StubEmbedder


def run(script, *args):
    sys.argv = [script, *args]
    runpy.run_path(os.path.join(ROOT, script), run_name="__main__")


if __name__ == "__main__":
    from common import write_fasta
    from fetch_real_data import build_dataset
    from cluster_and_split import greedy_cluster, assign_groups
    wd = os.environ.get("SMOKE_DIR") or tempfile.mkdtemp(); os.makedirs(wd, exist_ok=True); os.chdir(wd)
    for d in ["data/processed", "outputs", "results"]:
        os.makedirs(d, exist_ok=True)
    rng = random.Random(0); AA = "ACDEFGHIKLMNPQRSTVWY"
    pos = [(f"sp|P{i:05d}|X Desc OS=Test species OX=1", "".join(rng.choice("KKRRLLAIVFG" + AA) for _ in range(rng.randint(12, 40)))) for i in range(300)]
    prots = [(f"p{i}", "".join(rng.choice(AA) for _ in range(rng.randint(100, 400)))) for i in range(200)]
    df, _ = build_dataset(pos, prots, {f"P{i:05d}" for i in range(30)}, pool_mult=30)
    df["cluster"] = greedy_cluster(df.seq.tolist(), 0.4); df["group"] = assign_groups(df)
    df.to_csv("data/processed/dataset.tsv", sep="\t", index=False)
    write_fasta("data/processed/plant_seeds.fasta", [(f"{r.id}|Test_species", r.seq) for r in df[df.plant == 1].itertuples()])
    for role in ["scorer", "judge"]:
        run("train.py", "--role", role, "--model", "esm_attn", "--neg", "hard", "--seeds", "2", "--epochs", "15")
    run("train.py", "--role", "scorer", "--model", "esm_mean", "--neg", "random", "--seeds", "2", "--epochs", "15")
    run("train.py", "--role", "scorer", "--model", "cnn_lstm", "--neg", "random", "--seeds", "2", "--epochs", "10")
    run("train.py", "--role", "scorer", "--model", "esm_attn", "--neg", "hard", "--seeds", "2", "--epochs", "15", "--mode", "random")
    run("search.py", "--scorer", "checkpoints/scorer_esm_attn_hard_s0.pt", "--n_seeds", "4", "--steps", "60")
    run("evaluate_candidates.py", "--n_controls", "3")
    print("\nSMOKE TEST PASSED")
