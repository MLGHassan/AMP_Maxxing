"""End-to-end smoke test of the whole pipeline on tiny STAND-IN data with a stub embedder.

No network, no ESM download; runs in a couple of minutes. It checks that every stage's
code path works -- the numbers it prints mean nothing.

Covered: cleaning rules, dataset build (all three negative sets), clustering, cluster
grouping, descriptor baselines, all four model rungs, the leakage control, the judge,
simulated annealing, the candidate audit, and the summary tables and figures.

Run:  python tests/smoke_test.py
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
        H = torch.zeros(len(seqs), 52, 320, dtype=torch.float16)
        M = torch.zeros(len(seqs), 52, dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids = [0] + [AA2ID[a] for a in s] + [0]
            H[i, :len(ids)] = self.W[ids].half(); M[i, :len(ids)] = True
        return H, M


import esm_embed
esm_embed.EsmEmbedder = StubEmbedder


def run(script, *args):
    print(f"\n----- {script} {' '.join(args)} -----", flush=True)
    sys.argv = [script, *args]
    runpy.run_path(os.path.join(ROOT, script), run_name="__main__")


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        raise SystemExit(f"SMOKE TEST FAILED: {msg}")


if __name__ == "__main__":
    from common import write_fasta
    from fetch_real_data import build_dataset, classify
    from cluster_and_split import greedy_cluster, assign_groups

    # --- the cleaning rules, which decide what even enters the dataset ------
    print("----- cleaning rules -----")
    keep, reason, _ = classify("A1", "KWKLFKKIEKVGQNIR", {})
    check(keep, "a plain linear peptide is kept")
    keep, reason, _ = classify("A2", "KWKLFKKIEKVGQNIR",
                               {"A2": {"Keywords": "Antimicrobial;D-amino acid"}})
    check(not keep and reason == "d_amino_acid", "a D-amino acid entry is dropped")
    keep, reason, _ = classify("A3", "KWKLFKKIEKVGQNIR",
                               {"A3": {"Cross-link": 'CROSSLNK 1..16; /note="Cyclopeptide (Asn-Gly)"'}})
    check(not keep and reason == "cyclic_backbone", "a head-to-tail cyclopeptide is dropped")
    keep, _, flags = classify("A4", "KWKLFKKIEKVGQNIR",
                              {"A4": {"Keywords": "Antimicrobial;Disulfide bond;Amidation"}})
    check(keep and flags == "amidation;disulfide", "a disulfide/amidated entry is kept and flagged")
    keep, reason, _ = classify("A5", "KWKLBKKIEKVGQNIR", {})
    check(not keep and reason.startswith("non_standard_residue"), "a non-standard residue is dropped")
    keep, reason, _ = classify("A6", "KWKLF", {})
    check(not keep and reason == "length_out_of_range", "a too-short peptide is dropped")

    wd = os.environ.get("SMOKE_DIR") or tempfile.mkdtemp()
    os.makedirs(wd, exist_ok=True); os.chdir(wd)
    print(f"\nworking in {wd}")
    for d in ["data/processed", "outputs", "results", "figures", "checkpoints"]:
        os.makedirs(d, exist_ok=True)

    rng = random.Random(0); AA = "ACDEFGHIKLMNPQRSTVWY"
    pos = [(f"sp|P{i:05d}|X Desc OS=Test species OX=1",
            "".join(rng.choice("KKRRLLAIVFG" + AA) for _ in range(rng.randint(12, 40))))
           for i in range(300)]
    # two entries that the cleaning step must throw out
    ann = {"P00001": {"Keywords": "Antimicrobial;D-amino acid"},
           "P00002": {"Cross-link": 'CROSSLNK 1..20; /note="Cyclopeptide (Asn-Gly)"'}}
    prots = [(f"p{i}", "".join(rng.choice(AA) for _ in range(rng.randint(100, 400))))
             for i in range(200)]
    pep_neg = [(f"q{i}", "".join(rng.choice("DDEESSTTGGNQ" + AA) for _ in range(rng.randint(12, 40))))
               for i in range(400)]

    # a stand-in "external database" record, to exercise the DRAMP merge path
    extra = [dict(id="DRAMPX0001", seq="KWKLFKKIGAVLKVLTTGLPALIS", organism="Test plant",
                  taxid=-1, plant=1, flags="chemistry_unverified", source="dramp3"),
             dict(id="DRAMPX0002", seq=pos[5][1], organism="dup of a UniProt entry",
                  taxid=-1, plant=0, flags="", source="dramp3")]
    df, log, report = build_dataset(pos, prots, {f"P{i:05d}" for i in range(30)},
                                    pool_mult=30, annotations=ann, peptide_negatives=pep_neg,
                                    extra_positives=extra)
    check(log["n_uniprot_pos_dropped"] >= 2,
          f"cleaning dropped the 2 planted bad entries ({log['uniprot_drop_reasons']})")
    check(log["n_external_pos_added"] == 1,
          f"the external record was merged ({log['n_external_pos_added']})")
    check(log["n_external_pos_duplicate_of_uniprot"] == 1,
          "an external sequence already in UniProt was deduplicated")
    check((df.source == "dramp3").sum() == 1, "the external record is tagged with its database")
    check(log["n_neg_peptide"] > 0, f"peptide negative set built ({log['n_neg_peptide']} rows)")
    check(len(report) == len(pos), "cleaning report has one row per fetched UniProt positive")
    for s in ["random", "hard", "peptide"]:
        check(df.sets.str.contains(s).any(), f"'{s}' negative set present in sequences.tsv")

    df["cluster"] = greedy_cluster(df.seq.tolist(), 0.4)
    df["group"] = assign_groups(df)
    df.to_csv("data/processed/dataset.tsv", sep="\t", index=False, lineterminator="\n",
              encoding="utf-8")
    write_fasta("data/processed/plant_seeds.fasta",
                [(f"{r.id}|Test_species", r.seq) for r in df[df.plant == 1].itertuples()])
    report.to_csv("data/processed/CLEANING_REPORT.tsv", sep="\t", index=False,
                  lineterminator="\n", encoding="utf-8")

    run("baselines.py", "--folds", "2", "--negatives", "random", "hard", "peptide")
    for role in ["scorer", "judge"]:
        run("train.py", "--role", role, "--model", "esm_attn", "--neg", "hard",
            "--folds", "2", "--epochs", "12")
    run("train.py", "--role", "scorer", "--model", "esm_attn1h", "--neg", "hard",
        "--folds", "2", "--epochs", "12")
    run("train.py", "--role", "scorer", "--model", "esm_mean", "--neg", "random",
        "--folds", "2", "--epochs", "12")
    run("train.py", "--role", "scorer", "--model", "cnn_lstm", "--neg", "peptide",
        "--folds", "2", "--epochs", "8")
    run("train.py", "--role", "judge", "--model", "cnn_lstm", "--neg", "hard",
        "--folds", "2", "--epochs", "8")
    run("train.py", "--role", "scorer", "--model", "esm_attn", "--neg", "hard",
        "--folds", "2", "--epochs", "12", "--mode", "random")
    run("train.py", "--role", "scorer", "--model", "esm_attn", "--neg", "hard",
        "--folds", "2", "--epochs", "12", "--protocol", "holdout", "--seeds", "2")
    run("search.py", "--scorer", "checkpoints/scorer_esm_attn_hard_s0.pt",
        "--n_seeds", "4", "--steps", "60")
    run("evaluate_candidates.py", "--n_controls", "3")
    run("evaluate_candidates.py", "--judge_glob", "checkpoints/judge_cnn_lstm_hard_s*.pt",
        "--tag", "_cnn")
    run("summarize.py")

    print("\n----- output files -----")
    import pandas as pd
    for p in ["results/baselines.json", "results/baselines.csv", "results/summary.md",
              "results/scorer_esm_attn_hard_cluster.json",
              "results/scorer_esm_attn_hard_random.json",
              "results/candidate_audit.csv", "results/candidate_audit_cnn.csv",
              "outputs/candidates.csv", "outputs/candidates.fasta",
              "outputs/candidate_library_ranked.fasta",
              "figures/ladder_mcc.png", "figures/candidate_audit.png"]:
        check(os.path.exists(p) and os.path.getsize(p) > 0, f"wrote {p}")
    c = pd.read_csv("results/candidate_audit.csv")
    for col in ["judge_p", "seed_judge_p", "random_mutant_judge_p",
                "edit_dist_nearest_known", "pct_identity_nearest_known", "charge",
                "hydrophobic_moment"]:
        check(col in c.columns, f"candidate audit has column '{col}'")
    check(c.pct_identity_nearest_known.between(0, 100).all(), "identity values are percentages")
    summ = open("results/summary.md", encoding="utf-8").read()
    for want in ["sensitivity", "specificity", "CV MCC", "Leakage check", "External reference"]:
        check(want in summ, f"summary.md reports '{want}'")

    print("\nSMOKE TEST PASSED")
