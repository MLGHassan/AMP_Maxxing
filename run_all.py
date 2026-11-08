"""Run the complete DeepAMP pipeline. Works on Windows, macOS and Linux.

    python run_all.py              # full run, skipping steps whose outputs exist
    python run_all.py --quick      # ~15 min smaller run, for checking the install
    python run_all.py --force      # redo every step
    python run_all.py --from 4     # start at step 4
    python run_all.py --list       # just print the steps

Every step is this interpreter running one of the scripts, so there is no shell
dependency and the three operating systems execute exactly the same commands.
"""
import argparse, os, subprocess, sys, time

PY = sys.executable


def steps(a):
    """-> list of (number, name, argv, output_that_proves_it_ran)."""
    f, s = a.folds, a.n_seeds
    neg_sets = ["random", "hard"] if a.quick else ["random", "hard", "peptide"]
    models = ["cnn_lstm", "esm_mean", "esm_attn"]
    out = [
        (1, "download + clean UniProt data", ["fetch_real_data.py"],
         "data/processed/sequences.tsv"),
        (2, "cluster at 40% identity and split by cluster", ["cluster_and_split.py"],
         "data/processed/dataset.tsv"),
        (3, "rung 1: descriptor baselines", ["baselines.py", "--folds", str(f),
                                             "--negatives", *neg_sets],
         "results/baselines.json"),
    ]
    n = 4
    for neg in neg_sets:
        for m in models:
            out.append((n, f"rungs 2-4: scorer {m} on {neg} negatives",
                        ["train.py", "--role", "scorer", "--model", m, "--neg", neg,
                         "--folds", str(f), "--epochs", str(a.epochs)],
                        f"results/scorer_{m}_{neg}_cluster.json"))
            n += 1
    if not a.quick:
        out.append((n, "ablation: single-head attention pooling",
                    ["train.py", "--role", "scorer", "--model", "esm_attn1h", "--neg", "hard",
                     "--folds", str(f), "--epochs", str(a.epochs)],
                    "results/scorer_esm_attn1h_hard_cluster.json")); n += 1
    out += [
        (n, "leakage check: same model, cluster-blind random split",
         ["train.py", "--role", "scorer", "--model", "esm_attn", "--neg", "hard",
          "--mode", "random", "--folds", str(f), "--epochs", str(a.epochs)],
         "results/scorer_esm_attn_hard_random.json"),
        (n + 1, "independent judge 1 (ESM multi-head attention)",
         ["train.py", "--role", "judge", "--model", "esm_attn", "--neg", "hard",
          "--folds", str(f), "--epochs", str(a.epochs)],
         "results/judge_esm_attn_hard_cluster.json"),
        (n + 2, "independent judge 2 (CNN-BiLSTM, different inductive bias)",
         ["train.py", "--role", "judge", "--model", "cnn_lstm", "--neg", "hard",
          "--folds", str(f), "--epochs", str(a.epochs)],
         "results/judge_cnn_lstm_hard_cluster.json"),
        (n + 3, "simulated annealing from plant seeds",
         ["search.py", "--scorer", "checkpoints/scorer_esm_attn_hard_s0.pt",
          "--n_seeds", str(s)],
         "outputs/candidates.csv"),
        (n + 4, "audit candidates with judge 1",
         ["evaluate_candidates.py", "--judge_glob", "checkpoints/judge_esm_attn_hard_s*.pt"],
         "results/candidate_audit.csv"),
        (n + 5, "audit candidates with judge 2",
         ["evaluate_candidates.py", "--judge_glob", "checkpoints/judge_cnn_lstm_hard_s*.pt",
          "--tag", "_cnn"],
         "results/candidate_audit_cnn.csv"),
        (n + 6, "tables and figures", ["summarize.py"], "results/summary.md"),
        (n + 7, "verify the run (integrity + ladder + search claims)",
         [os.path.join("tests", "verify_results.py")], None),
    ]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="2 folds, fewer epochs, fewer search seeds, 2 negative sets")
    ap.add_argument("--force", action="store_true", help="rerun steps whose outputs exist")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n_seeds", type=int, default=None)
    a = ap.parse_args()
    a.folds = a.folds if a.folds is not None else (2 if a.quick else 5)
    a.epochs = a.epochs if a.epochs is not None else (20 if a.quick else 80)
    a.n_seeds = a.n_seeds if a.n_seeds is not None else (8 if a.quick else 40)

    for d in ["data/raw", "data/processed", "checkpoints", "results", "outputs", "figures", "cache"]:
        os.makedirs(d, exist_ok=True)

    plan = steps(a)
    if a.list:
        for i, name, argv, _ in plan:
            print(f"{i:3d}. {name}\n       {PY} {' '.join(argv)}")
        raise SystemExit(0)

    print(f"DeepAMP pipeline: {len(plan)} steps, "
          f"{'QUICK' if a.quick else 'full'} settings, interpreter {PY}")
    t_all = time.time()
    for i, name, argv, proof in plan:
        if i < a.start:
            print(f"[{i}/{len(plan)}] skip (before --from): {name}"); continue
        if proof and os.path.exists(proof) and not a.force:
            print(f"[{i}/{len(plan)}] skip (already have {proof}): {name}"); continue
        print(f"\n[{i}/{len(plan)}] {name}\n       {' '.join(argv)}", flush=True)
        t = time.time()
        r = subprocess.run([PY, *argv])
        if r.returncode != 0:
            print(f"\nSTEP {i} FAILED ({name}) with exit code {r.returncode}.\n"
                  f"Fix it and rerun:  python run_all.py --from {i}")
            raise SystemExit(r.returncode)
        print(f"       done in {time.time() - t:.0f}s", flush=True)
    print(f"\nPIPELINE COMPLETE in {(time.time() - t_all) / 60:.1f} min. "
          f"See results/summary.md and figures/.")
