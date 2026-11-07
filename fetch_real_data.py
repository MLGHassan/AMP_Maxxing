"""Download real peptide data from UniProt and build positive / negative sets.

Positives : reviewed (Swiss-Prot) peptides, 10-50 aa, annotated Antimicrobial (KW-0929)
            or Antibiotic (KW-0044).  NOTE: this is a *curated annotation*, not a
            guarantee of experimentally measured activity.  Extra experimentally
            validated sequences (e.g. a DRAMP / DBAASP FASTA download) can be added
            with --extra_pos_fasta.
Negatives : length-matched random fragments cut from reviewed cytoplasmic proteins
            that are not annotated antimicrobial or secreted (Veltri et al. 2018 recipe).
            Second set additionally matched on length AND net charge (+-tol).

Outputs   : data/processed/sequences.tsv, data/processed/plant_seeds.fasta,
            data/raw/*.fasta, data/processed/DATA_SUMMARY.json
Usage     : python fetch_real_data.py
"""
import argparse, json, os, random, re, time, urllib.parse, urllib.request
import pandas as pd
from common import is_canonical, net_charge, read_fasta, write_fasta

BASE = "https://rest.uniprot.org/uniprotkb/search"
POS_Q = "(keyword:KW-0929 OR keyword:KW-0044) AND (reviewed:true) AND (length:[10 TO 50])"
PLANT = " AND (taxonomy_id:33090)"      # Viridiplantae
NEG_QS = [  # first one that the server accepts is used
    "(reviewed:true) AND (length:[100 TO 600]) AND (cc_scl_term:SL-0086) "
    "AND NOT (keyword:KW-0929) AND NOT (keyword:KW-0044) AND NOT (keyword:KW-0964)",
    "(reviewed:true) AND (length:[100 TO 600]) AND (locations:(location:cytoplasm)) "
    "AND NOT (keyword:KW-0929) AND NOT (keyword:KW-0044) AND NOT (keyword:KW-0964)",
]
HDR = {"User-Agent": "DeepAMP-EthnoHack/2.0 (student project)"}


def _get(url, retries=4):
    for k in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=60)
        except Exception as e:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def uniprot_fasta(query, max_records, page=500):
    url = BASE + "?" + urllib.parse.urlencode({"query": query, "format": "fasta", "size": page})
    text, release = [], None
    n = 0
    while url and n < max_records:
        r = _get(url)
        release = release or r.headers.get("X-UniProt-Release")
        chunk = r.read().decode()
        text.append(chunk)
        n += chunk.count("\n>") + (1 if chunk.startswith(">") else 0)
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        url = m.group(1) if m else None
    recs, name, buf = [], None, []
    for line in "".join(text).splitlines():
        if line.startswith(">"):
            if name:
                recs.append((name, "".join(buf)))
            name, buf = line[1:], []
        else:
            buf.append(line.strip())
    if name:
        recs.append((name, "".join(buf)))
    return recs[:max_records], release


def parse_header(h):
    acc = h.split("|")[1] if "|" in h else h.split()[0]
    os_ = re.search(r"OS=(.*?) OX=", h)
    ox = re.search(r"OX=(\d+)", h)
    return acc, (os_.group(1) if os_ else ""), (int(ox.group(1)) if ox else -1)


def random_fragment(prot, L, rng):
    i = rng.randint(0, len(prot) - L)
    return prot[i:i + L]


def build_dataset(pos_recs, neg_proteins, plant_accs, seed=0, pool_mult=60, tol_steps=(1.0, 2.0, 3.0)):
    """Pure function (no network): returns DataFrame of sequences + a log dict."""
    rng = random.Random(seed)
    rows, seen = [], set()
    for h, s in pos_recs:
        s = s.upper()
        if not is_canonical(s) or s in seen:
            continue
        seen.add(s)
        acc, org, tax = parse_header(h)
        rows.append(dict(id=acc, seq=s, label=1, sets="random,hard", source="uniprot_pos",
                         organism=org, taxid=tax, plant=int(acc in plant_accs)))
    pos = [r for r in rows]
    prots = [s.upper() for _, s in neg_proteins if set(s.upper()) <= set("ACDEFGHIKLMNPQRSTVWY")]
    prots = [p for p in prots if len(p) >= 60]
    log = {"n_pos_unique": len(pos), "n_neg_proteins": len(prots)}

    # --- negative set 1: random length-matched fragments -------------------
    neg_rand = []
    for r in pos:
        for _ in range(20):
            f = random_fragment(rng.choice(prots), len(r["seq"]), rng)
            if f not in seen:
                seen.add(f); neg_rand.append(f); break

    # --- negative set 2: matched on length AND net charge ------------------
    pool = {}
    for _ in range(pool_mult * len(pos)):
        L = len(rng.choice(pos)["seq"])
        f = random_fragment(rng.choice(prots), L, rng)
        pool.setdefault(L, []).append((f, net_charge(f)))
    used, neg_hard, dropped = set(), [], 0
    for r in pos:
        q, L = net_charge(r["seq"]), len(r["seq"])
        pick = None
        for tol in tol_steps:
            cands = [f for f, c in pool.get(L, []) if abs(c - q) <= tol and f not in used and f not in seen]
            if cands:
                pick = rng.choice(cands); break
        if pick is None:
            dropped += 1
        else:
            used.add(pick); neg_hard.append(pick)
    log.update(n_neg_random=len(neg_rand), n_neg_hard=len(neg_hard), n_pos_unmatched_in_hard=dropped)

    by_seq = {}
    for f in neg_rand:
        by_seq[f] = "random"
    for f in neg_hard:
        by_seq[f] = "random,hard" if f in by_seq else "hard"
    for i, (f, st) in enumerate(by_seq.items()):
        rows.append(dict(id=f"neg{i:05d}", seq=f, label=0, sets=st, source="uniprot_cyto_fragment",
                         organism="", taxid=-1, plant=0))
    return pd.DataFrame(rows), log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pos", type=int, default=6000)
    ap.add_argument("--max_neg_proteins", type=int, default=4000)
    ap.add_argument("--extra_pos_fasta", default=None, help="e.g. DRAMP/DBAASP FASTA (label as positives)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("data/raw", exist_ok=True); os.makedirs("data/processed", exist_ok=True)

    print("Fetching positives ...")
    pos, rel = uniprot_fasta(POS_Q, a.max_pos)
    write_fasta("data/raw/pos_uniprot.fasta", pos)
    print(f"  {len(pos)} records (UniProt release {rel})")
    plant, _ = uniprot_fasta(POS_Q + PLANT, a.max_pos)
    plant_accs = {parse_header(h)[0] for h, _ in plant}
    print(f"  {len(plant_accs)} plant records")

    neg, used_q = None, None
    for q in NEG_QS:
        try:
            neg, _ = uniprot_fasta(q, a.max_neg_proteins); used_q = q; break
        except Exception as e:
            print("  negative query failed, trying next:", e)
    assert neg, "could not fetch negative proteins"
    write_fasta("data/raw/neg_proteins.fasta", neg)
    print(f"  {len(neg)} cytoplasmic proteins")

    if a.extra_pos_fasta:
        extra = [(f"extra|{h.split()[0]}|OS= OX=-1", s) for h, s in read_fasta(a.extra_pos_fasta)]
        pos += extra
        print(f"  + {len(extra)} extra positives from {a.extra_pos_fasta}")

    df, log = build_dataset(pos, neg, plant_accs, seed=a.seed)
    df.to_csv("data/processed/sequences.tsv", sep="\t", index=False)
    seeds = [(f"{r.id}|{r.organism.replace(' ', '_')}", r.seq) for r in df[df.plant == 1].itertuples()]
    write_fasta("data/processed/plant_seeds.fasta", seeds)
    log.update(uniprot_release=rel, date=time.strftime("%Y-%m-%d"), pos_query=POS_Q,
               neg_query=used_q, n_plant_seeds=len(seeds), license="UniProt: CC BY 4.0")
    json.dump(log, open("data/processed/DATA_SUMMARY.json", "w"), indent=2)
    print(json.dumps(log, indent=2))
