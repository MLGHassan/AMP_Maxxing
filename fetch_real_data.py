"""Download real peptide data from UniProt, CLEAN it, and build the negative sets.

Positives : reviewed (Swiss-Prot) peptides, 10-50 aa, annotated Antimicrobial (KW-0929)
            or Antibiotic (KW-0044).  NOTE: this is a *curated annotation*, not a
            guarantee of experimentally measured activity.  Extra experimentally
            validated sequences (DRAMP 3.0 / DBAASP v3 / APD3 FASTA downloads) can be
            added with --extra_pos_fasta; they go through the same cleaning.

Cleaning  : the databases mix entries a linear-sequence model cannot represent.
            Entries are DROPPED when the chemistry is not a linear chain of the 20
            standard L-amino acids (D-amino acids, head-to-tail cyclopeptides,
            thioether/sactipeptide cross-links), and FLAGGED but kept when the
            modification leaves the backbone linear (disulfides, C-terminal
            amidation, lipidation).  Every decision is counted in DATA_SUMMARY.json
            and listed per accession in data/processed/CLEANING_REPORT.tsv.

Negatives : three sets, so the scorer is tested against progressively harder controls.
            random  - length-matched random fragments of cytoplasmic proteins
                      (Veltri et al. 2018 recipe)
            hard    - the same, additionally matched on net charge at pH 7
            peptide - REAL reviewed peptides of the same length that are not
                      annotated antimicrobial (hormones, neuropeptides, toxins...).
                      This removes the "mature peptide vs internal fragment" cue
                      that the first two sets leave open.

Outputs   : data/processed/sequences.tsv, data/processed/plant_seeds.fasta,
            data/processed/CLEANING_REPORT.tsv, data/processed/DATA_SUMMARY.json,
            data/raw/*.fasta
Usage     : python fetch_real_data.py
"""
import argparse, json, math, os, random, re, time, urllib.parse, urllib.request
import pandas as pd
from common import ENC, MAX_LEN, ascii_safe, is_canonical, net_charge, read_fasta, write_fasta

BASE = "https://rest.uniprot.org/uniprotkb/search"
POS_Q = "(keyword:KW-0929 OR keyword:KW-0044) AND (reviewed:true) AND (length:[10 TO 50])"
PLANT = " AND (taxonomy_id:33090)"      # Viridiplantae
NEG_QS = [  # first one that the server accepts is used
    "(reviewed:true) AND (length:[100 TO 600]) AND (cc_scl_term:SL-0086) "
    "AND NOT (keyword:KW-0929) AND NOT (keyword:KW-0044) AND NOT (keyword:KW-0964)",
    "(reviewed:true) AND (length:[100 TO 600]) AND (locations:(location:cytoplasm)) "
    "AND NOT (keyword:KW-0929) AND NOT (keyword:KW-0044) AND NOT (keyword:KW-0964)",
]
# real non-antimicrobial peptides of the same size, for the third negative set
PEP_NEG_Q = ("(reviewed:true) AND (length:[10 TO 50]) "
             "AND NOT (keyword:KW-0929) AND NOT (keyword:KW-0044)")
# every reviewed plant peptide, used only to tag DRAMP entries as plant-derived
PLANT_ALL_Q = "(reviewed:true) AND (length:[10 TO 50]) AND (taxonomy_id:33090)"
ANN_FIELDS = "accession,keyword,ft_crosslnk,cc_ptm"
HDR = {"User-Agent": "DeepAMP-EthnoHack/2.0 (student project)"}

# --- cleaning rules ---------------------------------------------------------
# dropped: the molecule is not a linear chain of 20 standard L-amino acids, so the
#          sequence string genuinely misrepresents it
DROP_KEYWORDS = {
    "D-amino acid":  "d_amino_acid",      # KW-0208
    "Thioether bond": "thioether_bond",   # KW-0883 (sactipeptides)
}
DROP_CROSSLNK = re.compile(r"Cyclopeptide|head-to-tail|\(Gln-Lys\)|Isoaspartate", re.I)
# flagged but kept: backbone stays linear, and these cover most plant defensins
FLAG_KEYWORDS = {
    "Disulfide bond": "disulfide",        # KW-1015
    "Amidation":      "amidation",        # KW-0027
    "Lipoprotein":    "lipidation",
    "Lantibiotic":    "lantibiotic",
}


def _get(url, retries=4):
    for k in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=60)
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def _paged(params, max_records, count):
    """Yield (chunk, release), following UniProt's Link: rel=next cursor.

    `count(chunk)` says how many RECORDS a chunk holds, which differs between
    FASTA (multi-line records) and TSV (one line per record).
    """
    url = BASE + "?" + urllib.parse.urlencode(params)
    n, release = 0, None
    while url and n < max_records:
        r = _get(url)
        release = release or r.headers.get("X-UniProt-Release")
        chunk = r.read().decode(ENC)
        yield chunk, release
        n += count(chunk)
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        url = m.group(1) if m else None


def _n_fasta(chunk):
    return chunk.count("\n>") + (1 if chunk.startswith(">") else 0)


def uniprot_fasta(query, max_records, page=500):
    text, release = [], None
    for chunk, rel in _paged({"query": query, "format": "fasta", "size": page},
                             max_records, _n_fasta):
        text.append(chunk); release = release or rel
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


def uniprot_annotations(query, max_records, fields=ANN_FIELDS, page=500):
    """-> {accession: {column: value}} for the cleaning rules."""
    rows, header = [], None
    for chunk, _ in _paged({"query": query, "format": "tsv", "size": page,
                            "fields": fields}, max_records,
                           lambda c: max(c.count("\n") - 1, 0)):
        lines = chunk.splitlines()
        if lines and lines[0].startswith("Entry"):
            header = lines[0].split("\t"); lines = lines[1:]
        rows.extend(l.split("\t") for l in lines if l.strip())
    if not header:
        return {}
    out = {}
    for r in rows:
        r = r + [""] * (len(header) - len(r))
        out[r[0]] = dict(zip(header, r))
    return out


def parse_header(h):
    acc = h.split("|")[1] if "|" in h else h.split()[0]
    os_ = re.search(r"OS=(.*?) OX=", h)
    ox = re.search(r"OX=(\d+)", h)
    return acc, (os_.group(1) if os_ else ""), (int(ox.group(1)) if ox else -1)


def classify(acc, seq, ann):
    """-> (keep: bool, reason: str, flags: str) for one positive entry.

    Chemistry that breaks the linear-L-peptide assumption is dropped; modifications
    that leave the backbone linear are kept and recorded.
    """
    if not is_canonical(seq):
        bad = ascii_safe("".join(sorted(set(seq) - set("ACDEFGHIKLMNPQRSTVWY"))), 20)
        return False, ("length_out_of_range" if not bad else f"non_standard_residue:{bad}"), ""
    a = ann.get(acc, {})
    kws = {k.strip() for k in a.get("Keywords", "").split(";") if k.strip()}
    crosslnk, ptm = a.get("Cross-link", ""), a.get("Post-translational modification", "")
    for kw, reason in DROP_KEYWORDS.items():
        if kw in kws:
            return False, reason, ""
    if DROP_CROSSLNK.search(crosslnk) or DROP_CROSSLNK.search(ptm):
        return False, "cyclic_backbone", ""
    flags = sorted(tag for kw, tag in FLAG_KEYWORDS.items() if kw in kws)
    return True, "", ";".join(flags)


def random_fragment(prot, L, rng):
    i = rng.randint(0, len(prot) - L)
    return prot[i:i + L]


def build_dataset(pos_recs, neg_proteins, plant_accs, seed=0, pool_mult=60,
                  tol_steps=(1.0, 2.0, 3.0), annotations=None, peptide_negatives=None,
                  extra_positives=None):
    """Pure function (no network): returns (DataFrame, log, cleaning_report).

    extra_positives: already-cleaned records from external_db (DRAMP / DBAASP / APD3),
    appended after the UniProt entries so UniProt metadata wins on a duplicate sequence.
    """
    ann = annotations or {}
    rng = random.Random(seed)
    rows, seen, report = [], set(), []
    for h, s in pos_recs:
        s = s.upper()
        acc, org, tax = parse_header(h)
        keep, reason, flags = classify(acc, s, ann)
        if not keep:
            report.append(dict(id=acc, decision="dropped", reason=reason, flags="", seq=s))
            continue
        if s in seen:
            report.append(dict(id=acc, decision="dropped", reason="duplicate_sequence",
                               flags=flags, seq=s))
            continue
        seen.add(s)
        report.append(dict(id=acc, decision="kept", reason="", flags=flags, seq=s))
        rows.append(dict(id=acc, seq=s, label=1, sets="random,hard,peptide", source="uniprot_pos",
                         organism=org, taxid=tax, plant=int(acc in plant_accs), flags=flags))
    n_uniprot_pos = len(rows)
    n_extra_dup = 0
    for r in (extra_positives or []):
        s = r["seq"].upper()
        if s in seen:
            n_extra_dup += 1
            continue
        seen.add(s)
        rows.append(dict(id=r["id"], seq=s, label=1, sets="random,hard,peptide",
                         source=r.get("source", "external"), organism=r.get("organism", ""),
                         taxid=r.get("taxid", -1), plant=int(r.get("plant", 0)),
                         flags=r.get("flags", "")))
    pos = list(rows)
    prots = [s.upper() for _, s in neg_proteins if set(s.upper()) <= set("ACDEFGHIKLMNPQRSTVWY")]
    prots = [p for p in prots if len(p) >= 60]
    drop_counts = {}
    for r in report:
        if r["decision"] == "dropped":
            drop_counts[r["reason"].split(":")[0]] = drop_counts.get(r["reason"].split(":")[0], 0) + 1
    log = {"n_uniprot_pos_fetched": len(pos_recs),
           "n_uniprot_pos_kept": n_uniprot_pos,
           "n_uniprot_pos_dropped": len(pos_recs) - n_uniprot_pos,
           "uniprot_drop_reasons": drop_counts,
           "n_external_pos_added": len(pos) - n_uniprot_pos,
           "n_external_pos_duplicate_of_uniprot": n_extra_dup,
           "n_pos_kept": len(pos),
           "n_pos_flagged": sum(1 for r in rows if r["flags"]),
           "n_neg_proteins": len(prots)}

    # Each positive records which negative sets it actually found a partner in, so a
    # comparison never contains a positive whose counterpart is missing.
    pos_sets = [set() for _ in pos]

    # --- negative set 1: random length-matched fragments -------------------
    neg_rand = []
    for i, r in enumerate(pos):
        for _ in range(20):
            f = random_fragment(rng.choice(prots), len(r["seq"]), rng)
            if f not in seen:
                seen.add(f); neg_rand.append(f); pos_sets[i].add("random"); break

    # --- negative set 2: matched on length AND net charge ------------------
    # Bucketed by (length, integer charge) so matching stays linear in the number of
    # positives; a flat scan is O(positives x pool) and does not survive a merged
    # UniProt + DRAMP positive set.
    pool = {}
    for _ in range(pool_mult * len(pos)):
        L = len(rng.choice(pos)["seq"])
        f = random_fragment(rng.choice(prots), L, rng)
        if f in seen:
            continue
        c = net_charge(f)
        pool.setdefault((L, int(math.floor(c))), []).append((f, c))
    for v in pool.values():
        rng.shuffle(v)
    def _take(bucket, q, tol):
        """Pop one unused fragment within `tol` of charge q; put the rest back."""
        skipped, out = [], None
        while bucket:
            f, c = bucket.pop()
            if f in used or f in seen:
                continue                       # consumed earlier: drop it for good
            if abs(c - q) <= tol:
                out = f; break
            skipped.append((f, c))             # still useful at a wider tolerance
        bucket.extend(skipped)
        return out

    used, neg_hard, dropped = set(), [], 0
    for i, r in enumerate(pos):
        q, L = net_charge(r["seq"]), len(r["seq"])
        pick = None
        for tol in tol_steps:
            lo, hi = int(math.floor(q - tol)), int(math.floor(q + tol))
            for b in range(lo, hi + 1):
                bucket = pool.get((L, b))
                if bucket:
                    pick = _take(bucket, q, tol)
                    if pick:
                        break
            if pick:
                break
        if pick is None:
            dropped += 1
        else:
            used.add(pick); neg_hard.append(pick); pos_sets[i].add("hard")
    log.update(n_neg_random=len(neg_rand), n_neg_hard=len(neg_hard), n_pos_unmatched_in_hard=dropped)

    # --- negative set 3: real non-antimicrobial peptides, length matched ---
    neg_pep, pep_unmatched = [], 0
    if peptide_negatives:
        by_len, seen_pep = {}, set()
        for _, s in peptide_negatives:
            s = s.upper()
            # the download repeats sequences across entries; each may be used once
            if is_canonical(s) and s not in seen and s not in seen_pep:
                seen_pep.add(s)
                by_len.setdefault(len(s), []).append(s)
        for v in by_len.values():
            rng.shuffle(v)
        # EXACT length matching only. Reviewed non-AMP peptides skew much longer than
        # AMPs (median 31 vs 20), so allowing a tolerance leaves length itself able to
        # separate the classes -- with a +-10 window, length alone scored ROC-AUC 0.665.
        # Matching exactly costs positives (those at lengths where supply runs out drop
        # out of THIS comparison only) and buys a set where length carries no signal.
        idx_by_len = {}
        for i, r in enumerate(pos):
            idx_by_len.setdefault(len(r["seq"]), []).append(i)
        for L, idxs in idx_by_len.items():
            bucket = by_len.get(L, [])
            n = min(len(idxs), len(bucket))
            for j in range(n):
                neg_pep.append(bucket[j]); pos_sets[idxs[j]].add("peptide")
            pep_unmatched += len(idxs) - n
        log.update(peptide_len_matching="exact")
    log.update(n_neg_peptide=len(neg_pep), n_pos_unmatched_in_peptide=pep_unmatched,
               n_pos_in_peptide_set=sum(1 for s in pos_sets if "peptide" in s))

    # record, on each positive, which comparisons it is actually part of
    for i, r in enumerate(pos):
        r["sets"] = ",".join(sorted(pos_sets[i]))

    by_seq = {}
    for name, lst in (("random", neg_rand), ("hard", neg_hard), ("peptide", neg_pep)):
        for f in lst:
            by_seq[f] = (by_seq.get(f, "") + "," + name).lstrip(",")
    for i, (f, st) in enumerate(by_seq.items()):
        src = "uniprot_peptide_nonamp" if st == "peptide" else "uniprot_cyto_fragment"
        rows.append(dict(id=f"neg{i:05d}", seq=f, label=0, sets=st, source=src,
                         organism="", taxid=-1, plant=0, flags=""))
    return pd.DataFrame(rows), log, pd.DataFrame(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pos", type=int, default=6000)
    ap.add_argument("--max_neg_proteins", type=int, default=4000)
    ap.add_argument("--max_pep_neg", type=int, default=14000,
                    help="size of the real-peptide negative pool to draw from; must "
                         "comfortably exceed the positive count after merging DRAMP")
    ap.add_argument("--extra_pos_fasta", default=None,
                    help="DBAASP v3 / APD3 FASTA download (added as positives; no "
                         "machine-readable chemistry, so only alphabet/length rules apply)")
    ap.add_argument("--dramp", default="auto",
                    help="'auto' downloads DRAMP 3.0 into data/raw/; or a path to "
                         "general_amps.txt; or 'none' to use Swiss-Prot only")
    ap.add_argument("--dramp_strict", action="store_true",
                    help="also drop DRAMP entries whose chemistry is uncurated")
    ap.add_argument("--no_peptide_negatives", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("data/raw", exist_ok=True); os.makedirs("data/processed", exist_ok=True)

    print("Fetching positives ...")
    pos, rel = uniprot_fasta(POS_Q, a.max_pos)
    write_fasta("data/raw/pos_uniprot.fasta", pos)
    print(f"  {len(pos)} records (UniProt release {rel})")
    print("Fetching positive annotations for the cleaning step ...")
    ann = uniprot_annotations(POS_Q, a.max_pos)
    print(f"  annotations for {len(ann)} accessions")
    plant, _ = uniprot_fasta(POS_Q + PLANT, a.max_pos)
    plant_accs = {parse_header(h)[0] for h, _ in plant}
    print(f"  {len(plant_accs)} plant records")
    # DRAMP's Source column is free text, so plant origin for DRAMP entries is resolved
    # through their Swiss-Prot cross-reference against all reviewed Viridiplantae peptides.
    all_plant_accs = set(plant_accs)
    if a.dramp != "none":
        all_plant_accs |= set(uniprot_annotations(PLANT_ALL_Q, 20000, fields="accession"))
        print(f"  {len(all_plant_accs)} reviewed plant accessions (for tagging DRAMP entries)")

    neg, used_q = None, None
    for q in NEG_QS:
        try:
            neg, _ = uniprot_fasta(q, a.max_neg_proteins); used_q = q; break
        except Exception as e:
            print("  negative query failed, trying next:", e)
    assert neg, "could not fetch negative proteins"
    write_fasta("data/raw/neg_proteins.fasta", neg)
    print(f"  {len(neg)} cytoplasmic proteins")

    pep_neg = None
    if not a.no_peptide_negatives:
        print("Fetching real non-antimicrobial peptides (third negative set) ...")
        pep_neg, _ = uniprot_fasta(PEP_NEG_Q, a.max_pep_neg)
        write_fasta("data/raw/neg_peptides.fasta", pep_neg)
        print(f"  {len(pep_neg)} non-AMP peptides")

    # --- external databases the proposal names -----------------------------
    extra_pos, extra_log, extra_report = [], {}, []
    if a.dramp != "none":
        from external_db import download_dramp, load_dramp
        try:
            dpath = a.dramp if a.dramp != "auto" else download_dramp()
            recs, dlog, drep = load_dramp(dpath, plant_accs=all_plant_accs,
                                          strict=a.dramp_strict)
            extra_pos += recs; extra_log.update(dlog)
            extra_report += [dict(r, db="dramp3") for r in drep]
            print(f"  DRAMP 3.0: {dlog['dramp_kept']} kept of {dlog['dramp_rows']} "
                  f"({dlog['dramp_plant']} plant); dropped {dlog['dramp_drop_reasons']}")
        except Exception as e:
            print(f"  WARNING: DRAMP unavailable ({type(e).__name__}: {e}).")
            print("  Continuing with Swiss-Prot only. Rerun later, or pass --dramp <path>.")
            extra_log["dramp_error"] = f"{type(e).__name__}: {e}"

    if a.extra_pos_fasta:
        from external_db import load_fasta_positives
        recs, elog, erep = load_fasta_positives(a.extra_pos_fasta, tag="extra_fasta")
        extra_pos += recs; extra_log.update(elog)
        extra_report += [dict(r, db="extra_fasta") for r in erep]
        print(f"  + {len(recs)} extra positives from {a.extra_pos_fasta}")

    df, log, report = build_dataset(pos, neg, plant_accs, seed=a.seed,
                                    annotations=ann, peptide_negatives=pep_neg,
                                    extra_positives=extra_pos)
    log.update(extra_log)
    if extra_report:
        report = pd.concat([report.assign(db="uniprot"), pd.DataFrame(extra_report)],
                           ignore_index=True)
    else:
        report = report.assign(db="uniprot")
    df.to_csv("data/processed/sequences.tsv", sep="\t", index=False,
              lineterminator="\n", encoding=ENC)
    report.to_csv("data/processed/CLEANING_REPORT.tsv", sep="\t", index=False,
                  lineterminator="\n", encoding=ENC)
    seeds = [(f"{r.id}|{r.organism.replace(' ', '_')}", r.seq) for r in df[df.plant == 1].itertuples()]
    write_fasta("data/processed/plant_seeds.fasta", seeds)
    log.update(uniprot_release=rel, date=time.strftime("%Y-%m-%d"), pos_query=POS_Q,
               neg_query=used_q, pep_neg_query=None if a.no_peptide_negatives else PEP_NEG_Q,
               n_plant_seeds=len(seeds), n_positives_total=int((df.label == 1).sum()),
               positives_by_source={k: int(v) for k, v in
                                    df[df.label == 1].source.value_counts().items()},
               license="UniProt: CC BY 4.0" +
                       ("; DRAMP 3.0: free for academic use, not redistributed here"
                        if a.dramp != "none" else ""))
    with open("data/processed/DATA_SUMMARY.json", "w", encoding=ENC, newline="\n") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))
