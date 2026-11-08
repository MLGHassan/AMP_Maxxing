"""Load extra validated AMPs from external databases (DRAMP 3.0, and generic FASTA).

The proposal names DRAMP 3.0 and DBAASP v3 as positive sources. DRAMP publishes a
tab-separated table with the annotation columns we need to apply the SAME chemistry
cleaning we apply to Swiss-Prot, so it is wired up end to end here. DBAASP and APD3
exports can be passed through `load_fasta_positives` instead; they carry no
machine-readable chemistry, so only the alphabet/length rules apply to them.

Licence note: DRAMP is free for academic use but is NOT redistributed in this
repository. It is downloaded at run time into data/raw/, which is gitignored.

Cleaning mirrors fetch_real_data.classify:
  dropped - the sequence string misrepresents the molecule (D-amino acids, cyclic or
            branched backbones, non-peptidic labels such as FITC or a fatty acid)
  flagged - the backbone stays linear (amidation, acetylation, disulfides), kept
  unverified - DRAMP has not curated the chemistry ("Not included yet"); kept and
            flagged, or dropped with strict=True
"""
import os
import re
import urllib.request

from common import ENC, ascii_safe, is_canonical

DRAMP_BASE = ("http://dramp.cpu-bioinfor.org/downloads/download.php"
              "?filename=download_data/DRAMP3.0_new/")
DRAMP_GENERAL = DRAMP_BASE + "general_amps.txt"
HDR = {"User-Agent": "DeepAMP-EthnoHack/2.0 (student project)"}

#: values of Linear/Cyclic/Branched that are not a single linear chain
_NON_LINEAR = {"cyclic", "branched", "tetrameric", "dimer", "dimeric", "trimeric"}
#: anything in a terminal-modification field that is not an ordinary peptide terminus
_LABEL_RE = re.compile(
    r"FITC|rhodamine|biotin|PEG|fluorescein|dansyl|TAMRA|Ahx|fatty acid|"
    r"palmit|myrist|octanoyl|decanoyl|lauric|stearoyl|DOTA|chloro|Cy5|Cy3|NBD",
    re.I)
_CYCLIZE_RE = re.compile(r"cycliz|cyclis", re.I)
#: modifications that leave a linear backbone -> keep, but record
_FLAGS = [("amidation", re.compile(r"amidation", re.I)),
          ("acetylation", re.compile(r"acetylation", re.I)),
          ("disulfide", re.compile(r"disulfide", re.I))]
_UNKNOWN = {"", "not included yet", "not found", "n/a", "na", "-", "none"}


def download_dramp(dest_dir="data/raw", url=DRAMP_GENERAL, filename="dramp_general_amps.txt",
                   timeout=180, force=False):
    """Download DRAMP's general AMP table. -> local path, or raises."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, filename)
    if os.path.exists(path) and os.path.getsize(path) > 100_000 and not force:
        print(f"  using cached {path} ({os.path.getsize(path) // 1024} KB)")
        return path
    print(f"  downloading DRAMP 3.0 from {url.split('?')[0]} ...")
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if len(data) < 100_000:
        raise RuntimeError(f"DRAMP download looks truncated ({len(data)} bytes)")
    with open(path, "wb") as f:
        f.write(data)
    print(f"  saved {path} ({len(data) // 1024} KB)")
    return path


def _unknown(v):
    return (v or "").strip().lower() in _UNKNOWN


def classify_dramp(row, strict=False):
    """-> (keep, reason, flags) for one DRAMP row, using its own annotation columns."""
    seq = (row.get("Sequence") or "").strip().upper()
    if not seq:
        return False, "empty_sequence", ""
    if not is_canonical(seq):
        # the residue list goes into a printed log line, so keep it ASCII:
        # DRAMP contains entries written with Greek letters
        bad = ascii_safe("".join(sorted(set(seq) - set("ACDEFGHIKLMNPQRSTVWY"))), 20)
        return False, ("length_out_of_range" if not bad else f"non_standard_residue:{bad}"), ""

    stereo = (row.get("Stereochemistry") or "").strip()
    if not _unknown(stereo) and stereo.lower() not in ("l", "l (to be determined)"):
        # 'D', 'Mix', 'Mixed(D-Leu9)' ...
        return False, "d_amino_acid", ""

    shape = (row.get("Linear/Cyclic/Branched") or "").strip().lower()
    if shape in _NON_LINEAR:
        return False, ("branched_backbone" if shape != "cyclic" else "cyclic_backbone"), ""

    nterm = (row.get("N-terminal_Modification") or "")
    cterm = (row.get("C-terminal_Modification") or "")
    other = (row.get("Other_Modifications") or "")
    joined = " ".join([nterm, cterm, other])
    if _CYCLIZE_RE.search(joined):
        return False, "cyclic_backbone", ""
    if _LABEL_RE.search(joined):
        return False, "chemical_label", ""

    flags = sorted({name for name, rx in _FLAGS if rx.search(joined)})
    if _unknown(stereo) or _unknown(shape):
        if strict:
            return False, "chemistry_unverified", ""
        flags.append("chemistry_unverified")
    return True, "", ";".join(sorted(set(flags)))


def load_dramp(path, plant_accs=(), strict=False, require_activity=True):
    """Parse and clean DRAMP's general table.

    plant_accs: UniProt accessions known to be Viridiplantae. DRAMP's Source column is
    free text, so plant origin is resolved through its Swiss_Prot_Entry cross-reference.

    -> (records, log, report_rows)
    """
    import csv
    csv.field_size_limit(10 ** 7)
    plant_accs = set(plant_accs)
    recs, report, counts = [], [], {}
    n_rows = 0
    with open(path, encoding=ENC, errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n_rows += 1
            acc = (row.get("Swiss_Prot_Entry") or "").strip()
            acc = "" if acc.lower() in ("n/a", "not found", "na") else acc.split()[0] if acc else ""
            did = (row.get("DRAMP_ID") or f"DRAMP{n_rows:05d}").strip()
            activity = row.get("Activity") or ""
            if require_activity and not re.search(r"antimicrob|antibacter", activity, re.I):
                counts["not_antibacterial"] = counts.get("not_antibacterial", 0) + 1
                report.append(dict(id=did, decision="dropped", reason="not_antibacterial",
                                   flags="", seq=(row.get("Sequence") or "").strip().upper()))
                continue
            keep, reason, flags = classify_dramp(row, strict=strict)
            seq = (row.get("Sequence") or "").strip().upper()
            if not keep:
                counts[reason.split(":")[0]] = counts.get(reason.split(":")[0], 0) + 1
                report.append(dict(id=did, decision="dropped", reason=reason, flags="", seq=seq))
                continue
            report.append(dict(id=did, decision="kept", reason="", flags=flags, seq=seq))
            # '|' and whitespace are field separators in our FASTA headers, so strip them
            org = re.sub(r"[|>\s]+", " ", (row.get("Source") or "")).strip()[:80]
            recs.append(dict(id=did, seq=seq, organism=org,
                             taxid=-1, plant=int(bool(acc) and acc in plant_accs),
                             flags=flags, source="dramp3", uniprot=acc))
    log = {"dramp_rows": n_rows, "dramp_kept": len(recs),
           "dramp_dropped": n_rows - len(recs), "dramp_drop_reasons": counts,
           "dramp_plant": sum(r["plant"] for r in recs)}
    return recs, log, report


def load_fasta_positives(path, tag="external"):
    """Generic FASTA (DBAASP / APD3 export). Only alphabet and length rules apply."""
    from common import read_fasta
    recs, report = [], []
    for h, s in read_fasta(path):
        s = s.strip().upper()
        pid = h.split()[0][:40]
        if not is_canonical(s):
            report.append(dict(id=pid, decision="dropped", reason="non_standard_or_length",
                               flags="", seq=s))
            continue
        report.append(dict(id=pid, decision="kept", reason="", flags="chemistry_unverified", seq=s))
        recs.append(dict(id=pid, seq=s, organism="", taxid=-1, plant=0,
                         flags="chemistry_unverified", source=tag, uniprot=""))
    return recs, {f"{tag}_rows": len(report), f"{tag}_kept": len(recs)}, report
