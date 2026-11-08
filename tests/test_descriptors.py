"""Check the physicochemical descriptors the proposal named modlAMP for.

modlAMP is unmaintained and will not build on current Python, so common.py computes
net charge, hydrophobicity, hydrophobic moment, pI and MW directly. This test pins
those implementations down so swapping the library out is defensible rather than
silent:

  1. against an independent re-implementation written straight from the published
     formulae (Eisenberg consensus scale; Eisenberg et al. 1982 hydrophobic moment),
  2. against hand-checkable invariants,
  3. against modlAMP itself, if it happens to be installed (skipped otherwise).

Run:  python tests/test_descriptors.py
"""
import math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import (EISENBERG, net_charge, hydrophobicity, hydrophobic_moment,
                    descriptors, DESC_NAMES)

SEQS = ["GLFDIVKKVVGALGSL", "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        "KKKKKKKKKK", "DDDDDDDDDD", "AAAAAAAAAAAAAAAAAAAA",
        "RGGRLCYCRRRFCVCVGR"]
fails = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got:.4f}, want {want:.4f} (tol {tol})")
    if not ok:
        fails.append(name)


# --- 1. independent re-implementation of the two scale-based descriptors -----
def ref_hydrophobicity(seq):
    """Mean Eisenberg consensus hydrophobicity, written out longhand."""
    total = 0.0
    for residue in seq:
        total = total + EISENBERG[residue]
    return total / len(seq)


def ref_moment(seq, delta_deg=100.0):
    """Eisenberg et al. 1982 hydrophobic moment, built from the sum-of-angles form.

    mu = sqrt( (sum H_i sin(i*d))^2 + (sum H_i cos(i*d))^2 ) / N
    Written here with complex arithmetic so it shares no code with common.py.
    """
    d = math.radians(delta_deg)
    acc = sum(EISENBERG[a] * complex(math.cos(i * d), math.sin(i * d))
              for i, a in enumerate(seq))
    return abs(acc) / len(seq)


print("1. against an independent implementation of the published formulae")
for s in SEQS:
    check(f"hydrophobicity({s[:12]})", hydrophobicity(s), ref_hydrophobicity(s), 1e-12)
    check(f"moment({s[:12]})", hydrophobic_moment(s), ref_moment(s), 1e-12)

# --- 2. invariants ----------------------------------------------------------
print("\n2. invariants")
q_k, q_d = net_charge("KKKKKKKKKK"), net_charge("DDDDDDDDDD")
# Hand-computed from the EMBOSS pKa values in common.py, at pH 7:
#   Lys  : 1/(1+10^(7-10.8))  = 0.99984  each, x10        = +9.9984
#   N-term: 1/(1+10^(7-8.6))  = +0.9755
#   C-term: -1/(1+10^(3.6-7)) = -0.9996
#   total                                                 = +9.974
check("charge(K10)", q_k, 9.974, 0.01)
#   Asp  : -1/(1+10^(3.9-7))  = -0.99921 each, x10        = -9.9921
#   plus the same two termini                             = -10.016
check("charge(D10)", q_d, -10.016, 0.01)
#   Ala has no ionisable side chain, so only the termini contribute: +0.9755-0.9996
check("charge(A20)", net_charge("AAAAAAAAAAAAAAAAAAAA"), -0.0241, 0.01)
# pH dependence: a His-containing peptide must gain charge as pH falls
q_hi, q_lo = net_charge("HHHHAAAA", ph=9.0), net_charge("HHHHAAAA", ph=5.0)
print(f"  {'ok  ' if q_lo > q_hi else 'FAIL'} charge rises as pH falls: "
      f"pH9 {q_hi:.3f} -> pH5 {q_lo:.3f}")
if q_lo <= q_hi:
    fails.append("charge pH monotonicity")
# a perfectly amphipathic helix must have a far larger moment than a uniform one
amphi = hydrophobic_moment("LKKLLKKLLKKLLKKL")
flat = hydrophobic_moment("AAAAAAAAAAAAAAAA")
print(f"  {'ok  ' if amphi > 5 * flat else 'FAIL'} amphipathic moment {amphi:.3f} "
      f">> uniform moment {flat:.3f}")
if amphi <= 5 * flat:
    fails.append("moment amphipathicity")
# descriptors() must return the documented order and length
d = descriptors(SEQS[0])
print(f"  {'ok  ' if len(d) == len(DESC_NAMES) else 'FAIL'} descriptors() returns "
      f"{len(d)} values for names {DESC_NAMES}")
if len(d) != len(DESC_NAMES):
    fails.append("descriptor arity")

# --- 3. modlAMP, if present -------------------------------------------------
print("\n3. against modlAMP (the library the proposal named)")
try:
    from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
    for s in SEQS[:3]:
        g = GlobalDescriptor([s]); g.calculate_charge(ph=7.0, amide=False)
        check(f"modlAMP charge({s[:12]})", net_charge(s), float(g.descriptor[0][0]), 0.15)
        p = PeptideDescriptor([s], "eisenberg"); p.calculate_global()
        check(f"modlAMP hydrophobicity({s[:12]})", hydrophobicity(s),
              float(p.descriptor[0][0]), 0.02)
        p = PeptideDescriptor([s], "eisenberg"); p.calculate_moment(angle=100)
        check(f"modlAMP moment({s[:12]})", hydrophobic_moment(s),
              float(p.descriptor[0][0]), 0.02)
except ImportError:
    print("  SKIPPED: modlAMP is not installed (expected on Python 3.12+).")
    print("  The checks in sections 1 and 2 are what justify the substitution.")

print()
if fails:
    print("DESCRIPTOR TEST FAILED:", ", ".join(fails))
    raise SystemExit(1)
print("DESCRIPTOR TEST PASSED")
