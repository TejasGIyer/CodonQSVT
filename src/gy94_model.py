"""
GY94 Codon Substitution Rate Matrix
====================================
Implementation of Goldman & Yang (1994) codon-based model of nucleotide
substitution for protein-coding DNA sequences.

Updates:
- Includes numerical optimization to calculate species-specific variability (V)
  based on target dN/dS (omega) ratios.
"""

import numpy as np

# =========================================================================
# STANDARD GENETIC CODE (Universal)
# =========================================================================
GENETIC_CODE = {
    'TTT': 'Phe', 'TTC': 'Phe', 'TTA': 'Leu', 'TTG': 'Leu',
    'CTT': 'Leu', 'CTC': 'Leu', 'CTA': 'Leu', 'CTG': 'Leu',
    'ATT': 'Ile', 'ATC': 'Ile', 'ATA': 'Ile', 'ATG': 'Met',
    'GTT': 'Val', 'GTC': 'Val', 'GTA': 'Val', 'GTG': 'Val',
    'TCT': 'Ser', 'TCC': 'Ser', 'TCA': 'Ser', 'TCG': 'Ser',
    'CCT': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
    'ACT': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
    'GCT': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
    'TAT': 'Tyr', 'TAC': 'Tyr', 'TAA': 'STOP', 'TAG': 'STOP',
    'CAT': 'His', 'CAC': 'His', 'CAA': 'Gln', 'CAG': 'Gln',
    'AAT': 'Asn', 'AAC': 'Asn', 'AAA': 'Lys', 'AAG': 'Lys',
    'GAT': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
    'TGT': 'Cys', 'TGC': 'Cys', 'TGA': 'STOP', 'TGG': 'Trp',
    'CGT': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
    'AGT': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
    'GGT': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
}

# 61 sense codons (excluding TAA, TAG, TGA)
SENSE_CODONS = sorted([c for c, aa in GENETIC_CODE.items() if aa != 'STOP'])
CODON_TO_INDEX = {c: i for i, c in enumerate(SENSE_CODONS)}
N_SENSE = len(SENSE_CODONS)  # 61

# =========================================================================
# GRANTHAM DISTANCE MATRIX (Grantham 1974)
# =========================================================================
_AA_ORDER = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly',
             'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser',
             'Thr', 'Trp', 'Tyr', 'Val']

_GRANTHAM_UPPER = {
    ('Ala','Arg'):112, ('Ala','Asn'):111, ('Ala','Asp'):126, ('Ala','Cys'):195,
    ('Ala','Gln'):91,  ('Ala','Glu'):107, ('Ala','Gly'):60,  ('Ala','His'):86,
    ('Ala','Ile'):94,  ('Ala','Leu'):96,  ('Ala','Lys'):106, ('Ala','Met'):84,
    ('Ala','Phe'):113, ('Ala','Pro'):27,  ('Ala','Ser'):99,  ('Ala','Thr'):58,
    ('Ala','Trp'):148, ('Ala','Tyr'):112, ('Ala','Val'):64,
    ('Arg','Asn'):86,  ('Arg','Asp'):96,  ('Arg','Cys'):180, ('Arg','Gln'):43,
    ('Arg','Glu'):54,  ('Arg','Gly'):125, ('Arg','His'):29,  ('Arg','Ile'):97,
    ('Arg','Leu'):102, ('Arg','Lys'):26,  ('Arg','Met'):91,  ('Arg','Phe'):97,
    ('Arg','Pro'):103, ('Arg','Ser'):110, ('Arg','Thr'):71,  ('Arg','Trp'):101,
    ('Arg','Tyr'):77,  ('Arg','Val'):96,
    ('Asn','Asp'):23,  ('Asn','Cys'):139, ('Asn','Gln'):46,  ('Asn','Glu'):42,
    ('Asn','Gly'):80,  ('Asn','His'):68,  ('Asn','Ile'):149, ('Asn','Leu'):153,
    ('Asn','Lys'):94,  ('Asn','Met'):142, ('Asn','Phe'):158, ('Asn','Pro'):91,
    ('Asn','Ser'):46,  ('Asn','Thr'):65,  ('Asn','Trp'):174, ('Asn','Tyr'):143,
    ('Asn','Val'):133,
    ('Asp','Cys'):154, ('Asp','Gln'):61,  ('Asp','Glu'):45,  ('Asp','Gly'):94,
    ('Asp','His'):81,  ('Asp','Ile'):168, ('Asp','Leu'):172, ('Asp','Lys'):101,
    ('Asp','Met'):160, ('Asp','Phe'):177, ('Asp','Pro'):108, ('Asp','Ser'):65,
    ('Asp','Thr'):85,  ('Asp','Trp'):181, ('Asp','Tyr'):160, ('Asp','Val'):152,
    ('Cys','Gln'):154, ('Cys','Glu'):170, ('Cys','Gly'):159, ('Cys','His'):174,
    ('Cys','Ile'):198, ('Cys','Leu'):198, ('Cys','Lys'):202, ('Cys','Met'):196,
    ('Cys','Phe'):205, ('Cys','Pro'):169, ('Cys','Ser'):112, ('Cys','Thr'):149,
    ('Cys','Trp'):215, ('Cys','Tyr'):194, ('Cys','Val'):192,
    ('Gln','Glu'):29,  ('Gln','Gly'):87,  ('Gln','His'):24,  ('Gln','Ile'):109,
    ('Gln','Leu'):113, ('Gln','Lys'):53,  ('Gln','Met'):101, ('Gln','Phe'):116,
    ('Gln','Pro'):76,  ('Gln','Ser'):68,  ('Gln','Thr'):42,  ('Gln','Trp'):130,
    ('Gln','Tyr'):99,  ('Gln','Val'):96,
    ('Glu','Gly'):98,  ('Glu','His'):40,  ('Glu','Ile'):134, ('Glu','Leu'):138,
    ('Glu','Lys'):56,  ('Glu','Met'):126, ('Glu','Phe'):140, ('Glu','Pro'):93,
    ('Glu','Ser'):80,  ('Glu','Thr'):65,  ('Glu','Trp'):152, ('Glu','Tyr'):122,
    ('Glu','Val'):121,
    ('Gly','His'):98,  ('Gly','Ile'):135, ('Gly','Leu'):138, ('Gly','Lys'):127,
    ('Gly','Met'):127, ('Gly','Phe'):153, ('Gly','Pro'):42,  ('Gly','Ser'):56,
    ('Gly','Thr'):59,  ('Gly','Trp'):184, ('Gly','Tyr'):147, ('Gly','Val'):109,
    ('His','Ile'):94,  ('His','Leu'):99,  ('His','Lys'):32,  ('His','Met'):87,
    ('His','Phe'):100, ('His','Pro'):77,  ('His','Ser'):89,  ('His','Thr'):47,
    ('His','Trp'):115, ('His','Tyr'):83,  ('His','Val'):84,
    ('Ile','Leu'):5,   ('Ile','Lys'):102, ('Ile','Met'):10,  ('Ile','Phe'):21,
    ('Ile','Pro'):95,  ('Ile','Ser'):142, ('Ile','Thr'):89,  ('Ile','Trp'):61,
    ('Ile','Tyr'):33,  ('Ile','Val'):29,
    ('Leu','Lys'):107, ('Leu','Met'):15,  ('Leu','Phe'):22,  ('Leu','Pro'):98,
    ('Leu','Ser'):145, ('Leu','Thr'):92,  ('Leu','Trp'):61,  ('Leu','Tyr'):36,
    ('Leu','Val'):32,
    ('Lys','Met'):95,  ('Lys','Phe'):102, ('Lys','Pro'):103, ('Lys','Ser'):121,
    ('Lys','Thr'):78,  ('Lys','Trp'):110, ('Lys','Tyr'):85,  ('Lys','Val'):97,
    ('Met','Phe'):28,  ('Met','Pro'):87,  ('Met','Ser'):135, ('Met','Thr'):81,
    ('Met','Trp'):67,  ('Met','Tyr'):36,  ('Met','Val'):21,
    ('Phe','Pro'):114, ('Phe','Ser'):155, ('Phe','Thr'):103, ('Phe','Trp'):40,
    ('Phe','Tyr'):22,  ('Phe','Val'):50,
    ('Pro','Ser'):74,  ('Pro','Thr'):38,  ('Pro','Trp'):147, ('Pro','Tyr'):110,
    ('Pro','Val'):68,
    ('Ser','Thr'):58,  ('Ser','Trp'):177, ('Ser','Tyr'):144, ('Ser','Val'):124,
    ('Thr','Trp'):128, ('Thr','Tyr'):92,  ('Thr','Val'):69,
    ('Trp','Tyr'):37,  ('Trp','Val'):88,
    ('Tyr','Val'):55,
}

def grantham_distance(aa1, aa2):
    """Look up the Grantham (1974) distance between two amino acids."""
    if aa1 == aa2:
        return 0
    key = (aa1, aa2) if (aa1, aa2) in _GRANTHAM_UPPER else (aa2, aa1)
    return _GRANTHAM_UPPER.get(key, 100)  # default 100 if not found

def is_transition(nuc1, nuc2):
    """Check if a single-nucleotide change is a transition."""
    transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
    return (nuc1, nuc2) in transitions

def codon_diff(codon_i, codon_j):
    """Compare two codons and return differences."""
    diffs = [(p, codon_i[p], codon_j[p]) for p in range(3) if codon_i[p] != codon_j[p]]
    n_diff = len(diffs)
    if n_diff == 1:
        pos, nuc_i, nuc_j = diffs[0]
        return n_diff, pos, is_transition(nuc_i, nuc_j)
    return n_diff, None, None

# =========================================================================
# OMEGA (dN/dS) CALCULATOR ENGINE
# =========================================================================
def calculate_implied_omega(codon_frequencies, kappa, V):
    """Calculates the implied dN/dS (omega) of the matrix for a given V."""
    pi = np.zeros(N_SENSE)
    for codon in SENSE_CODONS:
        pi[CODON_TO_INDEX[codon]] = codon_frequencies.get(codon, 0.0)

    pi_sum = pi.sum()
    if pi_sum > 0:
        pi /= pi_sum

    actual_nonsyn_rate = 0.0
    neutral_nonsyn_rate = 0.0

    for i, ci in enumerate(SENSE_CODONS):
        aa_i = GENETIC_CODE[ci]
        for j, cj in enumerate(SENSE_CODONS):
            if i == j: continue
            aa_j = GENETIC_CODE[cj]

            n_diff, _, is_ts = codon_diff(ci, cj)
            if n_diff != 1: continue

            d = grantham_distance(aa_i, aa_j)

            # Base unscaled mutation rate
            rate = pi[j]
            if is_ts: rate *= kappa

            # If the mutation changes the protein
            if aa_i != aa_j:
                actual_nonsyn_rate += pi[i] * rate * np.exp(-d / V)
                neutral_nonsyn_rate += pi[i] * rate

    if neutral_nonsyn_rate == 0:
        return 0
    return actual_nonsyn_rate / neutral_nonsyn_rate

# =========================================================================
# RATE MATRIX BUILDER
# =========================================================================
def build_gy94_rate_matrix(codon_frequencies, kappa=None, V=None):
    """Build the GY94 rate matrix Q (61x61).

    Defaults (kappa, V) are pulled from src.constants (the single source of
    truth, calibrated to the paper: kappa=1.8425, V=13.5) when not supplied.
    This removes the previous hard-coded (1.45, 43.99) defaults that did not
    match the paper and could silently produce a different Hamiltonian.
    """
    if kappa is None or V is None:
        from src.constants import GY94_KAPPA, GY94_V
        if kappa is None:
            kappa = GY94_KAPPA
        if V is None:
            V = GY94_V
    pi = np.zeros(N_SENSE)
    for codon in SENSE_CODONS:
        idx = CODON_TO_INDEX[codon]
        pi[idx] = codon_frequencies.get(codon, 0.0)

    pi_sum = pi.sum()
    if pi_sum > 0:
        pi /= pi_sum

    Q = np.zeros((N_SENSE, N_SENSE))

    n_transitions, n_transversions = 0, 0
    n_synonymous, n_nonsynonymous = 0, 0

    for i, codon_i in enumerate(SENSE_CODONS):
        aa_i = GENETIC_CODE[codon_i]

        for j, codon_j in enumerate(SENSE_CODONS):
            if i == j: continue

            aa_j = GENETIC_CODE[codon_j]
            n_diff, diff_pos, is_ts = codon_diff(codon_i, codon_j)

            if n_diff != 1: continue

            d = grantham_distance(aa_i, aa_j)
            selection = np.exp(-d / V)
            rate = pi[j] * selection

            if is_ts:
                rate *= kappa
                n_transitions += 1
            else:
                n_transversions += 1

            if aa_i == aa_j: n_synonymous += 1
            else: n_nonsynonymous += 1

            Q[i, j] = rate

    for i in range(N_SENSE):
        Q[i, i] = -np.sum(Q[i, :])

    avg_rate = -np.sum(pi * np.diag(Q))
    if avg_rate > 0:
        mu = 1.0 / avg_rate
        Q *= mu
    else:
        mu = 1.0

    row_sums = np.sum(Q, axis=1)
    avg_rate_after = -np.sum(pi * np.diag(Q))

    max_reversibility_error = 0.0
    for i in range(N_SENSE):
        for j in range(i + 1, N_SENSE):
            if Q[i, j] > 0 or Q[j, i] > 0:
                err = abs(pi[i] * Q[i, j] - pi[j] * Q[j, i])
                max_reversibility_error = max(max_reversibility_error, err)

    info = {
        'n_sense_codons': N_SENSE,
        'kappa': kappa,
        'V': V,
        'mu': mu,
        'n_transitions': n_transitions // 2,
        'n_transversions': n_transversions // 2,
        'n_synonymous': n_synonymous // 2,
        'n_nonsynonymous': n_nonsynonymous // 2,
        'avg_rate_after_scaling': avg_rate_after,
        'max_row_sum_error': np.max(np.abs(row_sums)),
        'max_reversibility_error': max_reversibility_error,
        'n_nonzero_offdiag': np.count_nonzero(Q - np.diag(np.diag(Q))),
        'eigenvalue_range': (np.min(np.real(np.linalg.eigvals(Q))),
                             np.max(np.real(np.linalg.eigvals(Q)))),
    }

    return Q, SENSE_CODONS, pi, info


def print_gy94_report(Q, sense_codons, pi, info):
    """Print a detailed report of the GY94 rate matrix."""
    print("=" * 70)
    print("  GY94 CODON SUBSTITUTION RATE MATRIX")
    print("  Goldman & Yang (1994) Mol. Biol. Evol. 11(5):725-736")
    print("=" * 70)

    print(f"\n  Parameters:")
    print(f"    kappa (Ts/Tv ratio):      {info['kappa']}")
    print(f"    V (gene variability):     {info['V']:.4f}")
    print(f"    mu (scaling factor):      {info['mu']:.6f}")

    print(f"\n  Matrix properties:")
    print(f"    Sense codons:             {info['n_sense_codons']}")
    print(f"    Non-zero off-diagonal:    {info['n_nonzero_offdiag']}")
    print(f"    Transition pairs:         {info['n_transitions']}")
    print(f"    Transversion pairs:       {info['n_transversions']}")
    print(f"    Synonymous pairs:         {info['n_synonymous']}")
    print(f"    Non-synonymous pairs:     {info['n_nonsynonymous']}")

    print(f"\n  Verification:")
    print(f"    Avg rate after scaling:   {info['avg_rate_after_scaling']:.10f} (should be 1.0)")
    print(f"    Max row sum error:        {info['max_row_sum_error']:.2e} (should be ~0)")
    print(f"    Max reversibility error:  {info['max_reversibility_error']:.2e} (should be ~0)")
    print(f"    Eigenvalue range:         [{info['eigenvalue_range'][0]:.4f}, {info['eigenvalue_range'][1]:.4f}]")

    print(f"\n  Top 10 highest substitution rates:")
    print(f"  {'From':>6} -> {'To':>6}  {'AA_i':>4} -> {'AA_j':>4}  {'Rate':>10}  {'Type':>6}  {'Grantham':>8}")
    print(f"  {'-'*6}    {'-'*6}  {'-'*4}    {'-'*4}  {'-'*10}  {'-'*6}  {'-'*8}")

    rates = []
    for i in range(N_SENSE):
        for j in range(N_SENSE):
            if i != j and Q[i, j] > 0:
                aa_i = GENETIC_CODE[sense_codons[i]]
                aa_j = GENETIC_CODE[sense_codons[j]]
                n_diff, _, is_ts = codon_diff(sense_codons[i], sense_codons[j])
                d = grantham_distance(aa_i, aa_j)
                ts_type = "Ts" if is_ts else "Tv"
                rates.append((Q[i, j], sense_codons[i], sense_codons[j], aa_i, aa_j, ts_type, d))

    rates.sort(reverse=True)
    for rate, ci, cj, aai, aaj, ts, d in rates[:10]:
        print(f"  {ci:>6} -> {cj:>6}  {aai:>4} -> {aaj:>4}  {rate:10.6f}  {ts:>6}  {d:8d}")


# =========================================================================
# Standalone execution has been removed.
# Use `python src/qsp_circuit.py` as the entry point for the GAPDH pipeline.
# =========================================================================
