"""
GAPDH coding sequences for 4 mammalian species.
Used for codon frequency estimation in the GY94 rate matrix.

Source: NCBI RefSeq
  HUMAN: NM_002046
  MOUSE: NM_008084
  RAT:   NM_017008
  DOG:   NM_001003142

Each sequence is 1002 nt = 334 codons.
"""

import numpy as np
from collections import Counter, OrderedDict


# =========================================================================
# RAW SEQUENCES (cleaned: uppercase, no whitespace)
# =========================================================================

HUMAN = (
    "ATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGG"
    "GCTGCTTTTAACTCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTC"
    "AACTACATGGTTTACATGTTCCAATATGATTCCACCCATGGCAAATTCCATGGCACCGTC"
    "AAGGCTGAGAACGGGAAGCTTGTCATCAATGGAAATCCCATCACCATCTTCCAGGAGCGA"
    "GATCCCTCCAAAATCAAGTGGGGCGATGCTGGCGCTGAGTACGTCGTGGAGTCCACTGGC"
    "GTCTTCACCACCATGGAGAAGGCTGGGGCTCATTTGCAGGGGGGAGCCAAAAGGGTCATC"
    "ATCTCTGCCCCCTCTGCTGATGCCCCCATGTTCGTCATGGGTGTGAACCATGAGAAGTAT"
    "GACAACAGCCTCAAGATCATCAGCAATGCCTCCTGCACCACCAACTGCTTAGCACCCCTG"
    "GCCAAGGTCATCCATGACAACTTTGGTATCGTGGAAGGACTCATGACCACAGTCCATGCC"
    "ATCACTGCCACCCAGAAGACTGTGGATGGCCCCTCCGGGAAACTGTGGCGTGATGGCCGC"
    "GGGGCTCTCCAGAACATCATCCCTGCCTCTACTGGCGCTGCCAAGGCTGTGGGCAAGGTC"
    "ATCCCTGAGCTGAACGGGAAGCTCACTGGCATGGCCTTCCGTGTCCCCACTGCCAACGTG"
    "TCAGTGGTGGACCTGACCTGCCGTCTAGAAAAACCTGCCAAATATGATGACATCAAGAAG"
    "GTGGTGAAGCAGGCGTCGGAGGGCCCCCTCAAGGGCATCCTGGGCTACACTGAGCACCAG"
    "GTGGTCTCCTCTGACTTCAACAGCGACACCCACTCCTCCACCTTTGACGCTGGGGCTGGC"
    "ATTGCCCTCAACGACCACTTTGTCAAGCTCATTTCCTGGTATGACAACGAATTTGGCTAC"
    "AGCAACAGGGTGGTGGACCTCATGGCCCACATGGCCTCCAAG"
)

MOUSE = (
    "ATGGTGAAGGTCGGTGTGAACGGATTTGGCCGTATTGGGCGCCTGGTCACCAGGGCTGCC"
    "ATTTGCAGTGGCAAAGTGGAGATTGTTGCCATCAACGACCCCTTCATTGACCTCAACTAC"
    "ATGGTCTACATGTTCCAGTATGACTCCACTCACGGCAAATTCAACGGCACAGTCAAGGCC"
    "GAGAATGGGAAGCTTGTCATCAACGGGAAGCCCATCACCATCTTCCAGGAGCGAGACCCC"
    "ACTAACATCAAATGGGGTGAGGCCGGTGCTGAGTATGTCGTGGAGTCTACTGGTGTCTTC"
    "ACCACCATGGAGAAGGCCGGGGCCCACTTGAAGGGTGGAGCCAAAAGGGTCATCATCTCC"
    "GCCCCTTCTGCCGATGCCCCCATGTTTGTGATGGGTGTGAACCACGAGAAATATGACAAC"
    "TCACTCAAGATTGTCAGCAATGCATCCTGCACCACCAACTGCTTAGCCCCCCTGGCCAAG"
    "GTCATCCATGACAACTTTGGCATTGTGGAAGGGCTCATGACCACAGTCCATGCCATCACT"
    "GCCACCCAGAAGACTGTGGATGGCCCCTCTGGAAAGCTGTGGCGTGATGGCCGTGGGGCT"
    "GCCCAGAACATCATCCCTGCATCCACTGGTGCTGCCAAGGCTGTGGGCAAGGTCATCCCA"
    "GAGCTGAACGGGAAGCTCACTGGCATGGCCTTCCGTGTTCCTACCCCCAATGTGTCCGTC"
    "GTGGATCTGACGTGCCGCCTGGAGAAACCTGCCAAGTATGATGACATCAAGAAGGTGGTG"
    "AAGCAGGCATCTGAGGGCCCACTGAAGGGCATCTTGGGCTACACTGAGGACCAGGTTGTC"
    "TCCTGCGACTTCAACAGCAACTCCCACTCTTCCACCTTCGATGCCGGGGCTGGCATTGCT"
    "CTCAATGACAACTTTGTCAAGCTCATTTCCTGGTATGACAATGAATACGGCTACAGCAAC"
    "AGGGTGGTGGACCTCATGGCCTACATGGCCTCCAAGGAGTAA"
)

RAT = (
    "ATGGTGAAGGTCGGTGTGAACGGATTTGGCCGTATCGGACGCCTGGTTACCAGGGCTGCC"
    "TTCTCTTGTGACAAAGTGGACATTGTTGCCATCAACGACCCCTTCATTGACCTCAACTAC"
    "ATGGTCTACATGTTCCAGTATGACTCTACCCACGGCAAGTTCAACGGCACAGTCAAGGCT"
    "GAGAATGGGAAGCTGGTCATCAACGGGAAACCCATCACCATCTTCCAGGAGCGAGATCCC"
    "GCTAACATCAAATGGGGTGATGCTGGTGCTGAGTATGTCGTGGAGTCTACTGGCGTCTTC"
    "ACCACCATGGAGAAGGCTGGGGCTCACCTGAAGGGTGGGGCCAAAAGGGTCATCATCTCC"
    "GCCCCTTCCGCTGATGCCCCCATGTTTGTGATGGGTGTGAACCACGAGAAATATGACAAC"
    "TCCCTCAAGATTGTCAGCAATGCATCCTGCACCACCAACTGCTTAGCCCCCCTGGCCAAG"
    "GTCATCCATGACAACTTTGGCATCGTGGAAGGGCTCATGACCACAGTCCATGCCATCACT"
    "GCCACTCAGAAGACTGTGGATGGCCCCTCTGGAAAGCTGTGGCGTGATGGCCGTGGGGCA"
    "GCCCAGAACATCATCCCTGCATCCACTGGTGCTGCCAAGGCTGTGGGCAAGGTCATCCCA"
    "GAGCTGAACGGGAAGCTCACTGGCATGGCCTTCCGTGTTCCTACCCCCAATGTATCCGTT"
    "GTGGATCTGACATGCCGCCTGGAGAAACCTGCCAAGTATGATGACATCAAGAAGGTGGTG"
    "AAGCAGGCGGCCGAGGGCCCACTAAAGGGCATCCTGGGCTACACTGAGGACCAGGTTGTC"
    "TCCTGTGACTTCAACAGCAACTCCCATTCTTCCACCTTTGATGCTGGGGCTGGCATTGCT"
    "CTCAATGACAACTTTGTGAAGCTCATTTCCTGGTATGACAATGAATATGGCTACAGCAAC"
    "AGGGTGGTGGACCTCATGGCCTACATGGCCTCCAAGGAGTAA"
)

DOG = (
    "ATGGTGAAGGTCGGAGTGAACGGATTTGGCCGTATTGGGCGCCTGGTCACCAGGGCTGCT"
    "TTTAACTCTGGCAAAGTGGATATTGTCGCCATCAATGACCCCTTCATTGATCTCAACTAC"
    "ATGGTGTACATGTTCCAGTATGATTCTACCCACGGCAAATTCCACGGCACAGTCAAGGCT"
    "GAGAACGGGAAACTTGTCATCAACGGGAAGTCCATCTCCATCTTCCAGGAGCGAGATCCC"
    "GCCAACATCAAATGGGGTGATGCTGGTGCTGAGTATGTTGTGGAGTCCACTGGGGTCTTC"
    "ACCACCATGGAGAAGGCTGGGGCTCACTTGAAAGGCGGGGCCAAGAGGGTCATCATCTCT"
    "GCTCCTTCTGCTGATGCCCCCATGTTTGTGATGGGCGTGAACCATGAGAAGTATGACAAC"
    "TCCCTCAAGATTGTCAGCAATGCCTCCTGCACCACCAACTGCTTGGCTCCTCTAGCCAAA"
    "GTCATCCATGACCACTTCGGCATCGTGGAGGGCCTCATGACCACCGTCCATGCCATCACT"
    "GCCACCCAGAAGACCGTGGACGGCCCCTCTGGGAAGATGTGGCGTGACGGCCGAGGGGCT"
    "GCCCAGAACATCATCCCTGCTTCCACTGGCGCTGCCAAGGCTGTGGGCAAGGTCATCCCT"
    "GAGCTGAACGGGAAGCTCACTGGCATGGCCTTCCGTGTCCCCACCCCCAATGTATCAGTT"
    "GTGGATCTGACCTGCCGCCTGGAGAAAGCTGCCAAATATGACGACATCAAGAAGGTAGTG"
    "AAGCAGGCATCGGAGGGACCCCTCAAAGGCATCCTGGGCTACACTGAGGACCAGGTGGTC"
    "TCCTGTGACTTCAACAGTGACACCCACTCTTCCACCTTCGACGCCGGGGCTGGCATTGCC"
    "CTCAATGACCACTTTGTCAAGCTCATTTCCTGGTATGACAATGAATTTGGCTACAGCAAC"
    "CGGGTGGTGGACCTCATGGTCCACATGGCCTCCAAGGAGTAA"
)

ALL_SEQUENCES = OrderedDict([
    ('HUMAN', HUMAN),
    ('MOUSE', MOUSE),
    ('RAT',   RAT),
    ('DOG',   DOG),
])

STOP_CODONS = {'TAA', 'TAG', 'TGA'}

# 61 sense codons in canonical sorted order — fixed regardless of input sequence
SENSE_CODONS_SORTED = sorted([
    a + b + c
    for a in 'ACGT' for b in 'ACGT' for c in 'ACGT'
    if (a + b + c) not in STOP_CODONS
])
assert len(SENSE_CODONS_SORTED) == 61
CANONICAL_INDEX = {codon: i for i, codon in enumerate(SENSE_CODONS_SORTED)}


def _split_codons(seq):
    """Split a sequence into codons of length 3."""
    return [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]


def pooled_codon_frequencies():
    """
    Pool codon counts across all 4 GAPDH sequences. Returns a dict
    mapping each sense codon to its normalized frequency.
    Stop codons are skipped.
    """
    counter = Counter()
    for name, seq in ALL_SEQUENCES.items():
        for codon in _split_codons(seq):
            if len(codon) == 3 and codon not in STOP_CODONS:
                counter[codon] += 1

    total = sum(counter.values())
    if total == 0:
        return {}
    return {codon: count / total for codon, count in counter.items()}


def get_combined_sequence():
    """Return the 4 GAPDH sequences concatenated into one string."""
    return "".join(ALL_SEQUENCES[name] for name in ['HUMAN', 'MOUSE', 'RAT', 'DOG'])


def build_gapdh_register(n_qubits=6):
    """
    Build a classical register dict compatible with aae_encode().

    Unlike compression2.build_classical_register (which numbers codons by
    first appearance and lets n_qubits float), this function uses a FIXED
    canonical 61-codon ordering and pads to 2^n_qubits states. This ensures
    n_qubits=6 (matching block_encoding/qsp_circuit defaults) regardless of
    which species combination is used.

    Returns the same shape of dict that compression2.build_classical_register
    returns, so it can be passed directly to aae_encode().
    """
    n_states = 2 ** n_qubits
    if n_states < 61:
        raise ValueError(f"n_qubits={n_qubits} too small for 61 sense codons")

    # Pool codons across all 4 species
    counter = Counter()
    for name, seq in ALL_SEQUENCES.items():
        for codon in _split_codons(seq):
            if len(codon) == 3 and codon not in STOP_CODONS:
                counter[codon] += 1

    # Build the unique codon registry using canonical sorted order
    seen = OrderedDict()
    unique_register = []
    for codon in SENSE_CODONS_SORTED:
        if codon in counter:
            idx = CANONICAL_INDEX[codon]
            seen[codon] = idx
            unique_register.append({
                'unique_index': idx,
                'codon': codon,
                'weight': counter[codon],
                'binary': format(idx, f'0{n_qubits}b'),
            })

    # Position register (concatenated sequence)
    combined = get_combined_sequence()
    codon_sequence = _split_codons(combined)
    position_register = []
    for pos, codon in enumerate(codon_sequence):
        if codon in CANONICAL_INDEX:
            idx = CANONICAL_INDEX[codon]
            position_register.append({
                'position': pos, 'codon': codon,
                'unique_index': idx, 'binary': format(idx, f'0{n_qubits}b'),
            })

    # Weight vector padded to 2^n_qubits states
    weight_vector = np.zeros(n_states)
    for entry in unique_register:
        weight_vector[entry['unique_index']] = entry['weight']

    # Normalized amplitude vector
    d = weight_vector.copy()
    norm = np.linalg.norm(d)
    if norm > 0:
        d /= norm
    p_comp = d ** 2

    return {
        'sequence': combined,
        'codon_sequence': codon_sequence,
        'num_codons': len(codon_sequence),
        'unique_codons': seen,
        'num_unique': len(unique_register),
        'weights': dict(counter),
        'unique_register': unique_register,
        'position_register': position_register,
        'num_qubits': n_qubits,
        'weight_vector': weight_vector,
        'd_normalized': d,
        'p_comp': p_comp,
    }


def print_summary():
    """Print a summary of the GAPDH dataset."""
    print("=" * 70)
    print("  GAPDH SEQUENCES SUMMARY")
    print("=" * 70)
    for name, seq in ALL_SEQUENCES.items():
        codons = _split_codons(seq)
        n_stop = sum(1 for c in codons if c in STOP_CODONS)
        print(f"  {name:>6}: {len(seq):>5} nt | {len(codons):>4} codons | {n_stop} stop")

    freqs = pooled_codon_frequencies()
    print(f"\n  Pooled codon frequencies:")
    print(f"    Unique sense codons: {len(freqs)}")
    top = sorted(freqs.items(), key=lambda x: -x[1])[:5]
    print(f"    Top 5 codons:")
    for codon, freq in top:
        print(f"      {codon}: {freq:.4f}")

    reg = build_gapdh_register(n_qubits=6)
    print(f"\n  Classical register (n_qubits=6):")
    print(f"    Total codons:        {reg['num_codons']}")
    print(f"    Unique sense codons: {reg['num_unique']}")
    print(f"    State space:         2^{reg['num_qubits']} = {2**reg['num_qubits']}")


if __name__ == "__main__":
    print_summary()
