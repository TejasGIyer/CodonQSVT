"""
QSP-GY94: Quantum Signal Processing for Genomic Sequence Evolution
===================================================================
Core source package containing the full quantum simulation pipeline.

Modules
-------
gy94_model      : GY94 codon substitution rate matrix construction
hamiltonian     : Hermitization and Pauli decomposition
aae_encoding    : Approximate Amplitude Encoding (brickwall ansatz)
block_encoding  : LCU block encoding for QSP
qsp_angles      : QSP phase angle computation via pyqsp
qsp_circuit     : Full QSP circuit assembly and simulation
trotter         : Trotterized time evolution (alternative to QSP)
experiment      : End-to-end experiment runner
"""
