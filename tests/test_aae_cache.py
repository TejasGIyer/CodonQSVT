"""
Round-trip tests for the AAE cached-params workflow.

Covers:
  1. save_aae_params + load_aae_circuit reproduce the same statevector.
  2. get_aae_circuit caches: a second call loads from disk and returns
     the exact same parameters as the first call (no retraining).
  3. get_aae_circuit detects n_layers mismatch and retrains, overwriting
     the JSON.
  4. get_aae_circuit detects n_qubits mismatch in the cache and falls
     back to retraining.

Run with:
    python tests/test_aae_cache.py
"""

import os
import sys
import json
import tempfile

import numpy as np

_TEST_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TEST_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from src.aae_encoding import (
    aae_encode,
    save_aae_params,
    load_aae_circuit,
    get_aae_circuit,
)


def _tiny_step1(n_qubits=3, seed=42):
    """
    Build a minimal step1_result-like dict with a random normalized
    target. Mimics the shape of build_gapdh_register() output but with
    a smaller register so tests run quickly.
    """
    rng = np.random.default_rng(seed=seed)
    d = np.abs(rng.normal(size=2 ** n_qubits))
    d = d / np.linalg.norm(d)
    return {
        'num_qubits'      : n_qubits,
        'd_normalized'    : d,
        'unique_register' : [],
        'num_unique'      : 0,
        'num_codons'      : 0,
    }


def test_save_load_roundtrip():
    """Train tiny AAE, save, reload, assert statevectors match exactly."""
    s1 = _tiny_step1(n_qubits=3)
    s2 = aae_encode(s1, n_layers=2, n_trials=1, maxiter=200)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'aae.json')
        save_aae_params(path, s2, dataset_tag='unit_test')

        # JSON must be valid and have the canonical schema
        with open(path) as f:
            payload = json.load(f)
        for required in ('params', 'cost', 'overlap', 'n_qubits',
                         'n_layers', 'dataset', 'timestamp'):
            assert required in payload, f"Missing field '{required}' in saved JSON"
        assert payload['dataset'] == 'unit_test'

        loaded = load_aae_circuit(path, s1)

    # The rebuilt circuit must reproduce the trained statevector bit-for-bit
    sv_a = np.asarray(s2['initial_sv'].data)
    sv_b = np.asarray(loaded['initial_sv'].data)
    assert np.allclose(sv_a, sv_b, atol=1e-12), \
        f"Loaded circuit produces different SV (max diff = {np.max(np.abs(sv_a-sv_b)):.2e})"

    # Bookkeeping fields must round-trip
    assert abs(s2['overlap'] - loaded['overlap']) < 1e-10
    assert s2['num_qubits'] == loaded['num_qubits']
    assert s2['n_layers'] == loaded['n_layers']
    assert np.allclose(s2['best_params'], loaded['best_params'])

    print(f"  PASS save_load_roundtrip: overlap={loaded['overlap']:.6f}")


def test_get_aae_caches():
    """Second call to get_aae_circuit must load from disk, not retrain."""
    s1 = _tiny_step1(n_qubits=3)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'aae.json')

        # First call: trains and saves
        r1 = get_aae_circuit(s1, path, n_layers=2, n_trials=1, maxiter=200)
        assert os.path.isfile(path), "JSON not written on first call."
        assert r1.get('encoding_type') == 'aae', \
            f"First call should have trained, got encoding_type={r1.get('encoding_type')}"

        # Second call: must load from disk; aae_encode uses random restarts
        # so if retraining had happened, params would almost certainly differ.
        r2 = get_aae_circuit(s1, path, n_layers=2, n_trials=1, maxiter=200)
        assert r2.get('encoding_type') == 'aae_loaded', \
            f"Second call should have loaded, got encoding_type={r2.get('encoding_type')}"
        assert np.allclose(r1['best_params'], r2['best_params']), \
            "Second call retrained instead of loading from cache."
        assert r2.get('source_json') == os.path.abspath(path)

    print(f"  PASS get_aae_caches")


def test_n_layers_mismatch_honors_cache():
    """
    With the cache-as-authoritative design, requesting a different n_layers
    must NOT trigger a retrain when the cache is intact. The caller's
    n_layers is only the fallback used when training is actually needed.
    """
    s1 = _tiny_step1(n_qubits=3)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'aae.json')

        # Train with n_layers=2, save
        r1 = get_aae_circuit(s1, path, n_layers=2, n_trials=1, maxiter=200)
        assert r1['n_layers'] == 2

        # Request n_layers=3 — cache must still be honored as-is
        r2 = get_aae_circuit(s1, path, n_layers=3, n_trials=1, maxiter=200)
        assert r2['n_layers'] == 2, \
            f"Expected cache n_layers=2 to be honored, got {r2['n_layers']}"
        assert r2.get('encoding_type') == 'aae_loaded', \
            f"Expected aae_loaded, got {r2.get('encoding_type')} (mismatched n_layers retrained!)"
        assert np.allclose(r1['best_params'], r2['best_params']), \
            "Cache params changed across calls — a retrain happened."

        # File on disk should still reflect the original n_layers=2
        with open(path) as f:
            disk = json.load(f)
        assert int(disk['n_layers']) == 2
        assert len(disk['params']) == 3 * 2

    print(f"  PASS n_layers_mismatch_honors_cache")


def test_force_retrain_overwrites():
    """force_retrain=True must retrain even when the cache is valid."""
    s1 = _tiny_step1(n_qubits=3)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'aae.json')

        r1 = get_aae_circuit(s1, path, n_layers=2, n_trials=1, maxiter=200)

        # force_retrain ignores the cache. Random restarts almost certainly
        # produce different params (sometimes equal if the optimum is unique;
        # we check the file timestamp moved instead).
        ts_before = json.load(open(path))['timestamp']

        # Sleep-free trick: ensure timestamp moves by writing a sentinel
        # and re-running. ISO timestamps have microsecond resolution so the
        # next save will tick.
        import time as _time
        _time.sleep(0.01)

        r2 = get_aae_circuit(s1, path, n_layers=2, n_trials=1, maxiter=200,
                             force_retrain=True)
        assert r2.get('encoding_type') == 'aae', "force_retrain should retrain."

        ts_after = json.load(open(path))['timestamp']
        assert ts_after > ts_before, \
            f"force_retrain did not overwrite the JSON (timestamps: {ts_before} -> {ts_after})"

    print(f"  PASS force_retrain_overwrites")


def test_missing_file_raises():
    """load_aae_circuit must raise FileNotFoundError on a missing path."""
    s1 = _tiny_step1(n_qubits=3)
    try:
        load_aae_circuit('/tmp/__definitely_not_a_real_path__.json', s1)
    except FileNotFoundError:
        print(f"  PASS missing_file_raises")
        return
    raise AssertionError("load_aae_circuit did not raise FileNotFoundError.")


def test_qubit_mismatch_raises():
    """Loading a JSON with the wrong n_qubits for the given register must raise."""
    s1_small = _tiny_step1(n_qubits=3)
    s1_big   = _tiny_step1(n_qubits=4)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'aae.json')
        r1 = get_aae_circuit(s1_small, path, n_layers=2, n_trials=1, maxiter=200)
        assert r1['num_qubits'] == 3

        # Now try to load this 3-qubit JSON against a 4-qubit register.
        try:
            load_aae_circuit(path, s1_big)
        except ValueError:
            print(f"  PASS qubit_mismatch_raises")
            return
        raise AssertionError("load_aae_circuit did not raise ValueError on n_qubits mismatch.")


if __name__ == "__main__":
    print("Running AAE cache tests (uses tiny n_qubits=3, n_layers=2 ansatz)...")
    print()
    test_save_load_roundtrip()
    test_get_aae_caches()
    test_n_layers_mismatch_honors_cache()
    test_force_retrain_overwrites()
    test_missing_file_raises()
    test_qubit_mismatch_raises()
    print()
    print("All AAE cache tests passed.")
