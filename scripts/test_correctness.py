#!/usr/bin/env python3
"""
Correctness tests for Assignment 2 (run BEFORE any benchmark).

Dense methods  ijk (baseline) / ikj / blocked(bs)      sparse methods  csr / csrfile

Dense tests    scalar, 2x2 (non-commutative), hand-computed 3x3, zero matrices, identity on the
               left and on the right, all-ones, small integer matrices (exact), random matrices
               versus NumPy (n not a multiple of the tile size included), bit-for-bit equality of
               all dense methods, blocked with many tile sizes (1 ... larger than n).
Sparse tests   empty matrices, identity, diagonal, empty rows/columns, fully dense matrix,
               random uniform and banded matrices at several densities versus scipy.sparse
               (identical nonzero pattern, relative error within the a-priori bound), sorted
               column indices, csr (dense input) == csrfile (CSR input), densified sparse result
               bit-for-bit equal to the dense baseline on the same logical matrices, spot-check
               rows (ROWCHK) consistent with the dumped result.

Tolerance for random data: all entries are non-negative, so for any summation order
|c_hat - c| <= gamma_m * c with m <= n terms; two different orders differ by at most
2*gamma_n*c (triangle inequality), gamma_n = n*u/(1-n*u), u = 2**-53. The accepted relative
error is rtol = 2*gamma_n with atol = 0; exact zeros must be reproduced exactly.
Integer-valued test matrices use small values, so every intermediate result is exactly
representable and the comparison is exact (no tolerance).

Output: results/correctness_tests.csv (one row per check). Exit status 1 if any check fails.
"""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_inputs as gi  # noqa: E402

ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, "build", "matmul2")
U = 2.0 ** -53
rows = []


def gamma(k):
    return k * U / (1.0 - k * U)


def record(test, method, n, bs, ok, err=None, note=""):
    rows.append(dict(test=test, method=method, n=n, block_size=bs if bs else "", result="pass" if ok else "FAIL",
                     max_rel_err="" if err is None else "%.3e" % err, note=note))
    if not ok:
        print("FAIL:", test, method, n, bs, err, note, flush=True)


def write_dense(path, M):
    np.ascontiguousarray(M, dtype="<f8").tofile(path)


def write_csr_from(path, M):
    S = sp.csr_matrix(M)
    S.sort_indices()
    gi.write_csr(path, M.shape[0], S.indptr.astype(np.int64), S.indices.astype(np.int32), S.data.astype(np.float64))


def run(method, n, a, b, bs=None, dump=True, reps=1, warmup=0, batch=1):
    out = tempfile.mktemp(prefix="c2_", suffix=".dump") if dump else "-"
    cmd = [EXE, method, str(n), str(reps), str(warmup), str(batch), a, b, out] + ([str(bs)] if bs else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or "DONE" not in p.stdout:
        raise RuntimeError("run failed: %s\n%s" % (" ".join(cmd), p.stderr))
    res = {"checksum": None, "nnz_c": None, "dump": out if dump else None, "rowchk": [], "nnz_in": None,
           "all_checksums": []}
    for ln in p.stdout.splitlines():
        t = ln.split()
        if t and t[0] in ("REP", "WARMUP"):
            res["all_checksums"].append(float(t[3]))
        if t and t[0] == "REP":
            res["checksum"] = float(t[3])
            for kv in t[4:]:
                if kv.startswith("nnz_c="):
                    res["nnz_c"] = int(kv[6:])
        elif t and t[0] == "ROWCHK":
            res["rowchk"].append((int(t[1]), int(t[2]), float(t[3])))
        elif t and t[0] == "NNZ":
            res["nnz_in"] = (int(t[1]), int(t[2]))
    return res


def read_dense_dump(path, n):
    M = np.fromfile(path, dtype="<f8").reshape(n, n)
    os.unlink(path)
    return M


def read_csr_dump(path):
    n, ptr, idx, val = gi.read_csr(path)
    os.unlink(path)
    return sp.csr_matrix((val, idx, ptr), shape=(n, n))


def rel_err(C, R):
    """max relative error where R != 0; exact zeros must match exactly (returns inf otherwise)."""
    zero = (R == 0)
    if np.any(C[zero] != 0):
        return float("inf")
    nz = ~zero
    if not nz.any():
        return 0.0
    return float(np.max(np.abs(C[nz] - R[nz]) / np.abs(R[nz])))


DENSE_METHODS = [("ijk", None), ("ikj", None), ("blocked", 1), ("blocked", 3), ("blocked", 8), ("blocked", 32),
                 ("blocked", 1000)]


def dense_case(tmp, name, A, B, exact_ref=None, rtol=None):
    n = A.shape[0]
    pa, pb = os.path.join(tmp, "A.bin"), os.path.join(tmp, "B.bin")
    write_dense(pa, A); write_dense(pb, B)
    ref = exact_ref if exact_ref is not None else A @ B
    first = None
    for m, bs in DENSE_METHODS:
        C = read_dense_dump(run(m, n, pa, pb, bs)["dump"], n)
        if rtol is None:
            ok, err = bool(np.array_equal(C, ref)), None
        else:
            err = rel_err(C, ref)
            ok = err <= rtol
        record(name, m, n, bs, ok, err)
        if first is None:
            first = C
        else:
            record(name + " [bitwise vs ijk]", m, n, bs, bool(np.array_equal(C, first)))
    return first


def dense_tests(tmp):
    rng = np.random.default_rng(12345)
    dense_case(tmp, "scalar 1x1 (3*4=12)", np.array([[3.0]]), np.array([[4.0]]), exact_ref=np.array([[12.0]]))
    A2 = np.array([[1.0, 2.0], [3.0, 4.0]]); B2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    dense_case(tmp, "2x2 non-commutative", A2, B2, exact_ref=np.array([[2.0, 1.0], [4.0, 3.0]]))
    A3 = np.arange(1.0, 10.0).reshape(3, 3); B3 = A3[::-1, ::-1].copy()
    dense_case(tmp, "3x3 hand-computed", A3, B3,
               exact_ref=np.array([[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]]))
    Z = np.zeros((9, 9)); R9 = rng.random((9, 9))
    dense_case(tmp, "zero x random", Z, R9, exact_ref=Z)
    dense_case(tmp, "random x zero", R9, Z, exact_ref=Z)
    I = np.eye(9)
    dense_case(tmp, "identity x random (left)", I, R9, exact_ref=R9)
    dense_case(tmp, "random x identity (right)", R9, I, exact_ref=R9)
    O = np.ones((7, 7))
    dense_case(tmp, "all-ones 7x7 (every entry 7)", O, O, exact_ref=np.full((7, 7), 7.0))
    Ai = rng.integers(-5, 6, (13, 13)).astype(np.float64); Bi = rng.integers(-5, 6, (13, 13)).astype(np.float64)
    dense_case(tmp, "small-integer matrices (exact)", Ai, Bi, exact_ref=(Ai.astype(np.int64) @ Bi.astype(np.int64)).astype(np.float64))
    for n in (5, 17, 33, 64, 100, 129):
        A, B = rng.random((n, n)), rng.random((n, n))
        dense_case(tmp, "random uniform vs NumPy", A, B, rtol=2 * gamma(n))
    # blocked with many tile sizes must be bit-for-bit equal to the baseline
    n = 100
    A, B = rng.random((n, n)), rng.random((n, n))
    pa, pb = os.path.join(tmp, "A.bin"), os.path.join(tmp, "B.bin")
    write_dense(pa, A); write_dense(pb, B)
    base = read_dense_dump(run("ijk", n, pa, pb)["dump"], n)
    for bs in (1, 2, 3, 7, 16, 32, 48, 64, 99, 100, 101, 128, 4096):
        C = read_dense_dump(run("blocked", n, pa, pb, bs)["dump"], n)
        record("blocked tile-size sweep [bitwise vs ijk]", "blocked", n, bs, bool(np.array_equal(C, base)))


def repeatability_tests(tmp):
    """A kernel must define every element of C itself: repeating the multiplication into the
    same output buffer (warm-up + repetitions, as in the benchmark) must give identical results."""
    rng = np.random.default_rng(99)
    n = 40
    A, B = rng.random((n, n)), rng.random((n, n))
    pa, pb = os.path.join(tmp, "A.bin"), os.path.join(tmp, "B.bin")
    write_dense(pa, A); write_dense(pb, B)
    for m, bs in DENSE_METHODS:
        r = run(m, n, pa, pb, bs, dump=False, reps=4, warmup=2, batch=3)
        record("repeated runs give identical checksum (reps=4, warmup=2, batch=3)", m, n, bs,
               len(r["all_checksums"]) == 6 and len(set(r["all_checksums"])) == 1)
    S = sp.random(n, n, density=0.2, random_state=3, format="csr")
    write_dense(pa, S.toarray())
    for m in ("csr",):
        r = run(m, n, pa, pa, dump=False, reps=4, warmup=2, batch=3)
        record("repeated runs give identical checksum (reps=4, warmup=2, batch=3)", m, n, None,
               len(r["all_checksums"]) == 6 and len(set(r["all_checksums"])) == 1)


def sparse_case(tmp, name, A, B, check_dense=True):
    """A, B: scipy CSR or ndarray. Runs csr (dense input) and csrfile, checks both."""
    A = sp.csr_matrix(A); B = sp.csr_matrix(B)
    A.sort_indices(); B.sort_indices()
    n = A.shape[0]
    pa, pb = os.path.join(tmp, "SA.bin"), os.path.join(tmp, "SB.bin")
    write_dense(pa, A.toarray()); write_dense(pb, B.toarray())
    ca, cb = os.path.join(tmp, "SA.csr"), os.path.join(tmp, "SB.csr")
    write_csr_from(ca, A); write_csr_from(cb, B)
    ref = (A @ B).tocsr(); ref.sort_indices(); ref.eliminate_zeros()
    r1 = run("csr", n, pa, pb)
    C1 = read_csr_dump(r1["dump"])
    r2 = run("csrfile", n, ca, cb)
    C2 = read_csr_dump(r2["dump"])
    tol = 2 * gamma(n)
    # 1. nonzero pattern identical to scipy
    same_pat = (C1.nnz == ref.nnz and np.array_equal(C1.indptr, ref.indptr) and np.array_equal(C1.indices, ref.indices))
    record(name + " [pattern vs scipy]", "csr", n, None, bool(same_pat), None, "nnz_c=%d" % C1.nnz)
    # 2. values within tolerance
    if same_pat and ref.nnz:
        err = float(np.max(np.abs(C1.data - ref.data) / np.abs(ref.data)))
        record(name + " [values vs scipy]", "csr", n, None, err <= tol, err)
    else:
        record(name + " [values vs scipy]", "csr", n, None, bool(same_pat), 0.0)
    # 3. csrfile == csr (identical bytes of the result)
    same = (np.array_equal(C1.indptr, C2.indptr) and np.array_equal(C1.indices, C2.indices) and np.array_equal(C1.data, C2.data))
    record(name + " [csr == csrfile]", "csrfile", n, None, bool(same))
    # 4. column indices strictly increasing inside every row
    ok_sorted = True
    for i in range(n):
        row = C1.indices[C1.indptr[i]:C1.indptr[i + 1]]
        if row.size > 1 and not np.all(np.diff(row) > 0):
            ok_sorted = False
            break
    record(name + " [sorted unique columns]", "csr", n, None, ok_sorted)
    # 5. input nonzero counts and output count reported by the program
    record(name + " [nnz reported]", "csr", n, None,
           (r1["nnz_in"] == (A.nnz, B.nnz) and r1["nnz_c"] == C1.nnz), None,
           "A=%d B=%d C=%d" % (A.nnz, B.nnz, C1.nnz))
    # 6. ROWCHK rows consistent with the dumped result
    ok_rc = True
    for (i, cnt, s) in r1["rowchk"]:
        row = C1.data[C1.indptr[i]:C1.indptr[i + 1]]
        s_ref = 0.0
        for v in row:
            s_ref += float(v)
        if cnt != row.size or s != s_ref:
            ok_rc = False
    record(name + " [ROWCHK consistent]", "csr", n, None, ok_rc and len(r1["rowchk"]) == 64)
    # 7. densified sparse result bit-for-bit equal to the dense baseline on the same matrices
    if check_dense:
        Cd = read_dense_dump(run("ijk", n, pa, pb)["dump"], n)
        record(name + " [densified == dense ijk bitwise]", "csr", n, None, bool(np.array_equal(C1.toarray(), Cd)))
        Cd2 = read_dense_dump(run("blocked", n, pa, pb, 32)["dump"], n)
        record(name + " [densified == dense blocked bitwise]", "csr", n, None, bool(np.array_equal(C1.toarray(), Cd2)))


def sparse_tests(tmp):
    rng = np.random.default_rng(777)
    n = 16
    sparse_case(tmp, "empty x empty", sp.csr_matrix((n, n)), sp.csr_matrix((n, n)))
    R = sp.random(n, n, density=0.3, random_state=1, format="csr")
    R.data = 1.0 - R.data * 0.5
    sparse_case(tmp, "empty x random", sp.csr_matrix((n, n)), R)
    sparse_case(tmp, "identity x random", sp.identity(n, format="csr"), R)
    sparse_case(tmp, "random x identity", R, sp.identity(n, format="csr"))
    D = sp.diags(rng.random(n) + 0.5, format="csr")
    sparse_case(tmp, "diagonal x diagonal", D, D)
    E = R.tolil(); E[3, :] = 0; E[:, 5] = 0; E = E.tocsr()
    sparse_case(tmp, "empty row and column", E, R)
    sparse_case(tmp, "1x1", sp.csr_matrix(np.array([[2.5]])), sp.csr_matrix(np.array([[4.0]])))
    full = rng.random((24, 24)) + 0.1
    sparse_case(tmp, "fully dense 24x24", full, full[::-1].copy())
    for n in (64, 200):
        for d in (1, 10, 50):
            for struct in ("unif", "band"):
                mats = []
                for which in (0, 1):
                    ptr, idx, val = gi.gen_csr(n, d, struct, which)
                    mats.append(sp.csr_matrix((val, idx, ptr), shape=(n, n)))
                sparse_case(tmp, "%s n=%d d=%g%%" % (struct, n, d), mats[0], mats[1])
    # conversion nnz reported equals the true count (dense input path)
    n = 50
    ptr, idx, val = gi.gen_csr(n, 10, "unif", 0)
    M = sp.csr_matrix((val, idx, ptr), shape=(n, n))
    pa = os.path.join(tmp, "x.bin"); write_dense(pa, M.toarray())
    r = run("csr", n, pa, pa, dump=False)
    record("dense->CSR conversion nnz", "csr", n, None, r["nnz_in"] == (M.nnz, M.nnz), None, "nnz=%d" % M.nnz)


def main():
    if not os.path.exists(EXE):
        sys.exit("build first: make build")
    with tempfile.TemporaryDirectory() as tmp:
        dense_tests(tmp)
        repeatability_tests(tmp)
        sparse_tests(tmp)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "correctness_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if r["result"] != "pass"]
    print("%d checks, %d failed" % (len(rows), len(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
