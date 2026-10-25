#!/usr/bin/env python3
"""Generates the input matrices (dense = the Assignment 1 files; sparse = random 'unif' or 'band', seeded)
and checks every file against the SHA-256 digests in data/manifest.json.
Usage: gen_inputs.py dense n1 n2 ...  |  gen_inputs.py sparse --sizes ... --dens ... --structs unif band [--dense]"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

SEED = 20261012
STRUCT_CODE = {"unif": 1, "band": 2}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dens_tag(d):
    return ("%g" % d)


def sparse_name(which, struct, n, d, ext):
    return "%s_%s_n%d_d%s.%s" % ("AB"[which], struct, n, dens_tag(d), ext)


def band_nnz(n, bw):
    """Number of entries of a centred band of width bw in an n x n matrix (clipped at the border)."""
    i = np.arange(n, dtype=np.int64)
    lo = np.maximum(0, i - (bw - 1) // 2)
    hi = np.minimum(n - 1, i + bw // 2)
    return int((hi - lo + 1).sum())


def band_width(n, frac):
    """Smallest band width whose clipped band has the entry count closest to frac*n*n."""
    target = frac * n * n
    lo, hi = 1, 2 * n - 1
    while lo < hi:                      # band_nnz is non-decreasing in bw
        mid = (lo + hi) // 2
        if band_nnz(n, mid) < target:
            lo = mid + 1
        else:
            hi = mid
    best = lo
    if best > 1 and abs(band_nnz(n, best - 1) - target) < abs(band_nnz(n, best) - target):
        best -= 1
    return best


def gen_csr(n, d, struct, which):
    """Return (ptr int64[n+1], idx int32[nnz], val float64[nnz]) of one sparse matrix."""
    rng = np.random.default_rng([SEED, n, which, STRUCT_CODE[struct], int(round(10 * d))])
    frac = d / 100.0
    ptr = np.zeros(n + 1, dtype=np.int64)
    idx_parts = []
    if struct == "unif":
        counts = rng.binomial(n, frac, size=n)
        for i in range(n):
            k = int(counts[i])
            if k:
                cols = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int32)
            else:
                cols = np.empty(0, dtype=np.int32)
            idx_parts.append(cols)
            ptr[i + 1] = ptr[i] + k
    else:                                     # band
        bw = band_width(n, frac)
        lo_off, hi_off = (bw - 1) // 2, bw // 2
        for i in range(n):
            lo, hi = max(0, i - lo_off), min(n - 1, i + hi_off)
            cols = np.arange(lo, hi + 1, dtype=np.int32)
            idx_parts.append(cols)
            ptr[i + 1] = ptr[i] + cols.size
    idx = np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=np.int32)
    nnz = int(ptr[-1])
    val = 1.0 - rng.random(nnz, dtype=np.float64)
    return ptr, idx.astype(np.int32), val


def write_csr(path, n, ptr, idx, val):
    with open(path, "wb") as f:
        f.write(b"CSR1")
        np.array([n, idx.size], dtype="<i8").tofile(f)
        ptr.astype("<i8").tofile(f)
        idx.astype("<i4").tofile(f)
        val.astype("<f8").tofile(f)


def read_csr(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"CSR1"
        n, nnz = np.fromfile(f, dtype="<i8", count=2)
        ptr = np.fromfile(f, dtype="<i8", count=n + 1)
        idx = np.fromfile(f, dtype="<i4", count=nnz)
        val = np.fromfile(f, dtype="<f8", count=nnz)
    return int(n), ptr, idx, val


def load_manifest():
    mpath = os.path.join(DATA, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            return json.load(f)
    return {"generator": "numpy.random.default_rng(PCG64), seed list [seed, n, which, (struct, 10*density)]",
            "numpy_version": np.__version__, "seed": SEED, "files": {}}


def record(manifest, path, extra, stats):
    base = os.path.basename(path)
    entry = dict(extra, sha256=sha256_file(path), bytes=os.path.getsize(path))
    old = manifest["files"].get(base)
    if old is None:
        manifest["files"][base] = entry
        stats["new"] += 1
    elif old["sha256"] == entry["sha256"]:
        stats["verified"] += 1
    else:
        stats["mismatch"].append(base)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    d = sub.add_parser("dense")
    d.add_argument("sizes", nargs="+", type=int)
    s = sub.add_parser("sparse")
    s.add_argument("--sizes", nargs="+", type=int, required=True)
    s.add_argument("--dens", nargs="+", type=float, required=True, help="percent of nonzero entries")
    s.add_argument("--structs", nargs="+", choices=list(STRUCT_CODE), default=["unif"])
    s.add_argument("--dense", action="store_true", help="also write the dense .bin form")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    manifest = load_manifest()
    stats = {"new": 0, "verified": 0, "mismatch": []}

    if args.kind == "dense":
        ref_path = os.path.join(DATA, "a1_reference_sha256.json")
        a1 = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
        for n in args.sizes:
            for which, name in ((0, "A"), (1, "B")):
                path = os.path.join(DATA, "%s_n%d.bin" % (name, n))
                rng = np.random.default_rng([SEED, n, which])
                rng.random(n * n, dtype=np.float64).tofile(path)
                record(manifest, path, {"n": n, "kind": "dense-uniform"}, stats)
                base = os.path.basename(path)
                if base in a1:
                    if a1[base] == manifest["files"][base]["sha256"]:
                        manifest["files"][base]["identical_to_assignment1"] = True
                    else:
                        stats["mismatch"].append(base + " (differs from Assignment 1)")
            print("dense n=%d done" % n, flush=True)
    else:
        for n in args.sizes:
            for dd in args.dens:
                for struct in args.structs:
                    mats = []
                    for which in (0, 1):
                        ptr, idx, val = gen_csr(n, dd, struct, which)
                        cpath = os.path.join(DATA, sparse_name(which, struct, n, dd, "csr"))
                        write_csr(cpath, n, ptr, idx, val)
                        extra = {"n": n, "kind": "sparse-" + struct, "density_pct_target": dd,
                                 "nnz": int(idx.size), "density_pct_actual": 100.0 * idx.size / (n * n)}
                        record(manifest, cpath, extra, stats)
                        if args.dense:
                            M = sp.csr_matrix((val, idx, ptr), shape=(n, n))
                            bpath = os.path.join(DATA, sparse_name(which, struct, n, dd, "bin"))
                            M.toarray().astype("<f8").tofile(bpath)
                            record(manifest, bpath, extra, stats)
                    print("sparse n=%d d=%g%% %s done" % (n, dd, struct), flush=True)

    if not stats["mismatch"]:
        with open(os.path.join(DATA, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    print("inputs: %d verified identical to manifest, %d newly recorded, %d DIFFERENT"
          % (stats["verified"], stats["new"], len(stats["mismatch"])))
    if stats["mismatch"]:
        print("WARNING: these files do not match the reference digests:", ", ".join(stats["mismatch"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
