#!/usr/bin/env python3
"""
Benchmark driver for Individual Assignment 2 (optimised dense methods and sparse CSR).

One *run* = one operating-system process executing one (method, input) configuration:
warm-up samples (discarded from the statistics, kept in the CSV) followed by measured samples.
Every sample is the time of ONE multiplication (for tiny inputs a batch of multiplications is timed
and divided by the batch size). The protocol is the one of Assignment 1:

  budgets   T_MULT  300 s  : maximum wall time of one timed sample; longer -> run killed, status "timeout"
            MEM_KIB 4 GiB  : maximum resident memory (polled every 0.5 s, plus RLIMIT_AS); exceeded ->
                             status "memory_limit". Both are results, no execution time is invented.
  repeats   t_est <= 1 s : 3 warm-up + 10 measured | <= 30 s : 1 + 5 | <= 120 s : 0 + 3 | else 0 + 2
  batching  t_est < 50 ms : a batch of ceil(0.1 s / t_est) multiplications is timed and divided by the batch
  order     size-major (for each size every configuration), so slow machine drift hits all methods alike
  failures  a series (same method/structure/density, growing n) stops after its first timeout / memory failure

Campaigns
  dense    baseline ijk, ikj, blocked on the Assignment 1 uniform inputs, declared size sequence   (Part A)
  bsweep   tile-size sweep of the blocked method at n = 1500 (a size NOT used in the comparison)     (Part A)
  flags    SUPPLEMENTARY: compiler options (-O2, -O2 -funroll-loops, -O3, -O3 -march=native)         (Part A)
  procvar  SUPPLEMENTARY: between-process variability at n = 1000 (run with --tag 1..k)
  sparse   CSR x CSR (with the dense->CSR conversion timed separately) and dense methods on MATCHED
           logical inputs: sizes x structures x densities                                            (Part B)
  scale    sparse-only scaling with CSR inputs until the time / memory budget is reached              (Part B)

Correctness (every run): the final product is dumped and checked against NumPy (dense) or SciPy (sparse);
accepted relative error 2*gamma_n, gamma_n = n*u/(1-n*u), u = 2**-53 (entries are non-negative, so the error of
any summation order is <= gamma_n * c). Exact zeros must be reproduced exactly; the nonzero pattern of the sparse
result must equal SciPy's. Spot-check rows (ROWCHK) are verified for every sparse run, also when no full dump exists.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import resource
import selectors
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen_inputs as gi  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
U = 2.0 ** -53
T_MULT = 300.0               # seconds per timed sample
MEM_KIB = 4 * 1024 * 1024    # 4 GiB
AS_LIMIT = MEM_KIB * 1024 + 256 * 1024 * 1024   # address-space cap: malloc fails cleanly near the budget

SIZES_DENSE = [100, 250, 500, 512, 750, 1000, 1024, 1250, 2000, 2048, 3000, 4000, 5000, 6000, 8000]
BS_SWEEP_N = 1500
BS_SWEEP = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
FLAG_SIZES = [1000, 2000]
PROCVAR_N = 1000
SPARSE_SIZES = [500, 1000, 2000]
DENSITIES = [0.1, 1, 5, 10, 20, 30, 50, 70, 90]   # percent of nonzero entries (sparsity 99.9 ... 10 %)
STRUCTS = ["unif", "band"]
IJK_AT_2000 = [("unif", 50), ("band", 0.1)]       # matched baseline runs at n = 2000 (cost: ~30 s each)
SCALE_DENS = {0.1: [2000, 4000, 8000, 16000, 32000, 64000, 128000], 1: [2000, 4000, 8000, 16000, 32000]}
FLAG_VARIANTS = ["O2", "O2-unroll", "O3", "O3-native"]
PRIOR = {"ijk": 2.5e-9, "ikj": 1.2e-9, "blocked": 1.0e-9}   # s per n^3 (pilot estimates)

RAW_FIELDS = ["campaign", "run_id", "config", "method", "numeric_type", "n", "structure", "density_pct",
              "block_size", "thread_count", "phase", "repetition", "batch_size", "time_per_mult_s",
              "conv_s", "nnz_c", "checksum"]
RUN_FIELDS = ["campaign", "run_id", "config", "method", "n", "structure", "density_pct", "block_size",
              "numeric_type", "thread_count", "cflags", "status", "reason", "reps_planned", "warmup_planned",
              "reps_completed", "warmup_completed", "batch_size", "t_est_s", "limit_sample_s", "limit_mem_kib",
              "wall_s", "rss_after_load_kib", "vmhwm_kib", "ru_maxrss_kib", "nnz_a", "nnz_b", "nnz_c",
              "predicted_bytes", "checksum_rel_err", "checksum_tol", "max_abs_err", "max_rel_err", "rtol_used",
              "correctness", "sha256_C", "start_utc", "returncode", "stderr_tail"]


# ----------------------------------------------------------------------------- helpers
def gamma(k):
    return k * U / (1.0 - k * U)


def plan_for(t_est):
    if t_est <= 1.0:
        return 10, 3
    if t_est <= 30.0:
        return 5, 1
    if t_est <= 120.0:
        return 3, 0
    return 2, 0


def batch_for(t_est, target=0.1):
    return 1 if t_est >= 0.05 else int(min(1_000_000, math.ceil(target / t_est)))


def exe_for(variant):
    return ROOT / "build" / ("matmul2" if variant == "O2" else "flags/matmul2_" + variant)


CFLAGS = {"O2": "-O2 -std=c11 -ffp-contract=off", "O2-unroll": "-O2 -funroll-loops -std=c11 -ffp-contract=off",
          "O3": "-O3 -std=c11 -ffp-contract=off", "O3-native": "-O3 -march=native -std=c11 -ffp-contract=off"}


def clean_env():
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    return env


def limit_child():
    resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))


def read_rss_kib(pid):
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def execute(cmd, t_limit, mem_kib):
    """Run one process; return (lines, status, reason, wall_s, ru_maxrss_kib, returncode, stderr_tail)."""
    errf = tempfile.TemporaryFile()
    t_start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, env=clean_env(),
                            start_new_session=True, preexec_fn=limit_child)
    fd = proc.stdout.fileno()
    os.set_blocking(fd, False)
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    buf, lines = b"", []
    last_progress = time.monotonic()
    status, reason, eof = "ok", "", False
    while not eof:
        for _ in sel.select(timeout=0.5):
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                eof = True
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                lines.append(line.decode(errors="replace").strip())
                last_progress = time.monotonic()
        if eof:
            break
        now = time.monotonic()
        if now - last_progress > t_limit:
            status, reason = "timeout", "no sample completed within %g s" % t_limit
        elif read_rss_kib(proc.pid) > mem_kib:
            status, reason = "memory_limit", "resident memory exceeded %d KiB" % mem_kib
        if status != "ok":
            os.killpg(proc.pid, signal.SIGKILL)
            break
    try:
        buf += os.read(fd, 1 << 20)
    except (BlockingIOError, OSError):
        pass
    for line in buf.decode(errors="replace").splitlines():
        if line.strip():
            lines.append(line.strip())
    _, wstatus, ru = os.wait4(proc.pid, 0)
    proc.returncode = 0
    wall = time.monotonic() - t_start
    rc = os.waitstatus_to_exitcode(wstatus)
    errf.seek(0)
    err_tail = errf.read()[-300:].decode(errors="replace").replace("\n", " | ")
    errf.close()
    if status == "ok" and (rc != 0 or "DONE" not in lines):
        status, reason = "error", "exit code %s" % rc
        if rc == 3 or "out of memory" in err_tail:
            status, reason = "memory_limit", "allocation failed at the %d MiB address-space cap" % (AS_LIMIT >> 20)
        elif rc == -9:
            status, reason = "memory_limit", "killed by SIGKILL (probable out-of-memory)"
    return lines, status, reason, wall, ru.ru_maxrss, rc, err_tail


def parse(lines):
    out = {"warmup": [], "rep": [], "rss_after_load": None, "vmhwm": None, "done": False,
           "nnz_in": None, "rowchk": []}
    for ln in lines:
        p = ln.split()
        if not p:
            continue
        if p[0] in ("WARMUP", "REP") and len(p) >= 4:
            kv = dict(x.split("=", 1) for x in p[4:] if "=" in x)
            rec = (int(p[1]), float(p[2]), float(p[3]), float(kv.get("conv_s", "nan")), int(kv.get("nnz_c", -1)))
            out["warmup" if p[0] == "WARMUP" else "rep"].append(rec)
        elif p[0] == "RSS_AFTER_LOAD_KB":
            out["rss_after_load"] = int(p[1])
        elif p[0] == "VMHWM_KB":
            out["vmhwm"] = int(p[1])
        elif p[0] == "NNZ":
            out["nnz_in"] = (int(p[1]), int(p[2]))
        elif p[0] == "ROWCHK":
            out["rowchk"].append((int(p[1]), int(p[2]), float(p[3])))
        elif p[0] == "DONE":
            out["done"] = True
    return out


# ----------------------------------------------------------------------------- validation
class DenseRef:
    """NumPy reference product of two dense input files (cached for the pair currently validated)."""
    def __init__(self):
        self.key = None

    def get(self, pa, pb, n):
        if self.key != (pa, pb):
            A = np.fromfile(pa, dtype="<f8").reshape(n, n)
            B = np.fromfile(pb, dtype="<f8").reshape(n, n)
            self.R = A @ B
            self.sum = float(np.sum(self.R, dtype=np.longdouble))
            self.key = (pa, pb)
            del A, B
        return self.R, self.sum


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dense(ref, job, dump, checksums):
    n = job["n"]
    R, Rsum = ref.get(job["a"], job["b"], n)
    C = np.memmap(dump, dtype="<f8", mode="r")
    if C.size != n * n:
        return dict(correctness="fail", reason="dump has wrong size")
    C = C.reshape(n, n)
    rtol = 2 * gamma(n)
    max_abs, max_rel, zero_ok = 0.0, 0.0, True
    for r0 in range(0, n, 256):
        c, r = np.asarray(C[r0:r0 + 256]), R[r0:r0 + 256]
        zero = (r == 0)
        if np.any(c[zero] != 0):
            zero_ok = False
        d = np.abs(c - r)
        max_abs = max(max_abs, float(d.max()))
        nz = ~zero
        if nz.any():
            max_rel = max(max_rel, float((d[nz] / np.abs(r[nz])).max()))
    ck_tol = gamma(n * n + 2 * n)
    ck_err = max(abs(c - Rsum) for c in checksums) / Rsum if checksums and Rsum else float("nan")
    ok = zero_ok and max_rel <= rtol and ck_err <= ck_tol and len(set(checksums)) == 1
    return dict(correctness="pass" if ok else "fail", max_abs_err=max_abs, max_rel_err=max_rel, rtol_used=rtol,
                checksum_rel_err=ck_err, checksum_tol=ck_tol, sha256_C=sha256_file(dump))


def load_inputs_csr(job):
    n = job["n"]
    if job["method"] == "csr":
        A = sp.csr_matrix(np.fromfile(job["a"], dtype="<f8").reshape(n, n))
        B = sp.csr_matrix(np.fromfile(job["b"], dtype="<f8").reshape(n, n))
    else:
        _, p, i, v = gi.read_csr(job["a"]); A = sp.csr_matrix((v, i, p), shape=(n, n))
        _, p, i, v = gi.read_csr(job["b"]); B = sp.csr_matrix((v, i, p), shape=(n, n))
    A.sort_indices(); B.sort_indices()
    return A, B


def validate_sparse(job, dump, res, checksums):
    n = job["n"]
    A, B = load_inputs_csr(job)
    rtol = 2 * gamma(n)
    out = dict(rtol_used=rtol)
    ok = len(set(checksums)) == 1
    # (a) spot-check rows against SciPy on the same rows
    rows = [r for r, _, _ in res["rowchk"]]
    rc_ok = len(rows) == 64
    if rows:
        sub = (A[rows] @ B).tocsr()
        sub.sort_indices(); sub.eliminate_zeros()
        for t, (i, cnt, s) in enumerate(res["rowchk"]):
            row = sub.data[sub.indptr[t]:sub.indptr[t + 1]]
            rs = float(np.sum(row, dtype=np.longdouble))
            if cnt != row.size or (rs and abs(s - rs) / rs > 4 * gamma(n)):
                rc_ok = False
    ok = ok and rc_ok
    # (b) full comparison when a dump exists
    if dump is not None and os.path.exists(dump):
        _, p, i, v = gi.read_csr(dump)
        C = sp.csr_matrix((v, i, p), shape=(n, n))
        ref = (A @ B).tocsr(); ref.sort_indices(); ref.eliminate_zeros()
        same = (C.nnz == ref.nnz and np.array_equal(C.indptr, ref.indptr) and np.array_equal(C.indices, ref.indices))
        rel = float(np.max(np.abs(C.data - ref.data) / np.abs(ref.data))) if same and ref.nnz else (0.0 if same else float("inf"))
        Rsum = float(np.sum(ref.data, dtype=np.longdouble))
        ck_tol = gamma(max(1, ref.nnz) + 2 * n)
        ck_err = max(abs(c - Rsum) for c in checksums) / Rsum if checksums and Rsum else 0.0
        ok = ok and same and rel <= rtol and ck_err <= ck_tol
        out.update(max_rel_err=rel, checksum_rel_err=ck_err, checksum_tol=ck_tol)
        if n <= 3000:   # dense form of the result: comparable with the sha256 of the dense methods
            out["sha256_C"] = hashlib.sha256(np.ascontiguousarray(C.toarray(), dtype="<f8").tobytes()).hexdigest()
    out["correctness"] = "pass" if ok else "fail"
    return out


# ----------------------------------------------------------------------------- jobs
def dense_files(n):
    return DATA / ("A_n%d.bin" % n), DATA / ("B_n%d.bin" % n)


def sparse_files(n, struct, d, ext):
    return (DATA / gi.sparse_name(0, struct, n, d, ext), DATA / gi.sparse_name(1, struct, n, d, ext))


def make_job(campaign, config, method, n, bs=0, variant="O2", struct="", dens=None, stop_on_fail=False):
    if method in ("ijk", "ikj", "blocked") and not struct:
        a, b = dense_files(n)
    elif method in ("ijk", "ikj", "blocked", "csr"):
        a, b = sparse_files(n, struct, dens, "bin")
    else:
        a, b = sparse_files(n, struct, dens, "csr")
    return dict(campaign=campaign, config=config, method=method, n=n, bs=bs, variant=variant, struct=struct,
                dens=dens, a=str(a), b=str(b), series=(config, struct, dens), stop_on_fail=stop_on_fail)


def predicted_bytes(job):
    n, f = job["n"], (job["dens"] or 100.0) / 100.0
    if job["method"] in ("ijk", "ikj", "blocked"):
        return 3 * 8 * n * n
    nnz_in = f * n * n
    nnz_c = n * n * (1.0 - math.exp(-n * f * f))
    csr = lambda z: 12 * z + 8 * (n + 1)
    base = 2 * csr(nnz_in) + csr(nnz_c) * 2.0          # x2: output growth by doubling
    return int(base + (2 * 8 * n * n if job["method"] == "csr" else 0))


def prior_time(job):
    n = job["n"]
    if job["method"] in PRIOR:
        return PRIOR[job["method"]] * n ** 3
    f = (job["dens"] or 100.0) / 100.0
    return 8e-9 * n ** 3 * f * f + 3e-9 * n * n * (1 + f * 10)


def job_list(args):
    c = args.campaign
    jobs = []
    if c == "dense":
        for n in (args.sizes or SIZES_DENSE):
            for m in ("ijk", "ikj", "blocked"):
                jobs.append(make_job(c, m, m, n, bs=args.bs if m == "blocked" else 0, stop_on_fail=True))
    elif c == "bsweep":
        for bs in BS_SWEEP:
            jobs.append(make_job(c, "blocked-bs%d" % bs, "blocked", BS_SWEEP_N, bs=bs))
    elif c == "flags":
        for n in (args.sizes or FLAG_SIZES):
            for v in FLAG_VARIANTS:
                for m in ("ijk", "ikj", "blocked"):
                    jobs.append(make_job(c, "%s-%s" % (m, v), m, n, bs=args.bs if m == "blocked" else 0, variant=v))
    elif c == "procvar":
        for m in ("ijk", "ikj", "blocked"):
            jobs.append(make_job(c, m, m, args.sizes[0] if args.sizes else PROCVAR_N, bs=args.bs if m == "blocked" else 0))
    elif c == "sparse":
        for n in (args.sizes or SPARSE_SIZES):
            for struct in STRUCTS:
                for d in DENSITIES:
                    jobs.append(make_job(c, "csr", "csr", n, struct=struct, dens=d))
                    jobs.append(make_job(c, "ikj", "ikj", n, struct=struct, dens=d))
                    jobs.append(make_job(c, "blocked", "blocked", n, bs=args.bs, struct=struct, dens=d))
                    if n <= 1000 or (struct, d) in IJK_AT_2000:
                        jobs.append(make_job(c, "ijk", "ijk", n, struct=struct, dens=d))
    elif c == "scale":
        allsizes = sorted({n for v in SCALE_DENS.values() for n in v})
        for n in (args.sizes or allsizes):
            for d, sizes in SCALE_DENS.items():
                if n in sizes:
                    jobs.append(make_job(c, "csrfile", "csrfile", n, struct="unif", dens=d, stop_on_fail=True))
    return jobs


# ----------------------------------------------------------------------------- engine
def append_csv(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def done_keys(path):
    keys = set()
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                keys.add((r["campaign"], r["config"], int(r["n"]), r["structure"], r["density_pct"]))
    return keys


def key_of(job):
    return (job["campaign"], job["config"], job["n"], job["struct"], "" if job["dens"] is None else "%g" % job["dens"])


def run_jobs(jobs, args, name):
    RESULTS.mkdir(exist_ok=True)
    runs_csv, raw_csv = RESULTS / "runs.csv", RESULTS / "raw_measurements.csv"
    finished = done_keys(runs_csv) if args.resume else set()
    ref = DenseRef()
    tmp = Path(tempfile.mkdtemp(prefix="matmul2_dump_"))
    state = {}
    for job in jobs:
        n, key = job["n"], key_of(job)
        prev_n, prev_t, stop_reason = state.get(job["series"], [None, None, None])
        if key in finished:
            with open(raw_csv, newline="") as f:
                ts = [float(r["time_per_mult_s"]) for r in csv.DictReader(f)
                      if (r["campaign"], r["config"], int(r["n"]), r["structure"], r["density_pct"]) == key
                      and r["phase"] == "measured"]
            if ts:
                prev_n, prev_t = n, statistics.median(ts)
            state[job["series"]] = [prev_n, prev_t, stop_reason]
            continue
        t_est = prev_t * (n / prev_n) ** 3 * 1.3 if prev_t else prior_time(job)
        reps, warm = plan_for(t_est)
        batch = batch_for(t_est)
        dens = job["dens"]
        run_id = "%s-%s-n%d%s" % (name, job["config"], n, "" if not job["struct"] else "-%s-d%g" % (job["struct"], dens))
        base = dict(campaign=name, run_id=run_id, config=job["config"], method=job["method"], n=n,
                    structure=job["struct"], density_pct="" if dens is None else "%g" % dens,
                    block_size=job["bs"] or "", numeric_type="float64", thread_count=1,
                    cflags=CFLAGS[job["variant"]], reps_planned=reps, warmup_planned=warm, batch_size=batch,
                    t_est_s="%.6g" % t_est, limit_sample_s=T_MULT, limit_mem_kib=MEM_KIB,
                    predicted_bytes=predicted_bytes(job),
                    start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
        pred_ok = predicted_bytes(job) <= 3 * MEM_KIB * 1024
        if stop_reason or t_est > 3 * T_MULT or not pred_ok:
            why = stop_reason or ("predicted %.0f s per multiplication exceeds 3x the %g s limit" % (t_est, T_MULT)
                                  if t_est > 3 * T_MULT else "predicted memory exceeds 3x the %d GiB budget" % (MEM_KIB >> 20))
            append_csv(runs_csv, RUN_FIELDS, [dict(base, status="not_attempted", reason=why, reps_completed=0,
                                                   warmup_completed=0)])
            print("[%s] %-16s n=%-6d %s not attempted (%s)" % (name, job["config"], n, job["struct"], why), flush=True)
            continue
        for p in (job["a"], job["b"]):
            if not os.path.exists(p):
                sys.exit("missing input %s - run scripts/gen_inputs.py first" % p)
        sparse_m = job["method"] in ("csr", "csrfile")
        want_dump = True
        if job["method"] == "csrfile":
            want_dump = predicted_bytes(job) <= 1.5e9          # full dump / SciPy reference only for moderate outputs
        dump = tmp / "C.out"
        if dump.exists():
            dump.unlink()
        cmd = [str(exe_for(job["variant"])), job["method"], str(n), str(reps), str(warm), str(batch), job["a"], job["b"],
               str(dump) if want_dump else "-"] + ([str(job["bs"])] if job["bs"] else [])
        print("[%s] %-16s n=%-6d %-4s %-5s reps=%d warm=%d batch=%d t_est=%.3g s ..." %
              (name, job["config"], n, job["struct"], "" if dens is None else "d%g" % dens, reps, warm, batch, t_est),
              end=" ", flush=True)
        lines, status, reason, wall, ru_kib, rc, err_tail = execute(cmd, T_MULT, MEM_KIB)
        res = parse(lines)
        nnz_in = res["nnz_in"] or (None, None)
        nnz_c = res["rep"][-1][4] if res["rep"] and res["rep"][-1][4] >= 0 else None
        row = dict(base, status=status, reason=reason, reps_completed=len(res["rep"]), warmup_completed=len(res["warmup"]),
                   wall_s="%.3f" % wall, rss_after_load_kib=res["rss_after_load"], vmhwm_kib=res["vmhwm"],
                   ru_maxrss_kib=ru_kib, nnz_a=nnz_in[0], nnz_b=nnz_in[1], nnz_c=nnz_c, returncode=rc,
                   stderr_tail=err_tail)
        if status == "ok":
            checksums = [c for _, _, c, _, _ in res["rep"]]
            if sparse_m:
                row.update(validate_sparse(job, dump if want_dump else None, res, checksums))
            elif dump.exists():
                row.update(validate_dense(ref, job, dump, checksums))
            if row.get("correctness") != "pass":
                row["correctness"] = row.get("correctness", "fail")
                row["reason"] = "validation failed"
        else:
            row["correctness"] = "not_checked"
        raw = []
        for phase, items in (("warmup", res["warmup"]), ("measured", res["rep"])):
            for idx, t, chk, conv, nz in items:
                raw.append(dict(campaign=name, run_id=run_id, config=job["config"], method=job["method"],
                                numeric_type="float64", n=n, structure=job["struct"], density_pct=base["density_pct"],
                                block_size=base["block_size"], thread_count=1, phase=phase, repetition=idx,
                                batch_size=batch, time_per_mult_s=repr(t),
                                conv_s="" if conv != conv else repr(conv), nnz_c="" if nz < 0 else nz,
                                checksum=repr(chk)))
        append_csv(raw_csv, RAW_FIELDS, raw)
        append_csv(runs_csv, RUN_FIELDS, [row])
        if res["rep"]:
            prev_n, prev_t = n, statistics.median(t for _, t, _, _, _ in res["rep"])
        msg = status if status == "ok" else "%s (%s)" % (status, reason)
        print("%s, median=%s s, peak RSS=%s MiB, check=%s" %
              (msg, "%.4g" % prev_t if res["rep"] else "n/a",
               "%.1f" % (res["vmhwm"] / 1024) if res["vmhwm"] else "n/a", row["correctness"]), flush=True)
        if status in ("timeout", "memory_limit") and job["stop_on_fail"]:
            stop_reason = "an earlier, smaller size already hit the %s limit" % status.replace("_", " ")
        if status == "error":
            print("   stderr:", err_tail, flush=True)
        state[job["series"]] = [prev_n, prev_t, stop_reason]
    for p in tmp.glob("*"):
        p.unlink()
    tmp.rmdir()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True, choices=["dense", "bsweep", "flags", "procvar", "sparse", "scale"])
    ap.add_argument("--sizes", nargs="+", type=int)
    ap.add_argument("--bs", type=int, default=96, help="tile size of the blocked method (chosen from the bsweep campaign)")
    ap.add_argument("--resume", action="store_true", help="skip runs already present in results/runs.csv")
    ap.add_argument("--tag", default="", help="suffix of the campaign name (independent repeats)")
    ap.add_argument("--tmult", type=float, help="override the per-sample time limit (testing only)")
    ap.add_argument("--results-dir", help="write CSVs here instead of results/ (testing only)")
    args = ap.parse_args()
    global T_MULT, RESULTS
    if args.tmult:
        T_MULT = args.tmult
    if args.results_dir:
        RESULTS = Path(args.results_dir)
        RESULTS.mkdir(parents=True, exist_ok=True)
    jobs = job_list(args)
    name = args.campaign + args.tag
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / ("campaign_%s.json" % name), "w") as f:
        json.dump(dict(campaign=name, sizes=sorted({j["n"] for j in jobs}), configs=sorted({j["config"] for j in jobs}),
                       block_size=args.bs, t_mult_s=T_MULT, mem_budget_kib=MEM_KIB, as_limit_bytes=AS_LIMIT,
                       densities_pct=DENSITIES, structures=STRUCTS, cflags=CFLAGS,
                       rep_policy="t_est<=1s:10 reps+3 warmup; <=30s:5+1; <=120s:3+0; else 2+0",
                       batch_policy="batch=ceil(0.1 s / t_est) when t_est<0.05 s, else 1"), f, indent=2)
    run_jobs(jobs, args, name)


if __name__ == "__main__":
    main()
