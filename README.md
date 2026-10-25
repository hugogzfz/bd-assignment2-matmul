# Big Data (ULPGC 2026-2027) – Individual Assignment 2
## Optimised dense matrix multiplication and sparse matrices (C, sequential)

Author: Hugo González Fernández · Bachelor’s Degree in Data Science and Engineering, ULPGC

Research questions
* **Part A** – How much do loop reordering (i-k-j) and cache blocking reduce the runtime of the Assignment 1 baseline (i-j-k)
  for dense `float64` n × n matrices, with identical inputs, arithmetic and timing boundary?
* **Part B** – For which sizes, densities and nonzero structures does an own CSR × CSR (Gustavson) multiplication beat the dense
  methods on the same logical matrices, and what do conversion, indexing and output growth cost?

The report (`Assignment02_GonzalezFernandez_Report.pdf`) contains design, results, discussion and limits. This README gives everything
needed to rebuild and re-run the study. Everything is sequential and single-threaded.

## 1. Layout

```
src/c/matmul2.c            all methods: ijk (A1 baseline, unchanged), ikj, blocked, csr, csrfile
scripts/gen_inputs.py      dense (A1 files, digests checked against Assignment 1) and sparse inputs (seeded)
scripts/test_correctness.py   377 checks: known answers, NumPy/SciPy references, bitwise equality, edge cases, repeatability
scripts/run_benchmarks.py  measurement driver (campaigns: bsweep, dense, flags, procvar, sparse, scale)
scripts/env_info.py        environment -> results/environment.json
scripts/vec_evidence.py    gcc vectorisation report for each flag set
scripts/analyze.py         summaries, LaTeX tables, figures
scripts/run_all_benchmarks.sh   the whole campaign chain, sequentially
results/                   raw_measurements.csv (every sample), runs.csv (one row per process), summary_*.csv, logs, environment.json
figures/  report/         figures and LaTeX sources of the report
data/manifest.json        SHA-256 and nnz of every input (the .bin / .csr files are NOT committed)
data/a1_reference_sha256.json   digests of the Assignment 1 inputs
```

## 2. Requirements (as used)

Linux x86-64 (the driver reads `/proc/self/status`, so **it does not run on macOS**), gcc 13.x, Python 3.13 with
`numpy 2.5.3 pandas 3.0.5 matplotlib 3.11.2 scipy 1.18.1` (`pip install -r requirements.txt`), and `latexmk`/`pdflatex` only for the PDF.
NumPy/SciPy are used only for input generation, validation and analysis – never inside a timed kernel.
About 6 GB of disk are needed for the generated inputs.

## 3. Exact commands

```bash
pip install -r requirements.txt
make build          # gcc -O2 -std=c11 -ffp-contract=off  (+ three flag variants for the supplementary experiment)
make inputs         # generates (or verifies against data/manifest.json) all dense and sparse inputs
make test           # correctness suite -> results/correctness_tests.csv   (must report 0 failed)
make vecinfo        # vectorisation evidence
make bench-quick    # reduced run (~3 min: dense n<=1000, sparse n=500) -> results_quick/, to check that everything works on your machine
make bench          # the full campaigns of the report (several hours; resumable: the driver is called with --resume)
make analyze        # tables + figures
make report         # report/report.pdf
```

Single run by hand: `build/matmul2 <method> <n> <reps> <warmup> <batch> <A> <B> <dumpC|-> [bs]`

```bash
build/matmul2 blocked 1000 5 1 1 data/A_n1000.bin data/B_n1000.bin - 96
build/matmul2 csr     1000 5 1 1 data/A_unif_n1000_d1.bin data/B_unif_n1000_d1.bin -
```
(sparse file names are those listed in `data/manifest.json`). Output lines: `META`, `NNZ`, `WARMUP|REP idx seconds checksum [conv_s= nnz_c=]`,
`ROWCHK`, `VMHWM_KB`, `DONE`. Methods `csr` read dense files and time the conversion separately; `csrfile` reads CSR files.

## 4. Fixed settings and timing boundary

`gcc -O2 -std=c11 -ffp-contract=off` (no `-march=native`, no `-ffast-math`) → all methods perform the same additions in the same order
and return identical bits. The timer wraps the multiplication kernel only: dense `C` is allocated and page-touched before the timer, the
zero-fill of `C` (needed by ikj/blocked) is inside; the CSR kernel includes workspace and output allocation/growth. Dense → CSR conversion is
timed separately per repetition (`conv_s`). File loading, checksum and dump are outside. Monotonic clock (`CLOCK_MONOTONIC`).

## 5. Protocol

Repetitions: estimated time ≤1 s → 3 warm-up + 10 measured; ≤30 s → 1+5; ≤120 s → 0+3; else 0+2; batching below 50 ms.
Median and IQR reported; warm-ups are kept in the raw CSV but excluded. Budgets: 300 s per timed sample, 4 GiB per process
(RSS polled every 0.5 s + address-space cap); violations are recorded as `timeout` / `memory_limit`, never replaced by invented values.
Memory metric: `VmHWM` (peak resident set of the whole process). Every run dumps its last product and the driver validates it against NumPy
(dense) or SciPy (sparse); the column `correctness` must be `pass`.

## 6. Inputs

Dense: the Assignment 1 matrices (uniform [0,1), `default_rng([20261012, n, w])`), digests identical to Assignment 1. Sparse: density = % of
**nonzeros** (0.1, 1, 5, 10, 20, 30, 50, 70, 90 % → sparsity 99.9–10 %), structures `unif` (per-row Binomial count at random distinct columns) and `band`
(filled band centred on the diagonal), sizes 500/1000/2000 for the matched comparison; `unif` CSR files up to n = 128000 (d = 0.1 %) and 32000 (d = 1 %) for scaling.
Seeds, achieved nnz and SHA-256 are in `data/manifest.json`; `gen_inputs.py` aborts if a regenerated file differs from the manifest.

## 7. Columns of the raw CSV files

`results/raw_measurements.csv` (one row per timed or warm-up sample): `campaign, run_id, config, method, numeric_type, n, structure, density_pct,
block_size, thread_count, phase (warmup|measured), repetition, batch_size, time_per_mult_s, conv_s, nnz_c, checksum`.

`results/runs.csv` (one row per process): the same identifiers plus `cflags, status (ok|timeout|memory_limit|not_attempted|error), reason, reps_planned,
reps_completed, warmup_planned, warmup_completed, t_est_s, limit_sample_s, limit_mem_kib, wall_s, rss_after_load_kib, vmhwm_kib (peak RSS used in the report),
ru_maxrss_kib (cross-check), nnz_a, nnz_b, nnz_c, predicted_bytes, checksum_rel_err, checksum_tol, max_abs_err, max_rel_err, rtol_used, correctness (must be pass),
sha256_C, start_utc, returncode, stderr_tail`.

## 8. Notes and limits

* Reported numbers come from one Linux virtual machine (see `results/environment.json`); absolute times on other machines differ.
* No hardware performance counters were available; statements about cache/TLB are hypotheses, not measurements.
* A short pilot (n = 1500) and a smoke test (n = 500) were run before the main campaign; their use (tile size, extra densities, flag experiment) is disclosed in the report.
* Use of AI: see the AI-use disclosure at the end of the report.
