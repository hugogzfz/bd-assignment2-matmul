# Big Data (ULPGC 2026-2027) - Individual Assignment 2: optimised dense and sparse matrix multiplication
# Usage:  make all   (build, inputs, tests, benchmarks, analysis, report)
#         make build | inputs | test | vecinfo | bench-dense | bench-flags | bench-sparse | analyze | report
# The blocked method uses the tile size BS chosen from the bsweep campaign (see README / report).

PY      ?= python3
CC       = gcc
BS      ?= 96
# Primary C configuration (same as Assignment 1): -O2, no -march=native, FMA contraction disabled so that
# every method performs bit-for-bit the same arithmetic.
CFLAGS  ?= -O2 -std=c11 -ffp-contract=off -Wall -Wextra
DENSE_SIZES   := 100 250 500 512 750 1000 1024 1250 1500 2000 2048 3000 4000 5000 6000 8000
SPARSE_SIZES  := 500 1000 2000
DENSITIES     := 0.1 1 5 10 20 30 50 70 90
STRUCTS       := unif band

.PHONY: all build inputs inputs-quick test vecinfo bench bench-quick bench-dense bench-flags bench-sparse analyze report clean

all: build inputs test vecinfo bench analyze report

build:
	mkdir -p build/flags
	$(CC) $(CFLAGS) -o build/matmul2 src/c/matmul2.c
	$(CC) -O2 -funroll-loops   -std=c11 -ffp-contract=off -o build/flags/matmul2_O2-unroll  src/c/matmul2.c
	$(CC) -O3                  -std=c11 -ffp-contract=off -o build/flags/matmul2_O3         src/c/matmul2.c
	$(CC) -O3 -march=native    -std=c11 -ffp-contract=off -o build/flags/matmul2_O3-native  src/c/matmul2.c

inputs:
	$(PY) scripts/gen_inputs.py dense $(DENSE_SIZES)
	$(PY) scripts/gen_inputs.py sparse --sizes $(SPARSE_SIZES) --dens $(DENSITIES) --structs $(STRUCTS) --dense
	$(PY) scripts/gen_inputs.py sparse --sizes 2000 4000 8000 16000 32000 64000 128000 --dens 0.1 --structs unif
	$(PY) scripts/gen_inputs.py sparse --sizes 4000 8000 16000 32000 --dens 1 --structs unif

inputs-quick:
	$(PY) scripts/gen_inputs.py dense 100 250 500 512 1000
	$(PY) scripts/gen_inputs.py sparse --sizes 500 --dens $(DENSITIES) --structs $(STRUCTS) --dense

test: build
	$(PY) scripts/test_correctness.py

vecinfo: build
	$(PY) scripts/vec_evidence.py

bench: bench-dense bench-flags bench-sparse

# Reduced campaign (about 3 minutes): dense n<=1000 and sparse n=500, written to results_quick/ (does not touch results/).
bench-quick: build inputs-quick
	$(PY) scripts/run_benchmarks.py --campaign dense --sizes 100 250 500 512 1000 --bs $(BS) --results-dir results_quick
	$(PY) scripts/run_benchmarks.py --campaign sparse --sizes 500 --bs $(BS) --results-dir results_quick

bench-dense: build inputs
	$(PY) scripts/env_info.py
	$(PY) scripts/run_benchmarks.py --campaign bsweep --resume
	$(PY) scripts/run_benchmarks.py --campaign dense --bs $(BS) --resume

bench-flags: build inputs
	$(PY) scripts/run_benchmarks.py --campaign flags --bs $(BS) --resume
	for k in 1 2 3 4 5; do $(PY) scripts/run_benchmarks.py --campaign procvar --bs $(BS) --tag $$k --resume; done

bench-sparse: build inputs
	$(PY) scripts/run_benchmarks.py --campaign sparse --bs $(BS) --resume
	$(PY) scripts/run_benchmarks.py --campaign scale --resume

analyze:
	$(PY) scripts/analyze.py

report:
	cd report && latexmk -pdf -interaction=nonstopmode report.tex

clean:
	rm -rf build
