#!/usr/bin/env python3
"""Counts packed (SIMD) vs scalar multiply instructions in each kernel for every compiler-flag build
(objdump) -> results/vectorization_evidence.csv."""
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLAGS = {"O2": "-O2", "O2-unroll": "-O2 -funroll-loops", "O3": "-O3", "O3-native": "-O3 -march=native"}
KERNELS = ["mul_ijk", "mul_ikj", "mul_blocked", "csr_multiply"]


def exe(v):
    return ROOT / "build" / ("matmul2" if v == "O2" else "flags/matmul2_" + v)


def main():
    out_rows = []
    for v, fl in FLAGS.items():
        asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-M", "intel", str(exe(v))],
                             capture_output=True, text=True).stdout
        funcs = {}
        cur = None
        for line in asm.splitlines():
            m = re.match(r"^[0-9a-f]+ <(\w+)>:", line)
            if m:
                cur = m.group(1)
                funcs[cur] = []
            elif cur and "\t" in line:
                funcs[cur].append(line.split("\t", 1)[1])
        for k in KERNELS:
            body = funcs.get(k, [])
            if not body:
                # static function inlined or renamed by IPA: look for a clone such as mul_ikj.constprop.0
                for name in funcs:
                    if name.startswith(k + "."):
                        body = funcs[name]
                        break
            text = "\n".join(body)
            packed = len(re.findall(r"\bv?mulpd\b", text))
            scalar = len(re.findall(r"\bv?mulsd\b", text))
            fma = len(re.findall(r"\bvfn?m(add|sub)\w*", text))
            width = "zmm" if "zmm" in text else "ymm" if "ymm" in text else "xmm" if "xmm" in text else "-"
            out_rows.append(dict(variant=v, flags=fl + " -std=c11 -ffp-contract=off", function=k,
                                 instructions=len(body), packed_mul=packed, scalar_mul=scalar,
                                 fused_multiply_add=fma, widest_vector_register=width))
    with open(ROOT / "results" / "vectorization_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    for r in out_rows:
        print("%-10s %-13s instr=%-4d packed_mul=%-3d scalar_mul=%-3d fma=%-3d widest=%s" %
              (r["variant"], r["function"], r["instructions"], r["packed_mul"], r["scalar_mul"],
               r["fused_multiply_add"], r["widest_vector_register"]))


if __name__ == "__main__":
    main()
