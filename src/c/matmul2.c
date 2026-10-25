/*
 * Big Data (ULPGC, 2026-2027) - Individual Assignment 2
 * Optimised dense multiplication and sparse (CSR) multiplication, C implementation.
 *
 * Methods (first command-line argument)
 *   ijk      baseline, textbook triple loop i-j-k with a scalar accumulator
 *            (kernel REUSED unchanged from Assignment 1, src/c/matmul.c)
 *   ikj      loop reordering i-k-j: the inner loop walks rows of B and C contiguously
 *   blocked  cache blocking (tiling) of i, k and j with one tile size `bs`; inside a tile
 *            the order is i-k-j; tiles are visited in the order ii, kk, jj
 *   csr      sparse CSR x CSR -> CSR (Gustavson row-wise algorithm with a dense
 *            accumulator, "SPA"). Inputs are DENSE files that are converted to CSR
 *            (the conversion is timed separately from the multiplication kernel).
 *   csrfile  same kernel, inputs are CSR files (no conversion)
 *
 * Accumulation order. For every element c_ij all methods add the products a_ik*b_kj
 * for k = 0..n-1 in increasing order, starting from 0.0, using IEEE-754 binary64
 * without fused multiply-add (compile with -ffp-contract=off). Blocking only splits
 * the k range into consecutive chunks that are processed in increasing order, and
 * the sparse kernel skips terms that are exactly zero (adding +0.0 never changes a
 * non-negative sum). Therefore all methods are expected to produce bit-for-bit
 * identical results; the test-suite and the benchmark driver check this.
 *
 * Timed region (identical rule for every method): the kernel reads the two input
 * operands and defines every element of the output. The C buffer of the dense methods
 * is allocated and page-touched OUTSIDE the timed region (as in Assignment 1); the
 * zero-fill that i-k-j and blocked need is INSIDE the kernel. The sparse kernel
 * allocates its workspace and its output arrays (size unknown in advance, grown with
 * realloc) INSIDE the timed region; freeing the output, the checksum, the file loading
 * and the dense->CSR conversion are outside (the conversion is timed on its own).
 *
 * Usage:
 *   matmul2 <method> <n> <reps> <warmup> <batch> <A> <B> <dumpC|-> [bs]
 * Output (stdout, one record per line, flushed immediately):
 *   META method=.. n=.. reps=.. warmup=.. batch=.. bs=..
 *   NNZ <nnzA> <nnzB>                         (sparse methods)
 *   RSS_AFTER_LOAD_KB <kB>
 *   WARMUP|REP <idx> <seconds per multiplication> <checksum> [conv_s=<s>] [nnz_c=<k>]
 *   ROWCHK <row> <nnz of row> <sum of row>    (sparse methods, spot-check rows)
 *   VMHWM_KB <kB>
 *   DONE
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void die_oom(const char *what) {
    fprintf(stderr, "out of memory (%s)\n", what);
    exit(3);
}

static void *xmalloc(size_t bytes, const char *what) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) die_oom(what);
    return p;
}

/* ------------------------------------------------------------------ dense I/O */
static double *read_dense(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); exit(2); }
    double *m = xmalloc(n * n * sizeof(double), "dense input");
    if (fread(m, sizeof(double), n * n, f) != n * n) {
        fprintf(stderr, "short read on %s (expected %zu doubles)\n", path, n * n);
        exit(2);
    }
    fclose(f);
    return m;
}

/* ------------------------------------------------------------ dense kernels */

/* Baseline (Assignment 1, unchanged): i-j-k with a scalar accumulator. */
__attribute__((noinline))
static void mul_ijk(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t k = 0; k < n; k++) {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

/* Optimisation 1: loop reordering i-k-j. Same products, same order per element. */
__attribute__((noinline))
static void mul_ikj(const double *restrict a, const double *restrict b,
                    double *restrict c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double *restrict ci = c + i * n;
        memset(ci, 0, n * sizeof(double));
        for (size_t k = 0; k < n; k++) {
            const double aik = a[i * n + k];
            const double *restrict bk = b + k * n;
            for (size_t j = 0; j < n; j++) {
                ci[j] += aik * bk[j];
            }
        }
    }
}

/* Optimisation 2: cache blocking. Tiles of at most bs x bs elements of A, B and C. */
__attribute__((noinline))
static void mul_blocked(const double *restrict a, const double *restrict b,
                        double *restrict c, size_t n, size_t bs) {
    memset(c, 0, n * n * sizeof(double));
    for (size_t ii = 0; ii < n; ii += bs) {
        size_t imax = ii + bs < n ? ii + bs : n;
        for (size_t kk = 0; kk < n; kk += bs) {
            size_t kmax = kk + bs < n ? kk + bs : n;
            for (size_t jj = 0; jj < n; jj += bs) {
                size_t jmax = jj + bs < n ? jj + bs : n;
                for (size_t i = ii; i < imax; i++) {
                    double *restrict ci = c + i * n;
                    for (size_t k = kk; k < kmax; k++) {
                        const double aik = a[i * n + k];
                        const double *restrict bk = b + k * n;
                        for (size_t j = jj; j < jmax; j++) {
                            ci[j] += aik * bk[j];
                        }
                    }
                }
            }
        }
    }
}

/* Plain left-to-right sum of all elements in row-major order. */
static double checksum_dense(const double *c, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n * n; i++) s += c[i];
    return s;
}

/* ---------------------------------------------------------------- CSR format */
typedef struct {
    int64_t n;        /* square matrix, n rows and n columns */
    int64_t nnz;
    int64_t *ptr;     /* n+1 row pointers */
    int32_t *idx;     /* nnz column indices, strictly increasing inside a row */
    double *val;      /* nnz values */
} csr_t;

static void csr_free(csr_t *m) {
    free(m->ptr); free(m->idx); free(m->val);
    m->ptr = NULL; m->idx = NULL; m->val = NULL; m->nnz = 0;
}

/* Dense -> CSR (two passes: count, then fill). Entries equal to 0.0 are not stored. */
static csr_t dense_to_csr(const double *d, size_t n) {
    csr_t m;
    m.n = (int64_t)n;
    m.ptr = xmalloc((n + 1) * sizeof(int64_t), "csr ptr");
    int64_t nnz = 0;
    m.ptr[0] = 0;
    for (size_t i = 0; i < n; i++) {
        const double *row = d + i * n;
        for (size_t j = 0; j < n; j++) nnz += (row[j] != 0.0);
        m.ptr[i + 1] = nnz;
    }
    m.nnz = nnz;
    m.idx = xmalloc((size_t)nnz * sizeof(int32_t), "csr idx");
    m.val = xmalloc((size_t)nnz * sizeof(double), "csr val");
    int64_t p = 0;
    for (size_t i = 0; i < n; i++) {
        const double *row = d + i * n;
        for (size_t j = 0; j < n; j++) {
            if (row[j] != 0.0) { m.idx[p] = (int32_t)j; m.val[p] = row[j]; p++; }
        }
    }
    return m;
}

/* CSR file: "CSR1", int64 n, int64 nnz, int64 ptr[n+1], int32 idx[nnz], float64 val[nnz]. */
static csr_t read_csr(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); exit(2); }
    char magic[4];
    int64_t hdr[2];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CSR1", 4) != 0 ||
        fread(hdr, sizeof(int64_t), 2, f) != 2) {
        fprintf(stderr, "bad CSR header in %s\n", path); exit(2);
    }
    csr_t m;
    m.n = hdr[0]; m.nnz = hdr[1];
    m.ptr = xmalloc((size_t)(m.n + 1) * sizeof(int64_t), "csr ptr");
    m.idx = xmalloc((size_t)m.nnz * sizeof(int32_t), "csr idx");
    m.val = xmalloc((size_t)m.nnz * sizeof(double), "csr val");
    if (fread(m.ptr, sizeof(int64_t), (size_t)m.n + 1, f) != (size_t)m.n + 1 ||
        fread(m.idx, sizeof(int32_t), (size_t)m.nnz, f) != (size_t)m.nnz ||
        fread(m.val, sizeof(double), (size_t)m.nnz, f) != (size_t)m.nnz) {
        fprintf(stderr, "short read on %s\n", path); exit(2);
    }
    fclose(f);
    return m;
}

static int write_csr(const char *path, const csr_t *m) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    int64_t hdr[2] = { m->n, m->nnz };
    int ok = fwrite("CSR1", 1, 4, f) == 4 && fwrite(hdr, sizeof(int64_t), 2, f) == 2 &&
             fwrite(m->ptr, sizeof(int64_t), (size_t)m->n + 1, f) == (size_t)m->n + 1 &&
             fwrite(m->idx, sizeof(int32_t), (size_t)m->nnz, f) == (size_t)m->nnz &&
             fwrite(m->val, sizeof(double), (size_t)m->nnz, f) == (size_t)m->nnz;
    fclose(f);
    return ok;
}

static void sort_i32(int32_t *a, int32_t len);
static int cmp_i32(const void *x, const void *y) {
    int32_t a = *(const int32_t *)x, b = *(const int32_t *)y;
    return (a > b) - (a < b);
}
static void sort_i32(int32_t *a, int32_t len) {
    if (len <= 32) {                       /* insertion sort for short rows */
        for (int32_t i = 1; i < len; i++) {
            int32_t v = a[i], j = i - 1;
            while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
            a[j + 1] = v;
        }
    } else {
        qsort(a, (size_t)len, sizeof(int32_t), cmp_i32);
    }
}

/* Sparse kernel: C = A*B, Gustavson row-wise with a dense accumulator (SPA).
 * For row i of A, every nonzero a_ik scales row k of B and is accumulated in acc[];
 * flag[j] == i marks the columns touched in row i. Output columns are sorted so that C
 * is canonical CSR. Workspace and output are allocated here (inside the timed region). */
__attribute__((noinline))
static csr_t csr_multiply(const csr_t *A, const csr_t *B) {
    const int64_t n = A->n;
    csr_t C;
    C.n = n;
    C.ptr = xmalloc((size_t)(n + 1) * sizeof(int64_t), "C ptr");
    int64_t cap = A->nnz + B->nnz + 16;
    C.idx = xmalloc((size_t)cap * sizeof(int32_t), "C idx");
    C.val = xmalloc((size_t)cap * sizeof(double), "C val");
    double *acc = xmalloc((size_t)n * sizeof(double), "workspace acc");
    int32_t *flag = xmalloc((size_t)n * sizeof(int32_t), "workspace flag");
    int32_t *cols = xmalloc((size_t)n * sizeof(int32_t), "workspace cols");
    memset(flag, 0xFF, (size_t)n * sizeof(int32_t));   /* all -1 */
    int64_t nz = 0;
    C.ptr[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t cnt = 0;
        for (int64_t p = A->ptr[i]; p < A->ptr[i + 1]; p++) {
            const int32_t k = A->idx[p];
            const double a = A->val[p];
            for (int64_t q = B->ptr[k]; q < B->ptr[k + 1]; q++) {
                const int32_t j = B->idx[q];
                const double v = a * B->val[q];
                if (flag[j] != (int32_t)i) { flag[j] = (int32_t)i; acc[j] = v; cols[cnt++] = j; }
                else acc[j] += v;
            }
        }
        if (nz + cnt > cap) {                           /* grow the output (amortised doubling) */
            int64_t ncap = cap * 2;
            if (ncap < nz + cnt) ncap = nz + cnt;
            int32_t *ni = realloc(C.idx, (size_t)ncap * sizeof(int32_t));
            if (!ni) die_oom("C idx growth");
            C.idx = ni;
            double *nv = realloc(C.val, (size_t)ncap * sizeof(double));
            if (!nv) die_oom("C val growth");
            C.val = nv;
            cap = ncap;
        }
        if ((int64_t)cnt * 16 > n) {                    /* dense-ish row: scan the flags */
            for (int64_t j = 0; j < n; j++) {
                if (flag[j] == (int32_t)i) { C.idx[nz] = (int32_t)j; C.val[nz] = acc[j]; nz++; }
            }
        } else {                                        /* sparse row: sort the touched columns */
            sort_i32(cols, cnt);
            for (int32_t t = 0; t < cnt; t++) {
                C.idx[nz] = cols[t]; C.val[nz] = acc[cols[t]]; nz++;
            }
        }
        C.ptr[i + 1] = nz;
    }
    C.nnz = nz;
    free(acc); free(flag); free(cols);
    return C;
}

/* Plain left-to-right sum of the stored values (= dense row-major sum, zeros omitted). */
static double checksum_csr(const csr_t *c) {
    double s = 0.0;
    for (int64_t p = 0; p < c->nnz; p++) s += c->val[p];
    return s;
}

/* ------------------------------------------------------------------- utilities */
static long proc_status_kb(const char *key) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long val = -1;
    size_t kl = strlen(key);
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, kl) == 0 && line[kl] == ':') {
            val = strtol(line + kl + 1, NULL, 10);
            break;
        }
    }
    fclose(f);
    return val;
}

#define NROWCHK 64
static void print_rowchk(const csr_t *c) {
    /* 64 deterministic rows spread over the matrix: nnz and plain sum of each row */
    int64_t n = c->n;
    for (int t = 0; t < NROWCHK; t++) {
        int64_t i = (n * t) / NROWCHK;
        double s = 0.0;
        for (int64_t p = c->ptr[i]; p < c->ptr[i + 1]; p++) s += c->val[p];
        printf("ROWCHK %lld %lld %.17g\n", (long long)i, (long long)(c->ptr[i + 1] - c->ptr[i]), s);
    }
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s method n reps warmup batch A B dumpC|- [bs]\n", argv[0]);
        return 1;
    }
    const char *method = argv[1];
    size_t n = strtoull(argv[2], NULL, 10);
    int reps = atoi(argv[3]);
    int warmup = atoi(argv[4]);
    long batch = atol(argv[5]);
    if (batch < 1) batch = 1;
    const char *dump = strcmp(argv[8], "-") ? argv[8] : NULL;
    size_t bs = argc >= 10 ? strtoull(argv[9], NULL, 10) : 0;
    int dense_m = !strcmp(method, "ijk") || !strcmp(method, "ikj") || !strcmp(method, "blocked");
    int sparse_m = !strcmp(method, "csr") || !strcmp(method, "csrfile");
    if (!dense_m && !sparse_m) { fprintf(stderr, "unknown method %s\n", method); return 1; }
    if (!strcmp(method, "blocked") && bs < 1) { fprintf(stderr, "blocked needs bs >= 1\n"); return 1; }

    printf("META method=%s n=%zu reps=%d warmup=%d batch=%ld bs=%zu\n", method, n, reps, warmup, batch, bs);
    fflush(stdout);

    if (dense_m) {
        double *a = read_dense(argv[6], n);
        double *b = read_dense(argv[7], n);
        double *c = xmalloc(n * n * sizeof(double), "dense C");
        memset(c, 0, n * n * sizeof(double));    /* touch pages before timing */
        printf("RSS_AFTER_LOAD_KB %ld\n", proc_status_kb("VmRSS"));
        fflush(stdout);
        for (int w = 0; w < warmup + reps; w++) {
            int64_t t0 = now_ns();
            for (long q = 0; q < batch; q++) {
                if (!strcmp(method, "ijk")) mul_ijk(a, b, c, n);
                else if (!strcmp(method, "ikj")) mul_ikj(a, b, c, n);
                else mul_blocked(a, b, c, n, bs);
            }
            int64_t t1 = now_ns();
            printf("%s %d %.9e %.17g\n", w < warmup ? "WARMUP" : "REP", w < warmup ? w : w - warmup,
                   (double)(t1 - t0) * 1e-9 / (double)batch, checksum_dense(c, n));
            fflush(stdout);
        }
        printf("VMHWM_KB %ld\n", proc_status_kb("VmHWM"));
        fflush(stdout);
        if (dump) {
            FILE *f = fopen(dump, "wb");
            if (!f || fwrite(c, sizeof(double), n * n, f) != n * n) { fprintf(stderr, "cannot write dump\n"); return 4; }
            fclose(f);
        }
        printf("DONE\n");
        fflush(stdout);
        return 0;
    }

    /* ---- sparse methods ---- */
    int from_dense = !strcmp(method, "csr");
    double *da = NULL, *db = NULL;
    csr_t A, B;
    if (from_dense) {
        da = read_dense(argv[6], n);
        db = read_dense(argv[7], n);
        A = dense_to_csr(da, n);     /* initial conversion, only to report nnz and to keep RSS honest */
        B = dense_to_csr(db, n);
        csr_free(&A); csr_free(&B);
    } else {
        A = read_csr(argv[6]);
        B = read_csr(argv[7]);
        n = (size_t)A.n;
        if (B.n != A.n) { fprintf(stderr, "operand size mismatch\n"); return 2; }
    }
    printf("RSS_AFTER_LOAD_KB %ld\n", proc_status_kb("VmRSS"));
    fflush(stdout);

    csr_t C = {0};
    for (int w = 0; w < warmup + reps; w++) {
        double conv_s = 0.0;
        if (from_dense) {
            int64_t c0 = now_ns();
            A = dense_to_csr(da, n);
            B = dense_to_csr(db, n);
            int64_t c1 = now_ns();
            conv_s = (double)(c1 - c0) * 1e-9;
        }
        if (w == 0) { printf("NNZ %lld %lld\n", (long long)A.nnz, (long long)B.nnz); fflush(stdout); }
        int64_t t0 = now_ns();
        for (long q = 0; q < batch; q++) {
            if (C.ptr) csr_free(&C);              /* only when batching (tiny inputs) */
            C = csr_multiply(&A, &B);
        }
        int64_t t1 = now_ns();
        printf("%s %d %.9e %.17g conv_s=%.9e nnz_c=%lld\n", w < warmup ? "WARMUP" : "REP",
               w < warmup ? w : w - warmup, (double)(t1 - t0) * 1e-9 / (double)batch,
               checksum_csr(&C), conv_s, (long long)C.nnz);
        fflush(stdout);
        if (w == warmup + reps - 1) {
            print_rowchk(&C);
            printf("VMHWM_KB %ld\n", proc_status_kb("VmHWM"));
            fflush(stdout);
            if (dump && !write_csr(dump, &C)) { fprintf(stderr, "cannot write dump\n"); return 4; }
        }
        if (from_dense) { csr_free(&A); csr_free(&B); }
        csr_free(&C);
    }
    printf("DONE\n");
    fflush(stdout);
    return 0;
}
