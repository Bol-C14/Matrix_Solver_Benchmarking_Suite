/* ========================================================================== */
/* === KLU/Include/klu_internal.h =========================================== */
/* ========================================================================== */

/* For internal use in KLU routines only, not for user programs */

#ifndef _KLU_INTERNAL_H
#define _KLU_INTERNAL_H

#include "amd.h"
#include "btf.h"

/* ========================================================================== */
/* make sure debugging and printing is turned off */

#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef NPRINT
#define NPRINT
#endif

/* To enable debugging and assertions, uncomment this line:
 #undef NDEBUG
 */

/* To enable diagnostic printing, uncomment this line:
 #undef NPRINT
 */

/* ========================================================================== */

#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>

#undef ASSERT
#ifndef NDEBUG
#define ASSERT(a) assert(a)
#else
#define ASSERT(a)
#endif

#define SCALAR_IS_NAN(x) ((x) != (x))

/* true if an integer (stored in double x) would overflow (or if x is NaN) */
#define INT_OVERFLOW(x) ((!((x) * (1.0 + 1e-8) <= (double)Int_MAX)) || SCALAR_IS_NAN(x))

#undef TRUE
#undef FALSE
#undef MAX
#undef MIN
#undef PRINTF
#undef FLIP

#ifndef NPRINT
#define PRINTF(s) SUITESPARSE_PRINTF(s)
#else
#define PRINTF(s)
#endif

#define TRUE 1
#define FALSE 0
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/* FLIP is a "negation about -1", and is used to mark an integer i that is
 * normally non-negative.  FLIP (EMPTY) is EMPTY.  FLIP of a number > EMPTY
 * is negative, and FLIP of a number < EMTPY is positive.  FLIP (FLIP (i)) = i
 * for all integers i.  UNFLIP (i) is >= EMPTY. */
#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) (((i) < EMPTY) ? FLIP(i) : (i))

#define KLU_symbolic klu_symbolic
#define KLU_common klu_common
#define KLU_defaults klu_defaults
#define KLU_analyze klu_analyze
#define KLU_alloc_symbolic klu_alloc_symbolic
#define KLU_free_symbolic klu_free_symbolic
#define KLU_free klu_free
#define KLU_malloc klu_malloc
#define KLU_realloc klu_realloc

#define BTF_order btf_order
#define BTF_strongcomp btf_strongcomp

#define AMD_order amd_order

#define KLU_OK 0
#define KLU_SINGULAR (1) /* status > 0 is a warning, not an error */
#define KLU_OUT_OF_MEMORY (-2)
#define KLU_INVALID (-3)
#define KLU_TOO_LARGE (-4) /* integer overflow has occured */

#define Int_MAX INT_MAX

int KLU_valid(
    int n,
    int Ap[],
    int Ai[],
    double Ax[]);

typedef struct
{
    /* A (P,Q) is in upper block triangular form.  The kth block goes from
     * row/col index R [k] to R [k+1]-1.  The estimated number of nonzeros
     * in the L factor of the kth block is Lnz [k].
     */

    /* only computed if the AMD ordering is chosen: */
    double symmetry;  /* symmetry of largest block */
    double est_flops; /* est. factorization flop count */
    double lnz, unz;  /* estimated nz in L and U, including diagonals */
    double *Lnz;      /* size n, but only Lnz [0..nblocks-1] is used */

    /* computed for all orderings: */
    int
        n,        /* input matrix A is n-by-n */
        nz,       /* # entries in input matrix */
        *P,       /* size n */
        *Q,       /* size n */
        *R,       /* size n+1, but only R [0..nblocks] is used */
        nzoff,    /* nz in off-diagonal blocks */
        nblocks,  /* number of blocks */
        maxblock, /* size of largest block */
        ordering, /* ordering used (AMD, COLAMD, or GIVEN) */
        do_btf;   /* whether or not BTF preordering was requested */

    /* only computed if BTF preordering requested */
    int structural_rank; /* 0 to n-1 if the matrix is structurally rank
                          * deficient.  -1 if not computed.  n if the matrix has
                          * full structural rank */

} klu_symbolic;

typedef struct klu_common_struct
{

    /* ---------------------------------------------------------------------- */
    /* parameters */
    /* ---------------------------------------------------------------------- */

    double tol;         /* pivot tolerance for diagonal preference */
    double memgrow;     /* realloc memory growth size for LU factors */
    double initmem_amd; /* init. memory size with AMD: c*nnz(L) + n */
    double initmem;     /* init. memory size: c*nnz(A) + n */
    double maxwork;     /* maxwork for BTF, <= 0 if no limit */

    int btf;      /* use BTF pre-ordering, or not */
    int ordering; /* 0: AMD, 1: COLAMD, 2: user P and Q,
                   * 3: user function */
    int scale;    /* row scaling: -1: none (and no error check),
                   * 0: none, 1: sum, 2: max */

    /* pointer to user ordering function */
    int (*user_order)(int, int *, int *, int *, struct klu_common_struct *);

    /* pointer to user data, passed unchanged as the last parameter to the
     * user ordering function (optional, the user function need not use this
     * information). */
    void *user_data;

    int halt_if_singular; /* how to handle a singular matrix:
                           * FALSE: keep going.  Return a Numeric object with a zero U(k,k).  A
                           *   divide-by-zero may occur when computing L(:,k).  The Numeric object
                           *   can be passed to klu_solve (a divide-by-zero will occur).  It can
                           *   also be safely passed to klu_refactor.
                           * TRUE: stop quickly.  klu_factor will free the partially-constructed
                           *   Numeric object.  klu_refactor will not free it, but will leave the
                           *   numerical values only partially defined.  This is the default. */

    /* ---------------------------------------------------------------------- */
    /* statistics */
    /* ---------------------------------------------------------------------- */

    int status;   /* KLU_OK if OK, < 0 if error */
    int nrealloc; /* # of reallocations of L and U */

    int structural_rank; /* 0 to n-1 if the matrix is structurally rank
                          * deficient (as determined by maxtrans).  -1 if not computed.  n if the
                          * matrix has full structural rank.  This is computed by klu_analyze
                          * if a BTF preordering is requested. */

    int numerical_rank; /* First k for which a zero U(k,k) was found,
                         * if the matrix was singular (in the range 0 to n-1).  n if the matrix
                         * has full rank. This is not a true rank-estimation.  It just reports
                         * where the first zero pivot was found.  -1 if not computed.
                         * Computed by klu_factor and klu_refactor. */

    int singular_col; /* n if the matrix is not singular.  If in the
                       * range 0 to n-1, this is the column index of the original matrix A that
                       * corresponds to the column of U that contains a zero diagonal entry.
                       * -1 if not computed.  Computed by klu_factor and klu_refactor. */

    int noffdiag; /* # of off-diagonal pivots, -1 if not computed */

    double flops;   /* actual factorization flop count, from klu_flops */
    double rcond;   /* crude reciprocal condition est., from klu_rcond */
    double condest; /* accurate condition est., from klu_condest */
    double rgrowth; /* reciprocal pivot rgrowth, from klu_rgrowth */
    double work;    /* actual work done in BTF, in klu_analyze */

    size_t memusage; /* current memory usage, in bytes */
    size_t mempeak;  /* peak memory usage, in bytes */

} klu_common;

klu_symbolic *klu_analyze(
    /* inputs, not modified */
    int n,    /* A is n-by-n */
    int Ap[], /* size n+1, column pointers */
    int Ai[], /* size nz, row indices */
    klu_common *Common);

int klu_free_symbolic(
    klu_symbolic **Symbolic,
    klu_common *Common);

int klu_defaults(
    klu_common *Common);

void *klu_malloc /* returns pointer to the newly malloc'd block */
    (
        /* ---- input ---- */
        size_t n,    /* number of items */
        size_t size, /* size of each item */
        /* --------------- */
        klu_common *Common);

void *klu_free /* always returns NULL */
    (
        /* ---- in/out --- */
        void *p,     /* block of memory to free */
        size_t n,    /* number of items */
        size_t size, /* size of each item */
        /* --------------- */
        klu_common *Common);

void *klu_realloc /* returns pointer to reallocated block */
    (
        /* ---- input ---- */
        size_t nnew, /* requested # of items in reallocated block */
        size_t nold, /* current size of block, in # of items */
        size_t size, /* size of each item */
        /* ---- in/out --- */
        void *p, /* block of memory to realloc */
        /* --------------- */
        klu_common *Common);

#endif
