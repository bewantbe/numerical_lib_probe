/*
Compile:
$ gcc -shared -fPIC lapack_intercept.c -o lapack_intercept.so -ldl

Run:
# LD_PRELOAD=$PWD/lapack_intercept.so matlab17 -nojvm

Ref:

FORTRAN function DGESVD
http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html#ga84fdf22a62b12ff364621e4713ce02f2

Equavalent C prototpye to FORTRAN routine DGESVD
$MATLABROOT/extern/include/lapack.h

The dynamic linker tricks
https://rafalcieslak.wordpress.com/2013/04/02/dynamic-linker-tricks-using-ld_preload-to-cheat-inject-features-and-investigate-programs/

*/

#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>      // for dlsym()
#include <stddef.h>     // for ptrdiff_t
#include <sys/time.h>   // for gettimeofday()

double tofday(const struct timeval *tv0, const struct timeval *tv1)
{
  return tv1->tv_sec + 1e-6 * tv1->tv_usec
      - (tv0->tv_sec + 1e-6 * tv0->tv_usec);
}

/* Matlab integer type */
#define INT ptrdiff_t

typedef int (*dgesvd_type)(
  char *JOBU, char *JOBVT, INT *M, INT *N, double *A, INT *LDA,
  double *S, double *U, INT *LDU, double *VT, INT *LDVT,
  double *WORK, INT *LWORK, INT *INFO);

int dgesvd_(char *JOBU, char *JOBVT, INT *M, INT *N, double *A, INT *LDA,
            double *S, double *U, INT *LDU, double *VT, INT *LDVT,
            double *WORK, INT *LWORK, INT *INFO)
{
  printf("call dgesvd ...\n");
  dgesvd_type orig_dgesvd;
  orig_dgesvd = (dgesvd_type)dlsym(RTLD_NEXT, "dgesvd_");

  struct timeval tv0, tv1;
  gettimeofday(&tv0, NULL);
  int ret = orig_dgesvd(JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO);
  gettimeofday(&tv1, NULL);

  printf("done dgesvd (t = %.3f sec)\n", tofday(&tv0, &tv1));
  return ret;
}

#define LAPACK_INTERCEPT(fname, fname_st, CALL, TYPE_PARAM...) \
typedef int (*fname##_type)(TYPE_PARAM); \
int fname(TYPE_PARAM) \
{ \
  printf("call "fname_st" ...\n"); \
  fname##_type call \
    = (fname##_type)dlsym(RTLD_NEXT, fname_st); \
 \
  struct timeval tv0, tv1; \
  gettimeofday(&tv0, NULL); \
  int ret = CALL; \
  gettimeofday(&tv1, NULL); \
 \
  printf("done "fname_st" (t = %.3f sec)\n", tofday(&tv0, &tv1)); \
  return ret; \
}

LAPACK_INTERCEPT(dgesdd_, "dgesdd_",
  call(JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, IWORK, INFO),
  char *JOBZ, INT *M, INT *N, double *A, INT *LDA,
  double *S, double *U, INT *LDU, double *VT, INT *LDVT,
  double *WORK, INT *LWORK, INT *IWORK, INT *INFO)

LAPACK_INTERCEPT(mkl_lapack_dgebrd, "mkl_lapack_dgebrd",
  call(M, N, A, LDA, D, E, TAUQ, TAUP, WORK, LWORK, INFO),
  INT *M, INT *N, double *A, INT *LDA,
  double *D, double *E, double *TAUQ, double *TAUP,
  double *WORK, INT *LWORK, INT *INFO)

LAPACK_INTERCEPT(mkl_lapack_dgesvd, "mkl_lapack_dgesvd",
  call(JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO),
  char *JOBU, char *JOBVT, INT *M, INT *N, double *A, INT *LDA,
  double *S, double *U, INT *LDU, double *VT, INT *LDVT,
  double *WORK, INT *LWORK, INT *INFO)

LAPACK_INTERCEPT(mkl_lapack_dbdsqr, "mkl_lapack_dbdsqr",
  call(UPLO, N, NCVT, NRU, NCC, D, E, VT, LDVT, U, LDU, C, LDC, WORK, INFO),
  char *UPLO, INT *N, INT *NCVT, INT *NRU, INT *NCC, double *D, double *E,
  double *VT, INT *LDVT, double *U, INT *LDU, double *C, INT *LDC,
  double *WORK, INT *INFO)

LAPACK_INTERCEPT(mkl_lapack_dlasdq, "mkl_lapack_dlasdq",
  call(UPLO, SQRE, N, NCVT, NRU, NCC, D, E, VT, LDVT, U, LDU, C, LDC, WORK, INFO),
  char *UPLO, INT *SQRE, INT *N, INT *NCVT, INT *NRU, INT *NCC,
  double *D, double *E, double *VT, INT *LDVT, double *U, INT *LDU,
  double *C, INT *LDC, double *WORK, INT *INFO)

LAPACK_INTERCEPT(mkl_lapack_dbdsdc, "mkl_lapack_dbdsdc",
  call(UPLO, COMPQ, N, D, E, U, LDU, VT, LDVT, Q, IQ, WORK, IWORK, INFO),
  char *UPLO, char *COMPQ, INT *N, double *D, double *E,
  double *U, INT *LDU, double *VT, INT *LDVT, double *Q, INT *IQ,
  double *WORK, INT *IWORK, INT *INFO)

LAPACK_INTERCEPT(mkl_lapack_dgeqrf, "mkl_lapack_dgeqrf",
  call(M, N, A, LDA, TAU, WORK, LWORK, INFO),
  INT *M, INT *N, double *A, INT *LDA, double *TAU,
  double *WORK, INT *LWORK, INT INFO)

/* guessed by `dsyrdb` in MKL - wrong */
/*
LAPACK_INTERCEPT(mkl_lapack_dgerdb, "mkl_lapack_dgerdb",
  call(jobz, n, kd, a, lda, d, e, tau, z, ldz, work, lwork, info),
  char *jobz, INT *n, INT *kd, double *a, INT *lda,
  double *d, double *e, double *tau, double *z, INT *ldz,
  double *work, INT *lwork, INT *info)
*/

/*
LAPACK_INTERCEPT(mkl_lapack_dgerdb, "mkl_lapack_dgerdb",
  call(M, N, A, LDA, D, E, TAUQ, TAUP, WORK, LWORK, INFO),
  INT *M, INT *N, double *A, INT *LDA,
  double *D, double *E, double *TAUQ, double *TAUP,
  double *WORK, INT *LWORK, INT *INFO)
*/

