/*
Compile:
$ gcc -shared -fPIC lapack_intercept_sdd2svd.c -o lapack_intercept_sdd2svd.so -ldl

Run:
# LD_PRELOAD=$PWD/lapack_intercept_sdd2svd.so matlab17 -nojvm
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <malloc.h>
#include <dlfcn.h>      // for dlsym()
#include <stddef.h>     // for ptrdiff_t
#include <sys/time.h>   // for gettimeofday()

double tofday(const struct timeval *tv0, const struct timeval *tv1)
{
  return tv1->tv_sec + 1e-6 * tv1->tv_usec
      - (tv0->tv_sec + 1e-6 * tv0->tv_usec);
}

#define buflen 256
char buf[buflen] = "";

int print_mkl_ver()
{
/* For MKL v10.3
  void mkl_serv_getversionstring_c (char* buf, int len);
  mkl_serv_getversionstring_c(buf, buflen-1);
*/
/* For MKL v11.3.2, v2017.0.1, v2018.0.3 (e.g Mathematica)
  void mkl_get_version_string (char* buf, int len);
  mkl_get_version_string(buf, buflen-1);
*/
/* For MKL v2018.0.1 (e.g Matlab), need init MKL first.
  void mkl_serv_get_version_string (char* buf, int len);
  mkl_serv_get_version_string(buf, buflen-1);
*/
  void mkl_serv_get_version_string (char* buf, int len);
  mkl_serv_get_version_string(buf, buflen-1);
  printf("%s\n", buf);
  return 0;
}

/* Matlab integer type */
#define INT ptrdiff_t

int dgesvd_(
  char *JOBU, char *JOBVT, INT *M, INT *N, double *A, INT *LDA,
  double *S, double *U, INT *LDU, double *VT, INT *LDVT,
  double *WORK, INT *LWORK, INT *INFO);

int dgesdd_(
  char *JOBZ, INT *M, INT *N, double *A, INT *LDA,
  double *S, double *U, INT *LDU, double *VT, INT *LDVT,
  double *WORK, INT *LWORK, INT *IWORK, INT *INFO)
{
  //print_mkl_ver();
  printf("call dgesdd ... hand over to dgesvd\n");
  
  struct timeval tv0, tv1;
  gettimeofday(&tv0, NULL);
  int ret = dgesvd_(JOBZ, JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO);
  gettimeofday(&tv1, NULL);
  
  printf("done dgesdd (t = %.3f sec)\n", tofday(&tv0, &tv1));
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

/*
LAPACK_INTERCEPT(dgesdd_, "dgesdd_",
  call(JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, IWORK, INFO),
  char *JOBZ, INT *M, INT *N, double *A, INT *LDA,
  double *S, double *U, INT *LDU, double *VT, INT *LDVT,
  double *WORK, INT *LWORK, INT *IWORK, INT *INFO)
*/

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

/* To crash Matlab/Mathematica, so as to confirm dgerdb is called.
LAPACK_INTERCEPT(mkl_lapack_dgerdb, "mkl_lapack_dgerdb",
  call(jobz, n, kd, a, lda, d, e, tau, z, ldz, work, lwork, info),
  char *jobz, INT *n, INT *kd, double *a, INT *lda,
  double *d, double *e, double *tau, double *z, INT *ldz,
  double *work, INT *lwork, INT *info)
*/
