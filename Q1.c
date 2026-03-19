/*
 * GCR (Generalized Conjugate Residual) method for Hermitian matrices
 * Solves Ax = b where A is a 200x200 complex Hermitian matrix
 * Uses BLAS routines for all linear algebra operations
 * Outputs: residual norm and true error norm at each iteration
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

/* BLAS declarations */
extern void   zcopy_(int*, double complex*, int*, double complex*, int*);          /* y <- x                                      */
extern void   zaxpy_(int*, double complex*, double complex*, int*,                 /* y <- y + alpha * x                          */

                     double complex*, int*);
extern void   zscal_(int*, double complex*, double complex*, int*);                /* x <- alpha * x                              */
extern double dznrm2_(int*, double complex*, int*);                                /* ||x||_2                                     */
extern void   zdotc_(double complex*, int*, double complex*, int*,                 /* alpha <- x^H * y   (conjugate transpose)    */
                     double complex*, int*);
extern void   zgemv_(char*, int*, int*, double complex*, double complex*, int*,    /* y <- alpha * A * x + beta * y               */
                     double complex*, int*, double complex*, double complex*, int*);

#define N 200

int main(void)
{
    int i, j;
    int n   = N;
    int one = 1;

    /* 1. Allocate memory for all vectors and matrices                     */
    double complex *A      = malloc(n * n * sizeof(double complex)); /* system matrix A          */
    double complex *Atmp   = malloc(n * n * sizeof(double complex)); /* temp storage for A†      */
    double complex *x_true = malloc(n     * sizeof(double complex)); /* true solution vector     */
    double complex *b      = malloc(n     * sizeof(double complex)); /* right-hand side          */
    double complex *x      = malloc(n     * sizeof(double complex)); /* current iterate x_j      */
    double complex *r      = malloc(n     * sizeof(double complex)); /* residual r_j             */
    double complex *w      = malloc(n     * sizeof(double complex)); /* working vector           */
    double complex *Aw     = malloc(n     * sizeof(double complex)); /* A*w                      */
    double complex *Ar     = malloc(n     * sizeof(double complex)); /* A*r_j for alpha          */
    double complex *P      = malloc(n * n * sizeof(double complex)); /* basis vectors p_j (cols) */
    double complex *AP     = malloc(n * n * sizeof(double complex)); /* A*p_j stored as cols     */

    double complex one_c  =  1.0 + 0.0*I;
    double complex zero_c =  0.0 + 0.0*I;
    double complex neg_c  = -1.0 + 0.0*I;
    double complex half   =  0.5 + 0.0*I;

    /* 2. Build random complex Hermitian matrix: A = 0.5*(A + A†)         */
    srand((unsigned int)time(NULL));

    /* Fill A with random complex entries */
    for (i = 0; i < n * n; i++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        A[i] = re + im * I;
    }

    /* Compute conjugate transpose into Atmp */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            Atmp[j*n + i] = conj(A[i*n + j]);

    /* A = A + Atmp, then scale by 0.5 to symmetrise */
    int nn = n * n;
    zaxpy_(&nn, &one_c, Atmp, &one, A, &one);
    zscal_(&nn, &half,  A,    &one);

    /* 3. Build random complex true solution x_true, compute b = A*x_true */
    for (i = 0; i < n; i++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        x_true[i] = re + im * I;
    }

    /* b = A * x_true using zgemv */
    zgemv_("N", &n, &n, &one_c, A, &n, x_true, &one, &zero_c, b, &one);

    /*  4. Initialise: x0 = 0, r0 = b, p0 = r0 / ||A*r0||                 */
    for (i = 0; i < n; i++) x[i] = zero_c;

    /* r = b (since x0 = 0, r0 = b - A*x0 = b) */
    zcopy_(&n, b, &one, r, &one);

    /* Compute A*r0 to normalise p0 */
    zgemv_("N", &n, &n, &one_c, A, &n, r, &one, &zero_c, Ar, &one);
    double norm_Ar = dznrm2_(&n, Ar, &one);

    /* p0 = r0 / ||A*r0||, stored as column 0 of P */
    zcopy_(&n, r, &one, &P[0], &one);
    double complex scale = 1.0 / norm_Ar + 0.0*I;
    zscal_(&n, &scale, &P[0], &one);

    /* AP[:,0] = A * p0 */
    zgemv_("N", &n, &n, &one_c, A, &n, &P[0], &one, &zero_c, &AP[0], &one);

    /* 5. Open output file                                                 */
    FILE *fp = fopen("gcr_output.dat", "w");
    fprintf(fp, "# iter   residual_norm   true_error_norm\n");

    /* 6. GCR main loop: at most N iterations                             */
    for (j = 0; j < n; j++) {

        /* alpha_j = p_j† * A * r_j = (A*p_j)† * r_j  (A hermitian)     */
        double complex alpha_j;
        zdotc_(&alpha_j, &n, &AP[j*n], &one, r, &one);

        /* x_{j+1} = x_j + alpha_j * p_j */
        zaxpy_(&n, &alpha_j, &P[j*n], &one, x, &one);

        /* r_{j+1} = r_j - alpha_j * A*p_j */
        double complex neg_alpha = -alpha_j;
        zaxpy_(&n, &neg_alpha, &AP[j*n], &one, r, &one);

        /* Compute and save residual norm and true error norm   */
        double res_norm = dznrm2_(&n, r, &one);

        /* reuse w as temp: w = x - x_true */
        zcopy_(&n, x, &one, w, &one);
        zaxpy_(&n, &neg_c, x_true, &one, w, &one);
        double err_norm = dznrm2_(&n, w, &one);

        fprintf(fp, "%d  %.15e  %.15e\n", j+1, res_norm, err_norm);
        printf("iter %3d: ||r|| = %.6e  ||e|| = %.6e\n",
               j+1, res_norm, err_norm);

        /* Check convergence */
        if (res_norm < 1e-12) {
            printf("Converged at iteration %d\n", j+1);
            break;
        }

        /* Build p_{j+1} via orthogonalisation                 */

        /* w = r_{j+1} */
        zcopy_(&n, r, &one, w, &one);

        /* Aw = A * w */
        zgemv_("N", &n, &n, &one_c, A, &n, w, &one, &zero_c, Aw, &one);

        /* Inner loop: w = w - (p_i† * A * A * w) * p_i  for i = 0..j  */
        /* Use (A*p_i)† * (A*w) to avoid extra matvec                   */
        for (i = 0; i <= j; i++) {
            double complex coeff;
            zdotc_(&coeff, &n, &AP[i*n], &one, Aw, &one);

            double complex neg_coeff = -coeff;
            /* w  = w  - coeff * p_i  */
            zaxpy_(&n, &neg_coeff, &P[i*n],  &one, w,  &one);
            /* Aw = Aw - coeff * Ap_i (keep Aw consistent with w) */
            zaxpy_(&n, &neg_coeff, &AP[i*n], &one, Aw, &one);
        }

        /* Check for lucky breakdown: ||Aw|| = 0 */
        double norm_Aw = dznrm2_(&n, Aw, &one);
        if (norm_Aw < 1e-14) {
            printf("Lucky breakdown at iteration %d\n", j+1);
            break;
        }

        /* p_{j+1} = w / ||Aw|| */
        scale = 1.0 / norm_Aw + 0.0*I;
        zcopy_(&n, w,  &one, &P[(j+1)*n],  &one);
        zscal_(&n, &scale,   &P[(j+1)*n],  &one);

        /* AP[:,j+1] = A*p_{j+1} = Aw / ||Aw|| */
        zcopy_(&n, Aw, &one, &AP[(j+1)*n], &one);
        zscal_(&n, &scale,   &AP[(j+1)*n], &one);
    }

    fclose(fp);

    /* 7. Free all allocated memory                                        */
    free(A); free(Atmp); free(x_true); free(b);
    free(x); free(r); free(w); free(Aw); free(Ar);
    free(P); free(AP);

    return 0;
}
