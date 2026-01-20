#include "mkl.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

int main(int argc, char* argv[])
{
    /*
     * Alternative Orthogonalization Method
     * Solve differential equation: dC/ds = (I - C*C^dagger)*C
     * Explicit Euler scheme
     * Initial condition: C(0) = A (random m x m matrix)
     * Find orthonormal matrix B
     */


    MKL_INT m = 5;
    
    // Explicit Euler scheme
    double h = 1e-3;           // Step size
    double delta = 1e-5;       // Tolerance
    int max_iter = 200;        // Maximum iterations
    
    printf("  Matrix size m = %d\n", (int)m);
    printf("  Step size h = %.6e\n", h);
    printf("  Tolerance delta = %.6e\n", delta);
    printf("  Max iterations = %d\n\n", max_iter);
    
    // Allocate matrices
    MKL_Complex16 * A = (MKL_Complex16 *) malloc(m*m*sizeof(MKL_Complex16));
    MKL_Complex16 * C = (MKL_Complex16 *) malloc(m*m*sizeof(MKL_Complex16));
    MKL_Complex16 * CdagC = (MKL_Complex16 *) malloc(m*m*sizeof(MKL_Complex16));
    MKL_Complex16 * I_minus_CdagC = (MKL_Complex16 *) malloc(m*m*sizeof(MKL_Complex16));
    MKL_Complex16 * temp = (MKL_Complex16 *) malloc(m*m*sizeof(MKL_Complex16));
    
    // Initialize A with random complex numbers
    for(int i = 0; i < m*m; i++)
    {
        A[i].real = (2.0*rand())/(1.0*RAND_MAX) - 1.0;
        A[i].imag = (2.0*rand())/(1.0*RAND_MAX) - 1.0;
    }
    
    // C = A 
    for(int i = 0; i < m*m; i++)
    {
        C[i] = A[i];
    }
    
    // BLAS parameters
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_TRANSPOSE noTrans = CblasNoTrans;
    CBLAS_TRANSPOSE conjTrans = CblasConjTrans;
    MKL_Complex16 alpha = (MKL_Complex16) {1.0, 0.0};
    MKL_Complex16 beta = (MKL_Complex16) {0.0, 0.0};
    MKL_Complex16 neg_one = (MKL_Complex16) {-1.0, 0.0};
    MKL_Complex16 h_complex = (MKL_Complex16) {h, 0.0};
    
    printf("Iter | Tr(C*C^dagger) | |Tr(C*C^dagger) - m| | Criterion (m*delta)\n");
    printf("-----|----------------|--------------------|-----------------\n");
    
    // Explicit Euler iteration
    int k;
    double trace_error;
    
    for(k = 0; k < max_iter; k++)
    {
        // C^dagger * C
        // CdagC = C^dagger * C (m x m)
        cblas_zgemm(layout, conjTrans, noTrans, m, m, m,
                    &alpha, C, m, C, m, &beta, CdagC, m);
        
        // trace of C*C^dagger
        MKL_Complex16 trace_CdagC = (MKL_Complex16) {0.0, 0.0};
        for(int i = 0; i < m; i++)
        {
            trace_CdagC.real += CdagC[i*m + i].real;
            trace_CdagC.imag += CdagC[i*m + i].imag;
        }
        
        // |Tr(C*C^dagger) - m|
        trace_error = fabs(trace_CdagC.real - (double)m);
        
        // Print iteration info
        printf("%4d | %14.6e | %18.6e | %.6e\n", 
               k, trace_CdagC.real, trace_error, (double)m * delta);
        
        // Check stopping criterion
    
        if(trace_error <= (double)m * delta)
        {
            printf("\nConverged after %d iterations!\n", k);
            printf("Final |Tr(C*C^dagger) - m| = %.6e\n", trace_error);
            break;
        }
        
        // I - C*C^dagger
        // Initialize I_minus_CdagC as identity matrix I
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < m; j++)
            {
                if(i == j)
                {
                    I_minus_CdagC[i*m + j].real = 1.0;
                    I_minus_CdagC[i*m + j].imag = 0.0;
                }
                else
                {
                    I_minus_CdagC[i*m + j].real = 0.0;
                    I_minus_CdagC[i*m + j].imag = 0.0;
                }
            }
        }
        
        // I_minus_CdagC = I - CdagC
        // Use zaxpy for element-wise operation 
        MKL_INT mm = m * m;
        MKL_INT inc = 1;
        zaxpy(&mm, &neg_one, CdagC, &inc, I_minus_CdagC, &inc);
        
        // (I - C*C^dagger)*C
        // temp = (I - C*C^dagger) * C
        cblas_zgemm(layout, noTrans, noTrans, m, m, m,
                    &alpha, I_minus_CdagC, m, C, m, &beta, temp, m);
        
        // Explicit Euler update
        // C_{k+1} = C_k + h * (I - C*C^dagger)*C
        // C = C + h * temp
        zaxpy(&mm, &h_complex, temp, &inc, C, &inc);
    }
    
    if(k == max_iter)
    {
        printf("\nWarning: Maximum iterations reached without convergence.\n");
        printf("Final |Tr(C*C^dagger) - m| = %.6e\n", trace_error);
        printf("Consider increasing max_iter or decreasing h.\n");
    }
    
    // Check orthonormality
    // C*C^dagger， which should be close to identity
    cblas_zgemm(layout, noTrans, conjTrans, m, m, m,
                &alpha, C, m, C, m, &beta, CdagC, m);
    
    printf("C*C^dagger matrix (should be close to identity):\n");
    for(int i = 0; i < m; i++)
    {
        printf("  Row %d: ", i+1);
        for(int j = 0; j < m; j++)
        {
            printf("(%.4f,%.4f) ", CdagC[i*m + j].real, CdagC[i*m + j].imag);
        }
        printf("\n");
    }
    
    // Check diagonal and off-diagonal errors
    double max_diag_err = 0.0;
    double max_offdiag_err = 0.0;
    
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < m; j++)
        {
            double err;
            if(i == j)
            {
                double err_real = CdagC[i*m + j].real - 1.0;
                double err_imag = CdagC[i*m + j].imag;
                err = sqrt(err_real*err_real + err_imag*err_imag);
                if(err > max_diag_err) max_diag_err = err;
            }
            else
            {
                err = sqrt(pow(CdagC[i*m + j].real, 2) + pow(CdagC[i*m + j].imag, 2));
                if(err > max_offdiag_err) max_offdiag_err = err;
            }
        }
    }
    
    printf("\nMax diagonal error:     %.6e\n", max_diag_err);
    printf("Max off-diagonal error: %.6e\n\n", max_offdiag_err);
    
    if(max_diag_err < 1e-4 && max_offdiag_err < 1e-4)
    {
        printf("SUCCESS: C is approximately orthonormal \n");
    }
    else
    {
        printf("WARNING: C may not be sufficiently orthonormal.\n");
    }
    
    // Free memory
    free(A);
    free(C);
    free(CdagC);
    free(I_minus_CdagC);
    free(temp);

    return 0;
}
