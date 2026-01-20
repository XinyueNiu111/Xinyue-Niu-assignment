#include "mkl.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

int main(int argc, char* argv[])
{
    /*
     * Arnoldi Iteration Implementation
     * Matrix A: m x m
     * Vector b: initial vector
     * N: iteration times (N << m)
     * Goal: Build orthonormal basis for Krylov subspace
     */
    

    MKL_INT m = 100;
    MKL_INT N = 10;
    MKL_INT inc = 1;
    
    printf("Arnoldi Iteration: m=%d, N=%d\n", (int)m, (int)N);
    
    // Allocate arrays
    MKL_Complex16 * A = (MKL_Complex16 *) malloc(m*m*sizeof(MKL_Complex16));
    MKL_Complex16 * b = (MKL_Complex16 *) malloc(m*sizeof(MKL_Complex16));
    MKL_Complex16 * Q = (MKL_Complex16 *) malloc(m*(N+1)*sizeof(MKL_Complex16));
    MKL_Complex16 * H = (MKL_Complex16 *) malloc((N+1)*N*sizeof(MKL_Complex16));
    MKL_Complex16 * v = (MKL_Complex16 *) malloc(m*sizeof(MKL_Complex16));
    
    // Initialize A with random complex numbers
    for(int i = 0; i < m*m; i++)
    {
        A[i].real = (2.0*rand())/(1.0*RAND_MAX) - 1.0;
        A[i].imag = (2.0*rand())/(1.0*RAND_MAX) - 1.0;
    }
    
    // Initialize b 
    for(int i = 0; i < m; i++)
    {
        b[i].real = (2.0*rand())/(1.0*RAND_MAX) - 1.0;
        b[i].imag = (2.0*rand())/(1.0*RAND_MAX) - 1.0;
    }
    
    // Initialize H to zero
    for(int i = 0; i < (N+1)*N; i++)
    {
        H[i].real = 0.0;
        H[i].imag = 0.0;
    }
    
    // q1 = b/||b||_2
    double norm_b = dznrm2(&m, b, &inc);
    
    MKL_INT incQ = N + 1;
    zcopy(&m, b, &inc, &(Q[0]), &incQ);
    
    // Normalize: q1 = b/||b||_2
    double fac = 1.0/norm_b;
    zdscal(&m, &fac, &(Q[0]), &incQ);
    
    printf("Initial vector q1 computed, ||q1|| = %.15e\n\n", dznrm2(&m, &(Q[0]), &incQ));
    
    // Set the layout as row major for C
    CBLAS_LAYOUT layout = CblasRowMajor;
    // Leading dimension for A
    MKL_INT lda = m;
    // Specify no transposition of the matrix
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    
    // Scalars for matrix-vector product
    MKL_Complex16 alpha = (MKL_Complex16) {1.0, 0.0};
    MKL_Complex16 beta = (MKL_Complex16) {0.0, 0.0};
    
    for(int n = 0; n < N; n++)  
    {
        printf("  Iteration n = %d\n", n+1);
        
        // v = A*qn
        // qn is the n-th column of Q, starts at Q[n] with increment incQ
        cblas_zgemv(layout, trans, m, m, &alpha, A, lda, &(Q[n]), incQ, &beta, v, inc);
        
        for(int j = 0; j <= n; j++)  
        {
            // hjn = qj^dagger * v
            MKL_Complex16 hjn;
            zdotc(&hjn, &m, &(Q[j]), &incQ, v, &inc);
            
            // Store hjn in H
            // H is (N+1) x N matrix in row-major: H[j*N + n]
            H[j*N + n] = hjn;
            
            // v = v - hjn*qj
            MKL_Complex16 neg_hjn;
            neg_hjn.real = -hjn.real;
            neg_hjn.imag = -hjn.imag;
            zaxpy(&m, &neg_hjn, &(Q[j]), &incQ, v, &inc);
        }
        
        // h_{n+1,n} = ||v||_2
        double hn1n = dznrm2(&m, v, &inc);
        H[(n+1)*N + n].real = hn1n;
        H[(n+1)*N + n].imag = 0.0;
        
        // qn+1 = v/h_{n+1,n}
        zcopy(&m, v, &inc, &(Q[n+1]), &incQ);
        fac = 1.0/hn1n;
        zdscal(&m, &fac, &(Q[n+1]), &incQ);
        
        printf("    h_{%d,%d} = %.15e\n", n+2, n+1, hn1n);
    }

    
    // Arnoldi Relation A*QN = Q_{N+1}*H
    
    // Allocate matrices 
    MKL_Complex16 * QN = (MKL_Complex16 *) malloc(m*N*sizeof(MKL_Complex16));
    MKL_Complex16 * QN1 = (MKL_Complex16 *) malloc(m*(N+1)*sizeof(MKL_Complex16));
    MKL_Complex16 * AQN = (MKL_Complex16 *) malloc(m*N*sizeof(MKL_Complex16));
    MKL_Complex16 * QN1H = (MKL_Complex16 *) malloc(m*N*sizeof(MKL_Complex16));
    
    // Extract QN 
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < N; j++)
        {
            QN[i*N + j] = Q[i*(N+1) + j];
        }
    }
    
    // Extract QN1 
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j <= N; j++)
        {
            QN1[i*(N+1) + j] = Q[i*(N+1) + j];
        }
    }
    
    // use cblas_zgemm to compute A*QN 
    // A: m x m, QN: m x N, result: m x N
    cblas_zgemm(layout, CblasNoTrans, CblasNoTrans, m, N, m,
                &alpha, A, m, QN, N, &beta, AQN, N);
    
    // Q_{N+1}*H 
    // QN1: m x (N+1), H: (N+1) x N, result: m x N
    cblas_zgemm(layout, CblasNoTrans, CblasNoTrans, m, N, N+1,
                &alpha, QN1, N+1, H, N, &beta, QN1H, N);
    
    // Compute the difference AQN - QN1H
    MKL_Complex16 neg_one = (MKL_Complex16) {-1.0, 0.0};
    MKL_INT mn = m*N;
    zaxpy(&mn, &neg_one, QN1H, &inc, AQN, &inc);
    
    // Compute Frobenius norm of the difference
    double diff_norm = dznrm2(&mn, AQN, &inc);
    
    printf("||A*QN - Q_{N+1}*H||_F = %.15e\n", diff_norm);
    
    if(diff_norm < 1e-10)
    {
        printf("SUCCESS: Arnoldi relation verified \n\n");
    }
    else
    {
        printf("WARNING: Large error in Arnoldi relation.\n\n");
    }
    
    // Orthonormality Q^dagger * Q = I
    
    // Q^dagger * Q
    // Q: m x (N+1), result: (N+1) x (N+1)
    MKL_Complex16 * QtQ = (MKL_Complex16 *) malloc((N+1)*(N+1)*sizeof(MKL_Complex16));
    
    cblas_zgemm(layout, CblasConjTrans, CblasNoTrans, N+1, N+1, m,
                &alpha, QN1, N+1, QN1, N+1, &beta, QtQ, N+1);
    
    // Check: if QtQ is identity matrix
    printf("Checking Q^dagger*Q = I:\n");
    printf("Sample diagonal elements (should be 1.0):\n");
    
    double max_diag_err = 0.0;
    double max_offdiag_err = 0.0;
    
    for(int i = 0; i <= N; i++)
    {
        for(int j = 0; j <= N; j++)
        {
            double err_real, err_imag, err;
            
            if(i == j)
            {
                err_real = QtQ[i*(N+1) + j].real - 1.0;
                err_imag = QtQ[i*(N+1) + j].imag;
                err = sqrt(err_real*err_real + err_imag*err_imag);
                if(err > max_diag_err) max_diag_err = err;
                
                // Print first 3 and last diagonal elements
                if(i < 3 || i == N)
                {
                    printf("  Q^dagger*Q[%d,%d] = (%.6e, %.6e), error = %.6e\n",
                           i+1, j+1, QtQ[i*(N+1) + j].real, QtQ[i*(N+1) + j].imag, err);
                }
            }
            else
            {
                err_real = QtQ[i*(N+1) + j].real;
                err_imag = QtQ[i*(N+1) + j].imag;
                err = sqrt(err_real*err_real + err_imag*err_imag);
                if(err > max_offdiag_err) max_offdiag_err = err;
            }
        }
    }
    
    printf("\nMax diagonal error:     %.6e\n", max_diag_err);
    printf("Max off-diagonal error: %.6e\n\n", max_offdiag_err);
    
    if(max_diag_err < 1e-10 && max_offdiag_err < 1e-10)
    {
        printf("SUCCESS: Q vectors are orthonormal \n\n");
    }
    else
    {
        printf("WARNING: Q vectors may not be perfectly orthonormal.\n\n");
    }
    
    // Q^dagger * qi = ei
    
    for(int i = 0; i <= N; i++)
    {
        // qi is the i-th column of Q
        MKL_Complex16 * result = (MKL_Complex16 *) malloc((N+1)*sizeof(MKL_Complex16));
        
        cblas_zgemv(layout, CblasConjTrans, m, N+1, &alpha, QN1, N+1,
                    &(Q[i]), incQ, &beta, result, inc);
        
        // Check: if result is ei (1 at position i, 0 elsewhere)
        double max_err = 0.0;
        for(int j = 0; j <= N; j++)
        {
            double expected_real = (j == i) ? 1.0 : 0.0;
            double err = sqrt(pow(result[j].real - expected_real, 2) + pow(result[j].imag, 2));
            if(err > max_err) max_err = err;
        }
        
        if(i < 3 || i == N)
        {
            printf("  Q^dagger*q%d: max error = %.6e\n", i+1, max_err);
        }
        
        free(result);
    }
    
    printf("\n");
    
    free(A);
    free(b);
    free(Q);
    free(H);
    free(v);
    free(QN);
    free(QN1);
    free(AQN);
    free(QN1H);
    free(QtQ);
    
    return 0;
}

