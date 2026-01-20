#include "mkl.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

int main(int argc, char* argv[])
{
    /*
     * BLAS parameters for triangular matrix multiplication
     * cblas_ztrmm: B = alpha * op(A) * B
     * A is triangular
     */
    CBLAS_LAYOUT layout = CblasRowMajor;      // Row-major storage 
    CBLAS_SIDE side = CblasLeft;              // A is on the left: A*B
    CBLAS_UPLO uplo = CblasUpper;             // A is upper triangular
    CBLAS_TRANSPOSE trans = CblasNoTrans;     // No transpose
    CBLAS_DIAG diag = CblasNonUnit;           // Diagonal is not unit 
    
    MKL_Complex16 alpha = (MKL_Complex16) {1.0, 0.0};  // alpha = 1
    
    for(int m = 2; m <= 50; m++)
    {
        // Build triangular matrix A (m x m)
        // Diagonal: zeros
        // Upper triangle: random non-zero complex numbers
        
        MKL_Complex16 * A = (MKL_Complex16 *) malloc(m*m*sizeof(MKL_Complex16));
        
        // Fill A with zero diagonal
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < m; j++)
            {
                if(i < j)  // Upper triangular part 
                {
                    // Random non-zero complex number
                    A[i*m + j].real = (2.0*rand())/(1.0*RAND_MAX) - 1.0;
                    A[i*m + j].imag = (2.0*rand())/(1.0*RAND_MAX) - 1.0;
                }
                else  // Diagonal (i==j) and lower part (i>j)
                {
                    A[i*m + j].real = 0.0;
                    A[i*m + j].imag = 0.0;
                }
            }
        }
        
        // Allocate Result matrix
        // Result will store A^1, A^2, A^3, ...
        
        MKL_Complex16 * Result = (MKL_Complex16 *) malloc(m*m*sizeof(MKL_Complex16));
        
        // Initialize Result = A
        memcpy(Result, A, m*m*sizeof(MKL_Complex16));
        
        // Find N: A^N = 0
        
        int N = 1;  
        int is_zero = 0;  
        
        for(N = 1; N <= m; N++)
        {
            is_zero = 1;  
            
            for(int i = 0; i < m*m; i++)
            {
                // Check if any element is non-zero
                if(Result[i].real != 0.0 || Result[i].imag != 0.0)
                {
                    is_zero = 0;  // Found non-zero element
                    break;
                }
            }
            
            if(is_zero)
            {
                break;
            }
            
            // Compute Result = A * Result by BLAS3
            // B = alpha * A * B
            // A is triangular, B is general
          
            cblas_ztrmm(layout, side, uplo, trans, diag, m, m, &alpha, A, m, Result, m);
        }
        
        printf("m = %2d: N = %2d (A^%d = 0)\n", m, N, N);
        
        // Free memory 
        free(A);
        free(Result);
    }
    
    return 0;
}
