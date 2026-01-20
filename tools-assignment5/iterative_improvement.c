#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

// Compute complex vector's 2-norm 
double complex_norm(__CLPK_integer n, __CLPK_doublecomplex *x) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += x[i].r * x[i].r + x[i].i * x[i].i;
    }
    return sqrt(norm);
}

// Copy complex vector
void copy_complex_vector(__CLPK_integer n, __CLPK_doublecomplex *src, __CLPK_doublecomplex *dst) {
    for (int i = 0; i < n; i++) {
        dst[i].r = src[i].r;
        dst[i].i = src[i].i;
    }
}

// result = A*x - b (for residual calculation)
void compute_residual(__CLPK_integer n, __CLPK_doublecomplex *A, __CLPK_doublecomplex *x, 
                      __CLPK_doublecomplex *b, __CLPK_doublecomplex *result) {
    // Initialize result = -b
    for (int i = 0; i < n; i++) {
        result[i].r = -b[i].r;
        result[i].i = -b[i].i;
    }
    
    // Compute A*x and add to result
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            double real_part = A[idx].r * x[j].r - A[idx].i * x[j].i;
            double imag_part = A[idx].r * x[j].i + A[idx].i * x[j].r;
            result[i].r += real_part;
            result[i].i += imag_part;
        }
    }
}

int main(int argc, char *argv[]) {
    FILE *fp;
    
    // Open output file for writing
    fp = fopen("iterations.txt", "w");
    if (fp == NULL) {
        printf("Error opening file \n");
        return 1;
    }
    
    // random number
    srand(12345);
    
    // Loop different problem sizes
    for (int n = 2; n <= 500; n++) {
        printf("Processing n = %d...\n", n);
        
        // Allocate memory 
        __CLPK_doublecomplex *A = malloc(n * n * sizeof(__CLPK_doublecomplex));
        __CLPK_doublecomplex *A_LU = malloc(n * n * sizeof(__CLPK_doublecomplex));
        __CLPK_doublecomplex *x_true = malloc(n * sizeof(__CLPK_doublecomplex));
        __CLPK_doublecomplex *b = malloc(n * sizeof(__CLPK_doublecomplex));
        __CLPK_doublecomplex *x_approx = malloc(n * sizeof(__CLPK_doublecomplex));
        __CLPK_integer *ipiv = malloc(n * sizeof(__CLPK_integer));
        
        // Fill matrix A with random complex values
        for (int i = 0; i < n * n; i++) {
            A[i].r = (double)rand() / RAND_MAX;
            A[i].i = (double)rand() / RAND_MAX;
        }
        
        // Fill true solution x_true with random complex values
        for (int i = 0; i < n; i++) {
            x_true[i].r = (double)rand() / RAND_MAX;
            x_true[i].i = (double)rand() / RAND_MAX;
        }
        
        // Compute b = A * x_true
        for (int i = 0; i < n; i++) {
            b[i].r = 0.0;
            b[i].i = 0.0;
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;

                b[i].r += A[idx].r * x_true[j].r - A[idx].i * x_true[j].i;
                b[i].i += A[idx].r * x_true[j].i + A[idx].i * x_true[j].r;
            }
        }
        
        // Copy A to A_LU 
        memcpy(A_LU, A, n * n * sizeof(__CLPK_doublecomplex));
        
        // Compute LU factorization of A
        __CLPK_integer info;
        zgetrf_(&n, &n, A_LU, &n, ipiv, &info);
        
        if (info != 0) {
            printf("LU factorization failed for n=%d, info=%d\n", n, (int)info);
            free(A); free(A_LU); free(x_true); free(b); free(x_approx); free(ipiv);
            continue;
        }
        
        // Solve Ax = b with LU factorization 
        copy_complex_vector(n, b, x_approx);
        char trans = 'N';
        __CLPK_integer nrhs = 1;
        zgetrs_(&trans, &n, &nrhs, A_LU, &n, ipiv, x_approx, &n, &info);
        
        if (info != 0) {
            printf("Initial solve failed for n=%d\n", n);
            free(A); free(A_LU); free(x_true); free(b); free(x_approx); free(ipiv);
            continue;
        }
        
        // Iterative improvement
        int max_iter = 50;
        int iter_count = 0;
        double prev_rel_error = 1e10;
        double prev_rel_residual = 1e10;
        
        __CLPK_doublecomplex *residual = malloc(n * sizeof(__CLPK_doublecomplex));
        __CLPK_doublecomplex *error_vec = malloc(n * sizeof(__CLPK_doublecomplex));
        __CLPK_doublecomplex *delta = malloc(n * sizeof(__CLPK_doublecomplex));
        
        for (int iter = 0; iter < max_iter; iter++) {
            // residual r = A*x_approx - b
            compute_residual(n, A, x_approx, b, residual);
            
            // relative residual: ||r||_2 / ||b||_2
            double norm_residual = complex_norm(n, residual);
            double norm_b = complex_norm(n, b);
            double rel_residual = norm_residual / norm_b;
            
            // error vector: x_approx - x_true
            for (int i = 0; i < n; i++) {
                error_vec[i].r = x_approx[i].r - x_true[i].r;
                error_vec[i].i = x_approx[i].i - x_true[i].i;
            }
            
            // relative error: ||x_approx - x_true||_2 / ||x_true||_2
            double norm_error = complex_norm(n, error_vec);
            double norm_x_true = complex_norm(n, x_true);
            double rel_error = norm_error / norm_x_true;
            
            // Check stopping condition: both metrics stop decrease
            if (iter > 0 && rel_error >= prev_rel_error && rel_residual >= prev_rel_residual) {
                iter_count = iter;
                break;
            }
            
            // Update 
            prev_rel_error = rel_error;
            prev_rel_residual = rel_residual;
            
            // Solve A*delta = residual by using now LU factorization
            copy_complex_vector(n, residual, delta);
            zgetrs_(&trans, &n, &nrhs, A_LU, &n, ipiv, delta, &n, &info);
            
            // Update solution: x_approx = x_approx - delta
            for (int i = 0; i < n; i++) {
                x_approx[i].r -= delta[i].r;
                x_approx[i].i -= delta[i].i;
            }
            
            // If we reach max iterations
            if (iter == max_iter - 1) {
                iter_count = max_iter;
            }
        }
        
        // Write results to file
        fprintf(fp, "%d %d\n", n, iter_count);
        
        // Free memory
        free(residual);
        free(error_vec);
        free(delta);
        free(A);
        free(A_LU);
        free(x_true);
        free(b);
        free(x_approx);
        free(ipiv);
    }
    
    fclose(fp);
    
    return 0;
}
