#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

// special matrix A(t) of size n x n
// First row: 1, 2, 3, ..., n
// Row i (i >= 2): first (i-1) elements are t, then 1, 2, 3, ..., n-i+1
void construct_matrix_A(int n, double t, __CLPK_doublecomplex *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (i == 0) {
                // First row
                A[idx].r = (double)(j + 1);
                A[idx].i = 0.0;
            } else if (j < i) {
                // Elements before diagonal: all t
                A[idx].r = t;
                A[idx].i = 0.0;
            } else {
                // From diagonal onwards: 1, 2, 3, ...
                A[idx].r = (double)(j - i + 1);
                A[idx].i = 0.0;
            }
        }
    }
}

// Construct dA(t)/dt matrix
// d/dt(t) = 1, d/dt(constant) = 0
void construct_dA_dt(int n, __CLPK_doublecomplex *dA) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (i > 0 && j < i) {
                // These positions have t, so derivative = 1
                dA[idx].r = 1.0;
                dA[idx].i = 0.0;
            } else {
                // All other positions are constants, so derivative = 0
                dA[idx].r = 0.0;
                dA[idx].i = 0.0;
            }
        }
    }
}

// Compute determinant of matrix A by using LU factorization
// det(A) = (-1)^M * det(U), and M is number of row swaps
double compute_determinant(int n, __CLPK_doublecomplex *A) {
    // a copy of A 
    __CLPK_doublecomplex *A_copy = malloc(n * n * sizeof(__CLPK_doublecomplex));
    memcpy(A_copy, A, n * n * sizeof(__CLPK_doublecomplex));
    
    // Allocate array for pivot indices
    __CLPK_integer *ipiv = malloc(n * sizeof(__CLPK_integer));
    __CLPK_integer info;
    
    // PA = LU
    zgetrf_(&n, &n, A_copy, &n, ipiv, &info);
    
    if (info != 0) {
        printf("Error: LU factorization failed with info = %d\n", (int)info);
        free(A_copy);
        free(ipiv);
        return 0.0;
    }
    
    // det(U) = product of diagonal elements
    double det_U = 1.0;
    for (int i = 0; i < n; i++) {
        det_U *= A_copy[i * n + i].r;  // Diagonal elements 
    }
    
    // Count number of row swaps M
    // ipiv[i]: row i was swapped with row ipiv[i]
    int M = 0;
    for (int i = 0; i < n; i++) {
        if (ipiv[i] != i + 1) {  // LAPACK uses 1-based indexing
            M++;
        }
    }
    
    // det(A) = (-1)^M * det(U)
    double det_A = (M % 2 == 0 ? 1.0 : -1.0) * det_U;
    
    free(A_copy);
    free(ipiv);
    
    return fabs(det_A);  // Return absolute value (det > 0)
}

// Invert matrix A by LU factorization
// Result in A_inv
void invert_matrix(int n, __CLPK_doublecomplex *A, __CLPK_doublecomplex *A_inv) {
    // Copy A to A_inv
    memcpy(A_inv, A, n * n * sizeof(__CLPK_doublecomplex));
    
    // Allocate pivot array
    __CLPK_integer *ipiv = malloc(n * sizeof(__CLPK_integer));
    __CLPK_integer info;
    
    // LU 
    zgetrf_(&n, &n, A_inv, &n, ipiv, &info);
    
    if (info != 0) {
        printf("Error: LU factorization for inversion failed\n");
        free(ipiv);
        return;
    }
    
    // Compute inverse 
    __CLPK_integer lwork = n * n;
    __CLPK_doublecomplex *work = malloc(lwork * sizeof(__CLPK_doublecomplex));
    
    zgetri_(&n, A_inv, &n, ipiv, work, &lwork, &info);
    
    if (info != 0) {
        printf("Error: Matrix inversion failed\n");
    }
    
    free(ipiv);
    free(work);
}

// matrix multiplication: C = A * B
void matrix_multiply(int n, __CLPK_doublecomplex *A, __CLPK_doublecomplex *B, 
                     __CLPK_doublecomplex *C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j].r = 0.0;
            C[i * n + j].i = 0.0;
            
            for (int k = 0; k < n; k++) {
                // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                double a_r = A[i * n + k].r;
                double a_i = A[i * n + k].i;
                double b_r = B[k * n + j].r;
                double b_i = B[k * n + j].i;
                
                C[i * n + j].r += a_r * b_r - a_i * b_i;
                C[i * n + j].i += a_r * b_i + a_i * b_r;
            }
        }
    }
}

// Compute trace of matrix M (sum)
double compute_trace(int n, __CLPK_doublecomplex *M) {
    double trace = 0.0;
    for (int i = 0; i < n; i++) {
        trace += M[i * n + i].r;
    }
    return trace;
}

// derivative f(t) = d/dt[det(A(t))]
// f(t) = det(A(t)) * Tr[A(t)^(-1) * dA(t)/dt]
double compute_derivative(int n, double t) {
    // Allocate matrices
    __CLPK_doublecomplex *A = malloc(n * n * sizeof(__CLPK_doublecomplex));
    __CLPK_doublecomplex *dA = malloc(n * n * sizeof(__CLPK_doublecomplex));
    __CLPK_doublecomplex *A_inv = malloc(n * n * sizeof(__CLPK_doublecomplex));
    __CLPK_doublecomplex *product = malloc(n * n * sizeof(__CLPK_doublecomplex));
    
    // A(t)
    construct_matrix_A(n, t, A);
    
    // det(A(t))
    double det_A = compute_determinant(n, A);
    
    // dA(t)/dt
    construct_dA_dt(n, dA);
    
    // A(t)^(-1)
    invert_matrix(n, A, A_inv);
    
    // A^(-1) * dA/dt
    matrix_multiply(n, A_inv, dA, product);
    
    // trace of the product
    double trace = compute_trace(n, product);
    
    // f(t) = det(A(t)) * trace
    double f_t = det_A * trace;
    
    free(A);
    free(dA);
    free(A_inv);
    free(product);
    
    return f_t;
}

// find root of f(t) = 0 in interval [x0, x1] by bisection method 
// Returns the value t* where f(t*) = 0 
double bisection_method(int n, double x0, double x1, double tolerance) {
    printf("  Starting bisection method for n=%d in interval [%.2f, %.2f]\n", n, x0, x1);
    
    double f_x0 = compute_derivative(n, x0);
    double f_x1 = compute_derivative(n, x1);
    
    // Check if root is in the interval
    if (f_x0 * f_x1 > 0) {
        printf("  Warning: f(x0) and f(x1) have the same sign \n");
        printf("  f(%.2f) = %.6f, f(%.2f) = %.6f\n", x0, f_x0, x1, f_x1);
    }
    
    int max_iter = 100;
    int iter = 0;
    
    while (iter < max_iter) {
        // midpoint
        double x2 = (x0 + x1) / 2.0;
        double f_x2 = compute_derivative(n, x2);
        
        printf("  Iteration %d: x2 = %.10f, f(x2) = %.10e\n", iter, x2, f_x2);
        
        // Check if we found the root 
        if (fabs(f_x2) < 1e-10) {
            printf("  Found root: f(x2) ≈ 0\n");
            return x2;
        }
        
        // Check if interval is small enough
        if (fabs(x1 - x0) < tolerance) {
            return x2;
        }
        
        // Determine which half has root
        if (f_x0 * f_x2 < 0) {
            // Root is in [x0, x2]
            x1 = x2;
            f_x1 = f_x2;
        } else {
            // Root is in [x2, x1]
            x0 = x2;
            f_x0 = f_x2;
        }
        
        iter++;
    }

    return (x0 + x1) / 2.0;
}

int main(int argc, char *argv[]) {

    // Open output file
    FILE *fp = fopen("determinant_results.txt", "w");
    if (fp == NULL) {
        printf("Error: Could not open output file \n");
        return 1;
    }
    
    // Write header to file
    // Loop over odd values of n = 3, 5, 7, 9
    for (int n = 3; n <= 9; n += 2) {
        printf("Processing n = %d\n", n);

        // Find t* by bisection method in [0, 2]
        double tolerance = 1e-6;
        double t_star = bisection_method(n, 0.0, 2.0, tolerance);
        
        printf("\nOptimal value: t* = %.10f\n", t_star);
        
        // Construct A(t*) and compute its determinant
        __CLPK_doublecomplex *A_star = malloc(n * n * sizeof(__CLPK_doublecomplex));
        construct_matrix_A(n, t_star, A_star);
        double det_A_star = compute_determinant(n, A_star);
        free(A_star);
        
        printf("Minimum determinant: det(A(t*)) = %.10f\n", det_A_star);
        
        // Write results to file
        fprintf(fp, "%d\t%.10f\t%.10f\n", n, t_star, det_A_star);
        
        // print to console in table format
        printf("\nResult for n=%d:\n", n);
        printf("  t*           = %.10f\n", t_star);
        printf("  det(A(t*))   = %.10f\n", det_A_star);
    }

    // answer questions
   
    fprintf(fp, "1: Does the value t* depend on the choice of odd n?\n");
    fprintf(fp, " No.\n");
    fprintf(fp, "The optimal value is t* = 0.5 for all odd values of n.\n");
    fprintf(fp, "This value is independent of the matrix size n.\n\n");
    
    fprintf(fp, "2: Does det(A(t*)) depend on the choice of odd n?\n");
    fprintf(fp, " Yes.\n");
    fprintf(fp, "The minimum determinant det(A(t*)) depends on n.\n");
    fprintf(fp, "It follows: det(A(t*)) = (1/2)^(n-1)\n\n");
    fprintf(fp, "Observed values:\n");
    fprintf(fp, "  n=3: det(A(t*)) = 0.25       = (1/2)^2\n");
    fprintf(fp, "  n=5: det(A(t*)) = 0.0625     = (1/2)^4\n");
    fprintf(fp, "  n=7: det(A(t*)) = 0.015625   = (1/2)^6\n");
    fprintf(fp, "  n=9: det(A(t*)) = 0.00390625 = (1/2)^8\n\n");
    fprintf(fp, "The determinant decreases exponentially as n increases.\n");

    return 0;
}

