#include "mkl.h"
#include "stdio.h"
#include "stdlib.h"

int main(int argc, char* argv[]) {

    // Set parameters
    int n = 10;      // length
    int Nv = 5;      // number
    int inc = 1;

    // Allocate space
    MKL_Complex16 *V = (MKL_Complex16 *) malloc(n * Nv * sizeof(MKL_Complex16));

    // Generate a random complex vector matrix
    for (int j = 0; j < Nv; j++) {
        for (int i = 0; i < n; i++) {
            int idx = j * n + i;
            V[idx].real = (double)rand() / RAND_MAX;
            V[idx].imag = (double)rand() / RAND_MAX;
        }
    }

    for (int j = 0; j < Nv; j++) {
        printf("Vector %d first 3 entries: ", j);
        for (int i = 0; i < 3; i++) {
            int idx = j * n + i;
            printf("(%.3f + i%.3f) ", V[idx].real, V[idx].imag);
        }
        printf("\n");
    }

    // Modified Gram–Schmidt (MGS) 
    for (int i = 0; i < Nv; i++) {

        // Calculate the 2-norm of the i-th vector and normalize it
        double norm = dznrm2(&n, &V[i*n], &inc);
        double scale = 1.0 / norm;
        zdscal(&n, &scale, &V[i*n], &inc);

        // Orthogonalization
        for (int j = i + 1; j < Nv; j++) {
            MKL_Complex16 dot;
            zdotc(&dot, &n, &V[i*n], &inc, &V[j*n], &inc); // dot = <vi, vj>
            MKL_Complex16 neg_dot = {-dot.real, -dot.imag}; // 取负
            zaxpy(&n, &neg_dot, &V[i*n], &inc, &V[j*n], &inc); // vj = vj - dot*vi
        }
    }

    // check

    for (int i = 0; i < Nv; i++) {
        for (int j = i + 1; j < Nv; j++) {
            MKL_Complex16 dot;
            zdotc(&dot, &n, &V[i*n], &inc, &V[j*n], &inc);
            printf("<v%d, v%d> = %.3e + i%.3e\n", i, j, dot.real, dot.imag);
        }
    }

    // check norms
    for (int i = 0; i < Nv; i++) {
        double norm = dznrm2(&n, &V[i*n], &inc);
        printf("||v%d|| = %.6f\n", i, norm);
    }

    free(V);
    return 0;
}

