#include "mkl.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    // Define the dimension of each vector and the number of vectors
    int n = 200;  
    int N = 10;   

    // Allocate memory for all vectors 
    MKL_Complex16 *V = (MKL_Complex16 *) malloc(n * N * sizeof(MKL_Complex16));
    if (V == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Initialize random seed
    srand(time(NULL));

    // Fill each vector with random complex numbers
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < n; i++) {
            int idx = j * n + i;  
            V[idx].real = (double)rand() / RAND_MAX;
            V[idx].imag = (double)rand() / RAND_MAX;
        }
    }

    // Print a few entries to verify
    printf("Example entries from the first few vectors:\n");
    for (int j = 0; j < 2; j++) {
        printf("Vector %d first 5 entries:\n", j);
        for (int i = 0; i < 5; i++) {
            int idx = j * n + i;
            printf("  V[%d][%d] = %.5f + i%.5f\n", i, j, V[idx].real, V[idx].imag);
        }
        printf("\n");
    }

    free(V);
    return 0;
}
