#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#include <mkl.h>
#include "mmio.h"


void readMtx(const char *filename, int *n, int *nnz, double **values, int **columns, int **rowPointers) {
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;

    if ((f = fopen(filename, "r")) == NULL) {
        perror("Could not open file");
        exit(1);
    }
    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        printf("Could not process Matrix Market size.\n");
        exit(1);
    }

    *n = N;
    *nnz = nz;

    *values = (double *)malloc(nz * sizeof(double));
    *columns = (int *)malloc(nz * sizeof(int));
    *rowPointers = (int *)malloc((M+1) * sizeof(int));

    // Temporary arrays to hold the data before converting to CSR
    std::vector<std::vector<std::pair<int, double>>> tempData(M);

    // Read the data
    for (int i = 0; i < nz; i++) {
        int row, col;
        double val;
        fscanf(f, "%d %d %lg\n", &row, &col, &val);
        row--; // Convert to 0-based indexing
        col--;
        tempData[row].push_back({col, val});
    }

    // Convert to CSR format
    int currentIdx = 0;
    (*rowPointers)[0] = 0;
    for (int i = 0; i < M; i++) {
        for (const auto &pair : tempData[i]) {
            (*columns)[currentIdx] = pair.first;
            (*values)[currentIdx] = pair.second;
            currentIdx++;
        }
        (*rowPointers)[i+1] = currentIdx;
    }

    fclose(f);
}

int main(int argc, char **argv){

    std::string filename = "";
    std::string bmatrix = "";
    int repLoops = 1000;
    int threads = 1;

    if (argc == 2)
    {
        filename = "A.mtx";
        bmatrix = "./vecb2.mtx";
        // filename = "./circuit_2/circuit_2.mtx";
        // bmatrix = "./circuit_2/vecb.mtx";
        // filename = "./scircuit/scircuit.mtx";
        // bmatrix = "./scircuit/vecb.mtx";
    }

    if (argc == 6)
    {
        filename = argv[2];
        bmatrix = argv[3];
        repLoops = atoi(argv[4]);
        threads = atoi(argv[5]);
    }

    // print threads
    // std::cout << "Threads: " << threads << std::endl;
    mkl_set_dynamic(0);
    mkl_set_num_threads(threads);

    /* Solver parameters */
    MKL_INT mtype = 11;       /* Real unsymmetric matrix */
    MKL_INT nrhs = 1;         /* Number of right hand sides. */
    void *pt[64] = {0};       /* Internal solver memory pointer pt, */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    MKL_INT iparm[64] = {0};  /* PARDISO control parameters. */
    MKL_INT maxfct, mnum, phase, error, msglvl;
    /* Auxiliary variables. */
    MKL_INT i;
    double ddum; /* Double dummy */
    MKL_INT idum; /* Integer dummy. */


    using Chrono = std::chrono::time_point<std::chrono::high_resolution_clock>;
    Chrono begin[3], end[3];
    double total[3] = {0};
    

    /* Setup Pardiso control parameters. */
    iparm[0] = 1;  /* No solver default */
    iparm[1] = 0;  /* Fill-in reordering from AMD */
    iparm[3] = 0;  /* No iterative-direct algorithm */
    iparm[4] = 0;  /* No user fill-in reducing permutation */
    iparm[5] = 0;  /* Write solution into x */
    iparm[6] = 0;  /* Not in use */
    iparm[7] = 0;  /* Max numbers of iterative refinement steps */
    iparm[8] = 0;  /* Not in use */
    iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0; /* Not in use */
    iparm[12] = 1; /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
    iparm[13] = 0; /* Output: Number of perturbed pivots */
    iparm[14] = 0; /* Not in use */
    iparm[15] = 0; /* Not in use */
    iparm[16] = 0; /* Not in use */
    iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1; /* Output: Mflops for LU factorization */
    iparm[19] = 0;  /* Output: Numbers of CG Iterations */
    iparm[23] = 0;  
    iparm[33] = threads;  /* Use parallel factorization */
    iparm[34] = 1;  /* Zero based indexing */
    iparm[59] = 0;

    maxfct = 1;    /* Maximum number of numerical factorizations. */
    mnum = 1;      /* Which factorization to use. */
    msglvl = 0;    /* Print statistical information in file */
    error = 0;     /* Initialize error flag */

    /* Initialize the internal solver memory pointer. This is only */
    /* necessary for the FIRST call of the PARDISO solver. */
    for (i = 0; i < 64; i++)
    {
        pt[i] = 0;
    }

    /* Load matrix here */
    int n, nnz;
    double *values, *b, *x;
    int *columns, *rowPointers;

    const char* matrixname = filename.c_str();

    readMtx(matrixname, &n, &nnz, &values, &columns, &rowPointers);;

    // int n = 5;
    // int ia[6] = { 1, 4, 7, 9, 12, 14 };
    // int ja[13] = { 1, 3, 5, 1, 2, 5, 2, 3, 4, 2, 4, 3, 5 };
    // double  a[13] = { 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0 };

    /* Reordering and Symbolic Factorization. This step also allocates */
    /* all memory that is necessary for the factorization. */
    phase = 11;
    begin[0] = std::chrono::high_resolution_clock::now();// Start timer
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
            &n, values, rowPointers, columns, &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);
    end[0] = std::chrono::high_resolution_clock::now();  // Stop timer
    total[0] += std::chrono::duration_cast<std::chrono::microseconds>(end[0] - begin[0]).count();

    if (error != 0)
    {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }

    // printf("\nSymbolic factorization completed in %f seconds ... ", duration1.count());

    /* Numerical factorization */
    for (int i = 0; i < repLoops; i++){
      phase = 22;
      begin[1] = std::chrono::high_resolution_clock::now();  // Start timer
      PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
              &n, values, rowPointers, columns, &idum, &nrhs,
              iparm, &msglvl, &ddum, &ddum, &error);
      end[1] = std::chrono::high_resolution_clock::now();  // Stop timer
      total[1] += std::chrono::duration_cast<std::chrono::microseconds>(end[1] - begin[1]).count();
      if (error != 0)
      {
          printf("\nERROR during numerical factorization: %d", error);
          exit(2);
      }
    }

    // printf("\nNumerical factorization completed in %f seconds ... ", duration2.count());

    /* Termination and release of memory */
    phase = -1; /* Release internal memory. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
            &n, values, rowPointers, columns, &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);

    std::cout << "Analyze time: " << total[0] << "\nFactorization time: " << total[1] / repLoops << std::endl;

    return 0;
}

