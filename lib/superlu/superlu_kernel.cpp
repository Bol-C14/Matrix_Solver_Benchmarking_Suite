#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>

#include "slu_ddefs.h"
#include "mmio.h"

// Function to read a Matrix Market file and convert it to CSC format
void readMtxCSC(const char *filename, int *n, int *nnz, double **values, int **row_indices, int **col_pointers) {
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

    std::vector<int> rows(nz), cols(nz);
    std::vector<double> vals(nz);

    // Read the data
    for (int i = 0; i < nz; i++) {
        int row, col;
        double val;
        if (fscanf(f, "%d %d %lg\n", &row, &col, &val) != 3) {
            printf("Error reading matrix data at line %d.\n", i+1);
            exit(1);
        }
        row--; // Convert to 0-based indexing
        col--;
        rows[i] = row;
        cols[i] = col;
        vals[i] = val;
    }

    fclose(f);

    // Convert to CSC format
    *values = (double *)malloc(nz * sizeof(double));
    *row_indices = (int *)malloc(nz * sizeof(int));
    *col_pointers = (int *)malloc((N+1) * sizeof(int));

    // Initialize col_pointers to zero
    memset(*col_pointers, 0, (N+1) * sizeof(int));

    // Count the number of entries in each column
    for (int i = 0; i < nz; i++) {
        (*col_pointers)[cols[i] + 1]++;
    }

    // Cumulative sum to get the starting position of each column
    for (int i = 0; i < N; i++) {
        (*col_pointers)[i+1] += (*col_pointers)[i];
    }

    // Temporary copy of col_pointers to keep track of current insertion points
    std::vector<int> current_col_pos(N, 0);
    for (int i = 0; i < N; i++) {
        current_col_pos[i] = (*col_pointers)[i];
    }

    // Populate row_indices and values
    for (int i = 0; i < nz; i++) {
        int col = cols[i];
        int dest_pos = current_col_pos[col];

        (*row_indices)[dest_pos] = rows[i];
        (*values)[dest_pos] = vals[i];

        current_col_pos[col]++;
    }
}

int main(int argc, char **argv){

    std::string filename = "";
    int repLoops = 1000;
    int threads = 1;

    // Parse command-line arguments
    if (argc >= 2)
    {
        filename = argv[1];
    } else {
        printf("Usage: %s [matrix-market-filename] [repLoops] [threads]\n", argv[0]);
        return 0;
    }

    if (argc >= 3)
    {
        repLoops = atoi(argv[2]);
    }

    if (argc >= 4)
    {
        threads = atoi(argv[3]);
    }

    // Note: SuperLU's threading model depends on how it's built.
    // Ensure SuperLU is compiled with multi-threading support if threads > 1.

    int n, nnz;
    double *values;
    int *row_indices, *col_pointers;

    const char* matrixname = filename.c_str();

    // Read the matrix from the Matrix Market file
    readMtxCSC(matrixname, &n, &nnz, &values, &row_indices, &col_pointers);

    // Create SuperMatrix A in compressed column format
    SuperMatrix A;
    dCreate_CompCol_Matrix(&A, n, n, nnz, values, row_indices, col_pointers, SLU_NC, SLU_D, SLU_GE);

    // Create right-hand side matrix B (all ones)
    int nrhs = 1;
    double *rhs = doubleMalloc(n * nrhs);
    for (int i = 0; i < n; i++) {
        rhs[i] = 1.0;
    }
    SuperMatrix B;
    dCreate_Dense_Matrix(&B, n, nrhs, rhs, n, SLU_DN, SLU_D, SLU_GE);

    // Initialize SuperLU options
    superlu_options_t options;
    set_default_options(&options);
    options.ColPerm = COLAMD;       // Column permutation strategy
    options.Fact = DOFACT;           // Compute factorizations
    options.IterRefine = NOREFINE;   // No iterative refinement
    options.PrintStat = NO;          // Suppress SuperLU statistics output

    // Initialize statistics
    SuperLUStat_t stat;
    StatInit(&stat);

    // Allocate permutation vectors
    int *perm_c = intMalloc(n); // Column permutation vector
    int *perm_r = intMalloc(n); // Row permutation vector
    int *etree = intMalloc(n);  // Elimination tree

    // Allocate scaling vectors and other necessary variables
    char equed[1];
    double *R = (double *) SUPERLU_MALLOC(n * sizeof(double));
    double *C = (double *) SUPERLU_MALLOC(n * sizeof(double));

    // Allocate arrays for error estimates
    double *ferr = doubleMalloc(nrhs);
    double *berr = doubleMalloc(nrhs);

    // Allocate solution vector X
    double *xact = doubleMalloc(n * nrhs);
    SuperMatrix X;
    dCreate_Dense_Matrix(&X, n, nrhs, xact, n, SLU_DN, SLU_D, SLU_GE);

    // Workspace parameters (not used here)
    void *work = NULL;
    int lwork = 0;

    // Variables for factorization results
    SuperMatrix L, U;
    GlobalLU_t Glu;
    mem_usage_t mem_usage;
    int info;

    // Time measurement variables
    using Chrono = std::chrono::time_point<std::chrono::high_resolution_clock>;
    Chrono begin_analyze, end_analyze, begin_factor, end_factor;
    double analyze_time = 0.0;
    double total_factor_time = 0.0;

    // ===========================
    // First Factorization (Analyze + Factorize)
    // ===========================
    begin_analyze = std::chrono::high_resolution_clock::now();

    dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           &L, &U, work, lwork, &B, &X, NULL, NULL, ferr, berr,
           &Glu, &mem_usage, &stat, &info);

    end_analyze = std::chrono::high_resolution_clock::now();
    analyze_time = std::chrono::duration_cast<std::chrono::microseconds>(end_analyze - begin_analyze).count();

    if (info != 0) {
        printf("dgssvx() error returns INFO= %d\n", info);
        exit(1);
    }

    // ===========================
    // Numerical Factorization Loop
    // ===========================
    // Set options to reuse the same symbolic factorization
    options.Fact = SamePattern_SameRowPerm;

    for (int i = 0; i < repLoops; i++) {

        // Destroy L and U from previous factorization
        Destroy_SuperNode_Matrix(&L);
        Destroy_CompCol_Matrix(&U);

        // Re-initialize statistics for each iteration
        StatInit(&stat);

        begin_factor = std::chrono::high_resolution_clock::now();

        dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
               &L, &U, work, lwork, &B, &X, NULL, NULL, ferr, berr,
               &Glu, &mem_usage, &stat, &info);

        end_factor = std::chrono::high_resolution_clock::now();
        total_factor_time += std::chrono::duration_cast<std::chrono::microseconds>(end_factor - begin_factor).count();

        if (info != 0) {
            printf("dgssvx() error returns INFO= %d\n", info);
            exit(1);
        }

        // Free statistics after each factorization
        StatFree(&stat);
    }

    // Compute average factorization time
    double average_factor_time = total_factor_time / repLoops;

    // Output the benchmarking results
    std::cout << "Analyze time: " << analyze_time << " microseconds" << std::endl;
    std::cout << "Average factorization time: " << average_factor_time << " microseconds" << std::endl;

    // ===========================
    // Clean Up
    // ===========================
    SUPERLU_FREE(rhs);
    SUPERLU_FREE(xact);
    SUPERLU_FREE(perm_r);
    SUPERLU_FREE(perm_c);
    SUPERLU_FREE(etree);
    SUPERLU_FREE(R);
    SUPERLU_FREE(C);
    SUPERLU_FREE(ferr);
    SUPERLU_FREE(berr);

    Destroy_SuperMatrix_Store(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);

    StatFree(&stat);

    return 0;
}
