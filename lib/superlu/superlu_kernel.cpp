#include <iostream>
#include <vector>
#include "slu_ddefs.h"
#include <chrono>
#include "MtxTools.hpp"  // Replace with your actual MTX file reading utility

int n, nrhs;
std::vector<double> vecb_nrhs;

void resetVectorB(const std::vector<double>& b) {
    for (int i = 0; i < nrhs; i++) {
        for (int j = 0; j < n; j++) {
            vecb_nrhs[i * n + j] = b[j];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <nrhs>" << std::endl;
        return 1;
    }

    std::string filename = "";
    std::string bmatrix = "";
    int repLoops = 10;

    if (argc == 2) {
        filename = "./mat2.mtx";
        bmatrix = "./vecb2.mtx";
    }

    if (argc == 5) {
        filename = argv[2];
        bmatrix = argv[3];
        repLoops = atoi(argv[4]);
    }

    const auto coo = MtxTools::read_mtx_coo(filename);
    if (coo.status == -1) {
        std::cerr << "Error reading matrix file: " << filename << std::endl;
        return 1;
    }

    auto csc = MtxTools::coo_to_csc(coo);
    n = csc.ncols;
    nrhs = atoi(argv[1]);
    vecb_nrhs.resize(n * nrhs);

    // Assuming that you have a function to read the right-hand side vector B
    // const auto vecb = MtxTools::read_mtx_vec(bmatrix);
    // if (vecb.status == -1) {
    //     std::cerr << "Error reading vector file: " << bmatrix << std::endl;
    //     return 1;
    // }

    // resetVectorB(vecb.data);

    // SuperLU structures
    SuperMatrix A, L, U, B;
    superlu_options_t options;
    SuperLUStat_t stat;
    int *perm_r = intMalloc(n);  // row permutations
    int *perm_c = intMalloc(n);  // column permutations
    int info;

    // Initialize the SuperLU options
    set_default_options(&options);
    options.ColPerm = NATURAL;
    StatInit(&stat);

    // Create the SuperLU matrix in column-compressed form
    dCreate_CompCol_Matrix(&A, n, n, csc.data.size(), csc.data.data(),
                           csc.indices.data(), csc.indptr.data(), SLU_NC, SLU_D, SLU_GE);

    // Create the right-hand side matrix B
    dCreate_Dense_Matrix(&B, n, nrhs, vecb_nrhs.data(), n, SLU_DN, SLU_D, SLU_GE);

    // Timing the factorization and solve process
    using Chrono = std::chrono::high_resolution_clock;
    double total[2] = {0.0};

    for (int i = 0; i < repLoops; i++) {
        auto begin = Chrono::now();

        // Factor and solve
        dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);

        auto end = Chrono::now();
        total[0] += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        if (info != 0) {
            std::cerr << "SuperLU dgssv failed with info = " << info << std::endl;
            return 1;
        }
    }

    std::cout << "Solution for nrhs = " << nrhs << ":" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "x[" << i << "] = " << vecb_nrhs[i] << std::endl;
    }

    std::cout << "Total time for factorization and solve (average over " << repLoops
              << " repetitions): " << total[0] / repLoops << " microseconds" << std::endl;

    if (perm_r) SUPERLU_FREE(perm_r);
    if (perm_c) SUPERLU_FREE(perm_c);
    if (A.Store) Destroy_CompCol_Matrix(&A);
    if (B.Store) Destroy_SuperMatrix_Store(&B);
    if (L.Store) Destroy_SuperNode_Matrix(&L);
    if (U.Store) Destroy_CompCol_Matrix(&U);

    return 0;
}
