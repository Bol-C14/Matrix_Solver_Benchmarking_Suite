/* ========================================================================== */
/* === KLU_kernel =========================================================== */
/* ========================================================================== */

#include <iostream>
#include "klu_factor.h"
#include "klu_solve.h"
#include <chrono>
#include <numeric>
#include "MtxTools.hpp"

std::vector<double> vecb_nrhs;

int n, nrhs;

void resetVectorB(const std::vector<double> &b)
{
    for (int i = 0; i < nrhs; i++)
    {
        for (int j = 0; j < n; j++)
        {
            vecb_nrhs[i * n + j] = b[j];
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <nrhs>" << std::endl;
        return 1;
    }
    // std::string homeDir = getenv("HOME");

    std::string filename = "";
    std::string bmatrix = "";

    int repLoops = 10;

    if (argc == 2)
    {
        filename = "./mat2.mtx";
        bmatrix = "./vecb2.mtx";
        // filename = "./circuit_2/circuit_2.mtx";
        // bmatrix = "./circuit_2/vecb.mtx";
        // filename = "./scircuit/scircuit.mtx";
        // bmatrix = "./scircuit/vecb.mtx";
    }

    if (argc == 5)
    {
        filename = argv[2];
        bmatrix = argv[3];
        repLoops = atoi(argv[4]);
    }

    const auto coo = MtxTools::read_mtx_coo(filename);
    if (coo.status == -1)
    {
        std::cerr << "Error reading matrix file: " << filename << std::endl;
        return 1;
    }

    auto csc = MtxTools::coo_to_csc(coo);
    n = csc.ncols;

    // read_bmatrix(bmatrix, b1, &nrhs);
    // const auto vecb = MtxTools::read_mtx_vec(bmatrix);
    // if (vecb.status == -1)
    // {
    //     std::cerr << "Error reading matrix file: " << bmatrix << std::endl;
    //     return 1;
    // }

    // std::vector<double> b1(nrhs * vecb.nrows);

    nrhs = atoi(argv[1]);
    vecb_nrhs.resize(n * nrhs);

    // resetVectorB(vecb.data);

    klu_common Common;
    KLU_numeric Numeric;
    klu_symbolic Symbolic;
    klu_defaults(&Common);

    using Chrono = std::chrono::time_point<std::chrono::high_resolution_clock>;
    Chrono begin[3], end[3];
    double total[3] = {0};

    for (size_t i = 0; i < repLoops; i++)
    {
        begin[0] = std::chrono::high_resolution_clock::now();
        Symbolic = *klu_analyze(n, csc.indptr.data(), csc.indices.data(), &Common);
        end[0] = std::chrono::high_resolution_clock::now();
        total[0] += std::chrono::duration_cast<std::chrono::microseconds>(end[0] - begin[0]).count();
    }



    // for (int i = 0; i < n; i++)
    //     printf("P[%d]=%d,Q[%d]=%d,R[%d]=%d,Lnz[%d]=%lf\n", i, Symbolic.P[i], i, Symbolic.Q[i], i, Symbolic.R[i], i, Symbolic.Lnz[i]);

    // printf("nblocks=%d,nzoff=%d,maxblock=%d,nnz=%d\n", Symbolic.nblocks, Symbolic.nzoff, Symbolic.maxblock, Symbolic.nz);

    int nzoff1 = Symbolic.nzoff + 1, n1 = n + 1;
    int lusize = Common.memgrow * (Symbolic.lnz + Symbolic.unz) + 4 * n + 1;

    Numeric.n = Symbolic.n;
    Numeric.nblocks = Symbolic.nblocks;
    Numeric.nzoff = Symbolic.nzoff;
    Numeric.Pnum = (int *)malloc(n * sizeof(int));
    Numeric.Offp = (int *)malloc(n1 * sizeof(int));
    Numeric.Offi = (int *)malloc(nzoff1 * sizeof(int));
    Numeric.Offx = (double *)malloc(nzoff1 * sizeof(double));
    Numeric.Lip = (int *)calloc(n, sizeof(int));
    Numeric.Uip = (int *)malloc(n * sizeof(int));
    Numeric.Llen = (int *)malloc(n * sizeof(int));
    Numeric.Ulen = (int *)malloc(n * sizeof(int));
    Numeric.LUsize = (int *)calloc(Symbolic.nblocks, sizeof(int));
    Numeric.LUbx = (double *)calloc(lusize * 2, sizeof(double));
    Numeric.Udiag = (double *)malloc(n * sizeof(double));
    Numeric.Rs = (double *)malloc(n * sizeof(double));
    Numeric.Pinv = (int *)malloc(n * sizeof(int));
    Numeric.worksize = n * sizeof(double) + MAX(n * 3 * sizeof(double), Symbolic.maxblock * 6 * sizeof(int));
    Numeric.Xwork = (double *)calloc(n * nrhs, sizeof(double));





    for (int i = 0; i < repLoops; i++)
    {
        // std::iota(b.begin(), b.end(), 0);
        // read_bmatrix(bmatrix, b, &nrhs);
        // b = b2;
        // resetVectorB(vecb.data);

        begin[1] = std::chrono::high_resolution_clock::now();
        klu_factor(csc.indptr.data(), csc.indices.data(), csc.data.data(), &Symbolic, &Numeric, &Common);
        end[1] = std::chrono::high_resolution_clock::now();
        total[1] += std::chrono::duration_cast<std::chrono::microseconds>(end[1] - begin[1]).count();

        // begin[2] = std::chrono::high_resolution_clock::now();
        // klu_solve(&Symbolic, &Numeric, n, nrhs, vecb_nrhs.data(), &Common);
        // end[2] = std::chrono::high_resolution_clock::now();
        // total[2] += std::chrono::duration_cast<std::chrono::microseconds>(end[2] - begin[2]).count();
    }

    std::cout << nrhs << std::endl;

    // for (int i = 0; i < 10; i++)
    // {
    //     for (int j = 0; j < nrhs - 1; j++)
    //         printf("x[%d,%d] = %g\t", i, j, vecb_nrhs[i + n * j]);
    //     printf("x[%d,%d] = %g\n", i, nrhs - 1, vecb_nrhs[i + n * (nrhs - 1)]);
    // }

    std::cout << "Analyze time: " << total[0] / repLoops << "\nFactorization time: " << total[1] / repLoops << std::endl;
    // printf("lusize=%d\n", Numeric.lusize_sum);


    return 0;
}

//
///* ========================================================================== */
///* === KLU_kernel =========================================================== */
///* ========================================================================== */
//
//#include <iostream>
//#include <cassert> // For using assertions
//#include <chrono>
//#include <numeric>
//#include <vector>
//#include <cstdlib> // For malloc, calloc, free
//#include "klu_factor.h"
//#include "klu_solve.h"
//#include "MtxTools.hpp"
//
//std::vector<double> vecb_nrhs;
//
//int n, nrhs;
//
//void resetVectorB(const std::vector<double> &b)
//{
//    std::cout << "[DEBUG] Resetting VectorB..." << std::endl;
//    for (int i = 0; i < nrhs; i++)
//    {
//        for (int j = 0; j < n; j++)
//        {
//            vecb_nrhs[i * n + j] = b[j];
//        }
//    }
//}
//
//int main(int argc, char **argv)
//{
//    if (argc < 2)
//    {
//        std::cout << "Usage: " << argv[0] << " <nrhs>" << std::endl;
//        return 1;
//    }
//
//    std::string filename = "";
//    std::string bmatrix = "";
//    int repLoops = 10;
//
//    if (argc == 2)
//    {
//        filename = "./mat2.mtx";
//        bmatrix = "./vecb2.mtx";
//    }
//
//    if (argc == 5)
//    {
//        filename = argv[2];
//        bmatrix = argv[3];
//        repLoops = atoi(argv[4]);
//    }
//
//    // Debug: print matrix file names and loop count
//    std::cout << "[DEBUG] Matrix file: " << filename << std::endl;
//    std::cout << "[DEBUG] Vector file: " << bmatrix << std::endl;
//    std::cout << "[DEBUG] Repetition loops: " << repLoops << std::endl;
//
//    const auto coo = MtxTools::read_mtx_coo(filename);
//    if (coo.status == -1)
//    {
//        std::cerr << "[ERROR] Error reading matrix file: " << filename << std::endl;
//        return 1;
//    }
//
//    auto csc = MtxTools::coo_to_csc(coo);
//    n = csc.ncols;
//    std::cout << "[DEBUG] Matrix dimensions: " << n << "x" << n << std::endl;
//
//    nrhs = atoi(argv[1]);
//    vecb_nrhs.resize(n * nrhs);
//
//    klu_common Common;
//    KLU_numeric Numeric;
//    klu_symbolic Symbolic;
//    klu_defaults(&Common);
//
//    // Check default settings
//    std::cout << "[DEBUG] klu_defaults returned, Common is configured" << std::endl;
//
//    using Chrono = std::chrono::time_point<std::chrono::high_resolution_clock>;
//    Chrono begin[3], end[3];
//    double total[3] = {0};
//
//    for (size_t i = 0; i < repLoops; i++)
//    {
//        begin[0] = std::chrono::high_resolution_clock::now();
//        Symbolic = *klu_analyze(n, csc.indptr.data(), csc.indices.data(), &Common);
//        end[0] = std::chrono::high_resolution_clock::now();
//        total[0] += std::chrono::duration_cast<std::chrono::microseconds>(end[0] - begin[0]).count();
//
//        std::cout << "[DEBUG] Completed klu_analyze for loop " << i + 1 << std::endl;
//    }
//
//    // Debug info for Symbolic result
//    std::cout << "[DEBUG] Symbolic analysis completed, nzoff=" << Symbolic.nzoff << ", nblocks=" << Symbolic.nblocks << std::endl;
//
//    int nzoff1 = Symbolic.nzoff + 1, n1 = n + 1;
//    int lusize = Common.memgrow * (Symbolic.lnz + Symbolic.unz) + 4 * n + 1;
//    std::cout << "[DEBUG] lusize=" << lusize << std::endl;
//
//    // Allocate Numeric memory
//    Numeric.n = Symbolic.n;
//    Numeric.nblocks = Symbolic.nblocks;
//    Numeric.nzoff = Symbolic.nzoff;
//    Numeric.Pnum = (int *)malloc(n * sizeof(int));
//    Numeric.Offp = (int *)malloc(n1 * sizeof(int));
//    Numeric.Offi = (int *)malloc(nzoff1 * sizeof(int));
//    Numeric.Offx = (double *)malloc(nzoff1 * sizeof(double));
//    Numeric.Lip = (int *)calloc(n, sizeof(int));
//    Numeric.Uip = (int *)malloc(n * sizeof(int));
//    Numeric.Llen = (int *)malloc(n * sizeof(int));
//    Numeric.Ulen = (int *)malloc(n * sizeof(int));
//    Numeric.LUsize = (int *)calloc(Symbolic.nblocks, sizeof(int));
//    Numeric.LUbx = (double *)calloc(lusize * 2, sizeof(double));
//    Numeric.Udiag = (double *)malloc(n * sizeof(double));
//    Numeric.Rs = (double *)malloc(n * sizeof(double));
//    Numeric.Pinv = (int *)malloc(n * sizeof(int));
//    Numeric.Xwork = (double *)calloc(n * nrhs, sizeof(double));
//
//    // Check if malloc/calloc failed
//    assert(Numeric.Pnum && "Allocation failed for Pnum");
//    assert(Numeric.Offp && "Allocation failed for Offp");
//    assert(Numeric.Offi && "Allocation failed for Offi");
//    assert(Numeric.Offx && "Allocation failed for Offx");
//    assert(Numeric.Lip && "Allocation failed for Lip");
//    assert(Numeric.Uip && "Allocation failed for Uip");
//    assert(Numeric.Llen && "Allocation failed for Llen");
//    assert(Numeric.Ulen && "Allocation failed for Ulen");
//    assert(Numeric.LUsize && "Allocation failed for LUsize");
//    assert(Numeric.LUbx && "Allocation failed for LUbx");
//    assert(Numeric.Udiag && "Allocation failed for Udiag");
//    assert(Numeric.Rs && "Allocation failed for Rs");
//    assert(Numeric.Pinv && "Allocation failed for Pinv");
//    assert(Numeric.Xwork && "Allocation failed for Xwork");
//
//    std::cout << "[DEBUG] Memory allocation completed for Numeric" << std::endl;
//
//    for (int i = 0; i < repLoops; i++)
//    {
//        begin[1] = std::chrono::high_resolution_clock::now();
//        klu_factor(csc.indptr.data(), csc.indices.data(), csc.data.data(), &Symbolic, &Numeric, &Common);
//        end[1] = std::chrono::high_resolution_clock::now();
//        total[1] += std::chrono::duration_cast<std::chrono::microseconds>(end[1] - begin[1]).count();
//
//        std::cout << "[DEBUG] klu_factor loop " << i + 1 << " completed." << std::endl;
//    }
//
//    // Print out some of the results
//    std::cout << "[DEBUG] nrhs: " << nrhs << std::endl;
//    std::cout << "Analyze time: " << total[0] / repLoops << "\nFactorization time: " << total[1] / repLoops << std::endl;
//
//    // Free allocated memory
//    free(Numeric.Pnum);
//    free(Numeric.Offp);
//    free(Numeric.Offi);
//    free(Numeric.Offx);
//    free(Numeric.Lip);
//    free(Numeric.Uip);
//    free(Numeric.Llen);
//    free(Numeric.Ulen);
//    free(Numeric.LUsize);
//    free(Numeric.LUbx);
//    free(Numeric.Udiag);
//    free(Numeric.Rs);
//    free(Numeric.Pinv);
//    free(Numeric.Xwork);
//
//    std::cout << "[DEBUG] Memory deallocation completed" << std::endl;
//
//    return 0;
//}
