/* ========================================================================== */
/* === KLU_kernel =========================================================== */
/* ========================================================================== */

#include <iostream>
#include "klu_factor.h"
#include "klu_solve.h"
#include <chrono>
#include <numeric>
#include <thread>
#include <algorithm>
#include <random>
#include "MtxTools.hpp"
#include "BS_thread_pool.hpp"
#include "ThreadAnalysis.hpp"

klu_common Common;
std::vector<KLU_numeric> Numerics;
klu_symbolic Symbolic;

std::vector<double> vecb_nrhs;
std::vector<std::vector<double>> vecb_nrhs_threads;
int n, nrhs;

const bool THREAD_ANALYSIS = false;

void setNumeric(KLU_numeric &Numeric)
{
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
}

std::vector<double> randomize_vector(const std::vector<double> vec, double factor)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d{1, factor};

    std::vector<double> result(vec.size());

    for (int i = 0; i < vec.size(); i++)
    {
        result[i] = vec[i] * d(gen);
    }
    return result;
}

using Chrono = std::chrono::time_point<std::chrono::high_resolution_clock>;

void single_task(MtxTools::MTX_CSX &csc, std::vector<double> &bb, KLU_numeric &Numeric, KLU_symbolic &Symbolic, KLU_common &Common, Chrono &begin, Chrono &end)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;

    klu_factor(csc.indptr.data(), csc.indices.data(), csc.data.data(), &Symbolic, &Numeric, &Common);

    begin = std::chrono::high_resolution_clock::now();
    klu_solve(&Symbolic, &Numeric, n, nrhs, bb.data(), &Common);
    end = std::chrono::high_resolution_clock::now();
}

void single_task_factor(MtxTools::MTX_CSX &csc, std::vector<double> &data, KLU_numeric &Numeric, KLU_symbolic &Symbolic, KLU_common &Common)
{
    klu_factor(csc.indptr.data(), csc.indices.data(), data.data(), &Symbolic, &Numeric, &Common);
}

void single_task_solve(std::vector<double> &bb, KLU_numeric &Numeric, KLU_symbolic &Symbolic, KLU_common &Common, Chrono &begin, Chrono &end)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;

    begin = std::chrono::high_resolution_clock::now();
    klu_solve(&Symbolic, &Numeric, n, nrhs, bb.data(), &Common);
    end = std::chrono::high_resolution_clock::now();
}

void dummy_task()
{
    // std::cout << "dummy task" << std::endl;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <thread> <nrhs>" << std::endl;
        return 1;
    }
    // std::string homeDir = getenv("HOME");

    std::string filename = "";
    std::string bmatrix = "";

    int repLoops = 10;

    if (argc == 3)
    {
        filename = "./mat2.mtx";
        bmatrix = "./vecb2.mtx";
        // filename = "./circuit_2/circuit_2.mtx";
        // bmatrix = "./circuit_2/vecb.mtx";
        // filename = "./scircuit/scircuit.mtx";
        // bmatrix = "./scircuit/vecb.mtx";
    }

    if (argc == 6)
    {
        filename = argv[3];
        bmatrix = argv[4];
        repLoops = atoi(argv[5]);
    }

    Numerics.resize(std::stoi(argv[1]));
    vecb_nrhs_threads.resize(std::stoi(argv[1]));

    const auto coo = MtxTools::read_mtx_coo(filename);
    if (coo.status == -1)
    {
        std::cerr << "Error reading matrix file: " << filename << std::endl;
        return 1;
    }

    auto csc = MtxTools::coo_to_csc(coo);
    n = csc.ncols;

    // randomize non-zero values
    std::vector<std::vector<double>> random_data(Numerics.size());
    for (int i = 0; i < Numerics.size(); i++)
    {
        random_data[i] = randomize_vector(csc.data, 0.1);
    }

    // read_bmatrix(bmatrix, b1, &nrhs);
    const auto vecb = MtxTools::read_mtx_vec(bmatrix);
    if (vecb.status == -1)
    {
        std::cerr << "Error reading matrix file: " << bmatrix << std::endl;
        return 1;
    }

    // std::vector<double> b1(nrhs * vecb.nrows);

    nrhs = atoi(argv[2]);
    vecb_nrhs.resize(n * nrhs);

    for (int i = 0; i < nrhs; i++)
    {
        for (int j = 0; j < n; j++)
        {
            vecb_nrhs[i * n + j] = vecb.data[j];
        }
    }
    auto vecb_nrhs_rand = randomize_vector(vecb_nrhs, 0.1);

    klu_defaults(&Common);

    Symbolic = *klu_analyze(n, csc.indptr.data(), csc.indices.data(), &Common);

    printf("nblocks=%d,nzoff=%d,maxblock=%d,nnz=%d\n", Symbolic.nblocks, Symbolic.nzoff, Symbolic.maxblock, Symbolic.nz);

    int nzoff1 = Symbolic.nzoff + 1, n1 = n + 1;
    int lusize = Common.memgrow * (Symbolic.lnz + Symbolic.unz) + 4 * n + 1;

    for (auto &numeric : Numerics)
    {
        setNumeric(numeric);
    }

    double total_factor = 0.0;
    double total_solve = 0.0;

    std::vector<ThreadAnlysis::ThreadData> threadData = {};

    // Constructs a thread pool with only 12 threads.
    BS::thread_pool pool(Numerics.size());

    for (int i = 0; i < Numerics.size(); i++)
    {
        pool.push_task(dummy_task);
    }
    pool.wait_for_tasks();

    for (int rep = 0; rep < repLoops; rep++)
    {

        for (auto &bb : vecb_nrhs_threads)
        {
            bb = vecb_nrhs_rand;
        }

        Chrono start, stop;

        // factorization

        start = std::chrono::high_resolution_clock::now();

        auto loopFactor = [&csc, &random_data](const int a, const int b)
        {for (int i = a; i < b; i++)
            {
                single_task_factor(std::ref(csc), std::ref(random_data[i]), std::ref(Numerics[i]), std::ref(Symbolic), std::ref(Common));
            } };

        pool.parallelize_loop(0, Numerics.size(), loopFactor).wait();

        stop = std::chrono::high_resolution_clock::now();
        total_factor += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

        // Solve

        std::vector<Chrono> begin(Numerics.size()), end(Numerics.size());

        start = std::chrono::high_resolution_clock::now();

        auto loopSolve = [&begin, &end](const int a, const int b)
        {for (int i = a; i < b; i++)
            {
                single_task_solve(std::ref(vecb_nrhs_threads[i]), std::ref(Numerics[i]), std::ref(Symbolic), std::ref(Common), std::ref(begin[i]), std::ref(end[i]));
            } };

        pool.parallelize_loop(0, Numerics.size(), loopSolve).wait();

        stop = std::chrono::high_resolution_clock::now();
        total_solve += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

        if (THREAD_ANALYSIS)
        {
            std::vector<double> bg(begin.size()), ed(end.size());
            for (int i = 0; i < Numerics.size(); i++)
            {
                bg[i] = std::chrono::duration_cast<std::chrono::microseconds>(begin[i] - begin[0]).count();
                ed[i] = std::chrono::duration_cast<std::chrono::microseconds>(end[i] - begin[0]).count();
                // std::cout << "Thread " << i << ": " << bg[i] << ", " << ed[i] << std::endl;
                const auto data = ThreadAnlysis::ThreadData{rep, i, bg[i], ed[i]};
                threadData.push_back(data);
            }

            double min_begin = *std::min_element(bg.begin(), bg.end());
            double max_end = *std::min_element(ed.begin(), ed.end());
        }

        // std::cout << "min_begin = " << min_begin << std::endl;
        // std::cout << "max_end = " << max_end << std::endl;
        // std::cout << "max duration [" << rep << "]: " << max_end - min_begin << std::endl;

        // printf("lusize=%d\n", Numerics[0].lusize_sum);
        /*for (int i = 0; i < 10; i++)
        {
            std::cout << "x[" << i << "] = ";
            for (int j = 0; j < nrhs; j++)
            {
                std::cout << vecb_nrhs_threads[0][i + j * n] << ", ";
            }
            std::cout << std::endl;
        }*/
    }

    if (THREAD_ANALYSIS)
    {
        ThreadAnlysis::writeThreadData(threadData, "./thread_data.csv");
    }

    std::cout << "total factor: " << total_factor / repLoops << std::endl;
    std::cout << "total solve: " << total_solve / repLoops << std::endl;

    return 0;
}