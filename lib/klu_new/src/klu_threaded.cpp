#include <iostream>
#include "klu.h"  // Make sure to include the new KLU header that matches the updated version
#include <chrono>
#include <numeric>
#include <thread>
#include <algorithm>
#include <random>
#include "MtxTools.hpp"  // Ensure this is compatible with the updated KLU version
#include "BS_thread_pool.hpp"  // Third-party thread pool library for parallelism
#include "ThreadAnalysis.hpp"  // Custom thread analysis for performance measurement

klu_common Common;
std::vector<klu_numeric*> Numerics;  // Update to use pointers for better memory management
klu_symbolic* Symbolic;

std::vector<double> vecb_nrhs;
std::vector<std::vector<double>> vecb_nrhs_threads;
int n, nrhs;

const bool THREAD_ANALYSIS = false;

// Function to initialize KLU_numeric structure
void setNumeric(klu_numeric* Numeric)
{
    // Initialization handled directly by KLU library in modern versions
}

// Function to randomize vector data
std::vector<double> randomize_vector(const std::vector<double>& vec, double factor)
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

void single_task_factor(MtxTools::MTX_CSX& csc, std::vector<double>& data, klu_numeric* Numeric)
{
    *Numeric = *klu_factor(csc.indptr.data(), csc.indices.data(), data.data(), Symbolic, &Common);
    if (!Numeric)
    {
        std::cerr << "KLU factorization failed!" << std::endl;
        std::exit(1);
    }
}

void single_task_solve(std::vector<double>& bb, klu_numeric* Numeric, Chrono& begin, Chrono& end)
{
    begin = std::chrono::high_resolution_clock::now();
    klu_solve(Symbolic, Numeric, n, nrhs, bb.data(), &Common);
    end = std::chrono::high_resolution_clock::now();
}

void dummy_task()
{
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <threads> <nrhs>" << std::endl;
        return 1;
    }

    std::string filename = "./mat2.mtx";
    std::string bmatrix = "./vecb2.mtx";
    int repLoops = 10;

    if (argc == 6)
    {
        filename = argv[3];
        bmatrix = argv[4];
        repLoops = std::stoi(argv[5]);
    }

    // Initialize thread-related structures
    int numThreads = std::stoi(argv[1]);
    Numerics.resize(numThreads);
    vecb_nrhs_threads.resize(numThreads);

    const auto coo = MtxTools::read_mtx_coo(filename);
    if (coo.status == -1)
    {
        std::cerr << "Error reading matrix file: " << filename << std::endl;
        return 1;
    }

    auto csc = MtxTools::coo_to_csc(coo);
    n = csc.ncols;
    nrhs = std::stoi(argv[2]);
    vecb_nrhs.resize(n * nrhs);

    // Randomize non-zero values for different thread inputs
    std::vector<std::vector<double>> random_data(numThreads);
    for (int i = 0; i < numThreads; i++)
    {
        random_data[i] = randomize_vector(csc.data, 0.1);
    }

    const auto vecb = MtxTools::read_mtx_vec(bmatrix);
    if (vecb.status == -1)
    {
        std::cerr << "Error reading vector file: " << bmatrix << std::endl;
        return 1;
    }

    for (int i = 0; i < nrhs; i++)
    {
        for (int j = 0; j < n; j++)
        {
            vecb_nrhs[i * n + j] = vecb.data[j];
        }
    }
    auto vecb_nrhs_rand = randomize_vector(vecb_nrhs, 0.1);

    // Initialize KLU with defaults and perform symbolic analysis
    klu_defaults(&Common);
    Symbolic = klu_analyze(n, csc.indptr.data(), csc.indices.data(), &Common);
    if (!Symbolic)
    {
        std::cerr << "KLU symbolic analysis failed!" << std::endl;
        return 1;
    }

    double total_factor = 0.0;
    double total_solve = 0.0;

    std::vector<ThreadAnalysis::ThreadData> threadData = {};
    BS::thread_pool pool(numThreads);

    // Initializing and warming up the thread pool
    for (int i = 0; i < numThreads; i++)
    {
        pool.push_task(dummy_task);
    }
    pool.wait_for_tasks();

    for (int rep = 0; rep < repLoops; rep++)
    {
        // Reset randomized RHS vectors
        for (auto& bb : vecb_nrhs_threads)
        {
            bb = vecb_nrhs_rand;
        }

        Chrono start, stop;

        // Factorization phase with parallel loop
        start = std::chrono::high_resolution_clock::now();
        pool.parallelize_loop(0, numThreads, [&csc, &random_data](int a, int b)
        {
            for (int i = a; i < b; i++)
            {
                single_task_factor(std::ref(csc), std::ref(random_data[i]), Numerics[i]);
            }
        }).wait();
        stop = std::chrono::high_resolution_clock::now();
        total_factor += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

        // Solve phase with parallel loop
        std::vector<Chrono> begin(numThreads), end(numThreads);
        start = std::chrono::high_resolution_clock::now();
        pool.parallelize_loop(0, numThreads, [&begin, &end](int a, int b)
        {
            for (int i = a; i < b; i++)
            {
                single_task_solve(vecb_nrhs_threads[i], Numerics[i], begin[i], end[i]);
            }
        }).wait();
        stop = std::chrono::high_resolution_clock::now();
        total_solve += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

        // Optional thread analysis
        if (THREAD_ANALYSIS)
        {
            std::vector<double> bg(begin.size()), ed(end.size());
            for (int i = 0; i < numThreads; i++)
            {
                bg[i] = std::chrono::duration_cast<std::chrono::microseconds>(begin[i] - begin[0]).count();
                ed[i] = std::chrono::duration_cast<std::chrono::microseconds>(end[i] - begin[0]).count();
                threadData.emplace_back(rep, i, bg[i], ed[i]);
            }
        }
    }

    // Output timing results
    std::cout << "Average factorization time: " << total_factor / repLoops << " µs" << std::endl;
    std::cout << "Average solve time: " << total_solve / repLoops << " µs" << std::endl;

    // Cleanup KLU structures
    klu_free_symbolic(&Symbolic, &Common);
    for (auto& numeric : Numerics)
    {
        klu_free_numeric(&numeric, &Common);
    }

    // Write thread analysis data if enabled
    if (THREAD_ANALYSIS)
    {
        ThreadAnalysis::writeThreadData(threadData, "./thread_data.csv");
    }

    return 0;
}
