#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu_cpp.inl"
#include <iostream>
#include <chrono>

const char *const ORDERING_METHODS[] = {"", "", "", "", "AMD", "AMM", "AMO1", "AMO2", "AMO3", "AMDF"};

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
    int threads = 1;

    // For timing
    using Chrono = std::chrono::time_point<std::chrono::high_resolution_clock>;
    Chrono begin[3], end[3];
    double total[3] = {0};
    

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

    int ret;
    _double_t *ax = NULL, *b = NULL, *x = NULL;
    _uint_t *ai = NULL, *ap = NULL;
    _uint_t n, row, col, nz, nnz, i, j;
    CNicsLU solver;
    _double_t res[4], cond, det1, det2, fflop, sflop;
    size_t mem;

    const char* matrixname = filename.c_str();
    // std::cout<< "Matrix name: " << matrixname << std::endl;

    // read matrix A
    if (__FAIL(ReadMatrixMarketFile(matrixname, &row, &col, &nz, NULL, NULL, NULL, NULL, NULL, NULL)))
    {
        printf("Failed to read matrix A\n");
        goto EXIT;
    }
    n = row;
    nnz = nz;
    ax = new _double_t[nnz];
    ai = new _uint_t[nnz];
    ap = new _uint_t[n + 1];
    ReadMatrixMarketFile(matrixname, &row, &col, &nz, ax, ai, ap, NULL, NULL, NULL);
    printf("Matrix : row %d, col %d, nnz %d\n", n, n, nnz);

    // //read RHS B
    // b = new _double_t[n];
    // ReadMatrixMarketFile(bmatrix, &row, &col, &nz, b, NULL, NULL, NULL, NULL, NULL);

    x = new _double_t[n];
    memset(x, 0, sizeof(_double_t) * n);

    // INicsLU inst = NULL ;
    // _double_t * cfg = NULL ;
    // const _double_t * stat = NULL ;
    // const char * last_err = NULL ;

    // initialize solver
    ret = solver.Initialize();
    if (__FAIL(ret))
    {
        printf("Failed to initialize, return = %d\n", ret);
        goto EXIT;
    }
    // printf("NICSLU version %.0lf\n", solver.GetInformation(31));
    solver.SetConfiguration(0, 1.); // enable timer
    solver.SetConfiguration(3, 4); // enable timer

    // pre-ordering (do only once)
    begin[0] = std::chrono::high_resolution_clock::now();
    solver.Analyze(n, ax, ai, ap, MATRIX_ROW_REAL);
    end[0] = std::chrono::high_resolution_clock::now();
    total[0] += std::chrono::duration_cast<std::chrono::microseconds>(end[0] - begin[0]).count();

    // printf("analysis time: %g\n", solver.GetInformation(0));
    printf("best ordering method: %s\n", ORDERING_METHODS[(int)solver.GetInformation(16)]);


    // create threads (do only once)
    std::cout << "Threads: " << threads << std::endl;
    solver.CreateThreads(threads); // use specified # of threads, 0: all

    // factor & solve (first-time)
    for (int i; i< repLoops; i++){
        begin[1] = std::chrono::high_resolution_clock::now();
        solver.FactorizeMatrix(ax, 0); // use all created threads
        end[1] = std::chrono::high_resolution_clock::now();
        total[1] += std::chrono::duration_cast<std::chrono::microseconds>(end[1] - begin[1]).count();
        total[2] += solver.GetInformation(1)*1e6;
    }
    

    // printf("factor time: %g\n", solver.GetInformation(1));

    // std::cout << "Information Factorization: " << total[2] / repLoops << std::endl;
    std::cout << "Analyze time: " << total[0] / 1 << "\nFactorization time: " << total[2] / repLoops << std::endl;

    

EXIT:
    delete[] ax;
    delete[] ai;
    delete[] ap;
    delete[] b;
    delete[] x;
    solver.Free();
#ifdef _WIN32
    getchar();
#endif
    return 0;
}