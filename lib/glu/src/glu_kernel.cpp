#include <iostream>
#include <vector>
#include <set>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <fstream>
#include "symbolic.h"
#include "numeric.h"
#include "Timer.h"
#include "preprocess.h"
#include "nicslu.h"
#include <chrono>


using namespace std;

void help_message()
{
    cout << endl;
    cout << "GLU program V3.0" << endl;
    cout << "Usage: ./lu_cmd -i inputfile" << endl;
    cout << "Additional usage: ./lu_cmd -i inputfile -p" << endl;
    cout << "-p to enable perturbation" << endl;
}

int main(int argc, char** argv)
{
    Timer t;
    double utime;
    SNicsLU *nicslu;

    char *matrixName = NULL;
    bool PERTURB = false;

    double *ax = NULL;
    unsigned int *ai = NULL, *ap = NULL;
    unsigned int n;

    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <nrhs>" << std::endl;
        return 1;
    }
    std::string filename = "";
    std::string bmatrix = "";

    // For timing
    using Chrono = std::chrono::time_point<std::chrono::high_resolution_clock>;
    Chrono begin[3], end[3];
    double total[3] = {0};

    if (argc == 2)
    {
        filename = "circuit_2.mtx";
        bmatrix = "./vecb2.mtx";
        // filename = "./circuit_2/circuit_2.mtx";
        // bmatrix = "./circuit_2/vecb.mtx";
        // filename = "./scircuit/scircuit.mtx";
        // bmatrix = "./scircuit/vecb.mtx";
    }

    if (argc == 4)
    {
        filename = argv[2];
        bmatrix = argv[3];
    }

    matrixName = new char[filename.size() + 1];
    strcpy(matrixName, filename.c_str());

    nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));

    int err = preprocess(matrixName, nicslu, &ax, &ai, &ap);

    if (err)
    {
        // cout << "Reading matrix error" << endl;
        exit(1);
    }

    n = nicslu->n;

    cout << "Matrix Row: " << n << endl;
    cout << "Original nonzero: " << nicslu->nnz << endl;

    Symbolic_Matrix A_sym(n, cout, cerr);

    begin[0] = std::chrono::high_resolution_clock::now();
    A_sym.fill_in(ai, ap);

    // cout << "Symbolic time: " << utime << " ms" << endl;


    A_sym.csr();

    // cout << "CSR time: " << utime << " ms" << endl;


    A_sym.predictLU(ai, ap, ax);

    // cout << "PredictLU time: " << utime << " ms" << endl;


    A_sym.leveling();
    // cout << "Leveling time: " << utime << " ms" << endl;
    end[0] = std::chrono::high_resolution_clock::now();
    total[0] += std::chrono::duration_cast<std::chrono::microseconds>(end[0] - begin[0]).count();
    std::cout << "Analyze time: " << total[0] << std::endl;

#if GLU_DEBUG
    A_sym.ABFTCalculateCCA();
//    A_sym.PrintLevel();
#endif

    begin[1] = std::chrono::high_resolution_clock::now();
    LUonDevice(A_sym, cout, cerr, PERTURB);
    end[1] = std::chrono::high_resolution_clock::now();
    total[1] += std::chrono::duration_cast<std::chrono::microseconds>(end[1] - begin[1]).count();
    

    // clean up
    delete[] matrixName;

#if GLU_DEBUG
    A_sym.ABFTCheckResult();
#endif

    // //solve Ax=b
    // vector<REAL> b(n, 1.);
    // vector<REAL> x = A_sym.solve(nicslu, b);
    // {
    //     ofstream x_f("x.dat");
    //     for (double xx: x)
    //         x_f << xx << '\n';
    // }

}
