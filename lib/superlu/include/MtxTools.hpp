/**
 *
 *
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <limits>
#include <algorithm>

namespace MtxTools
{
    static constexpr bool debug = false;

    using index_t = std::vector<int>::size_type;

    struct MTX_COO_LINE
    {
        int row = 0;
        int col = 0;
        double data = 0;
    };

    struct MTX_COO
    {
        int nrows = 0;
        int ncols = 0;
        int nnz = 0;
        std::vector<MTX_COO_LINE> line = {};
        bool symmetric = false;
        bool pattern = false;
        int status = 0;
    };

    struct MTX_VEC
    {
        int nrows = 0;
        std::vector<double> data = {};
        int status = 0;
    };

    // Compressed Sparse Matrix Format
    struct MTX_CSX
    {
        int nrows = 0;
        int ncols = 0;
        int nnz = 0;
        std::vector<int> indices = {};
        std::vector<int> indptr = {};
        std::vector<double> data = {};
        int status = 0;
    };

    void read_mtx_generic(std::ifstream &file, MTX_COO &coo)
    {

        std::string line;
        std::getline(file, line);
        std::stringstream ss(line);
        ss >> coo.nrows >> coo.ncols >> coo.nnz;

        coo.line.resize(coo.nnz);

        if (debug)
            std::cout << "nrows: " << coo.nrows << " ncols: " << coo.ncols << " nnz: " << coo.nnz << std::endl;

        for (int i = 0; i < coo.nnz; i++)
        {
            std::getline(file, line);
            std::stringstream ss(line);
            int row, col;
            double val = 0.0;
            if (coo.pattern)
            {
                ss >> row >> col;
                val = 1.0;
            }
            else
            {
                ss >> row >> col >> val;
            }

            // convert to zero-based indexing
            coo.line[i].row = row - 1;
            coo.line[i].col = col - 1;
            coo.line[i].data = val;

            if (debug)
                std::cout << "row: " << coo.line[i].row << " col: " << coo.line[i].col << " val: " << coo.line[i].data << std::endl;
        }

        if (debug)
            std::cout << "COO matrix read\n"
                      << std::endl;
    }

    void read_mtx_symmetric(std::ifstream &file, MTX_COO &coo)
    {

        std::string line;
        std::getline(file, line);
        std::stringstream ss(line);
        ss >> coo.nrows >> coo.ncols >> coo.nnz;

        coo.line.resize(coo.nnz * 2);

        if (debug)
            std::cout << "nrows: " << coo.nrows << " ncols: " << coo.ncols << " nnz: " << coo.nnz << std::endl;

        for (int i = 0; i < coo.nnz * 2; i += 2)
        {
            std::getline(file, line);
            std::stringstream ss(line);
            int row, col;
            double val = 0.0;
            if (coo.pattern)
            {
                ss >> row >> col;
                val = 1.0;
            }
            else
            {
                ss >> row >> col >> val;
            }

            // convert to zero-based indexing
            coo.line[i].row = row - 1;
            coo.line[i].col = col - 1;
            coo.line[i].data = val;

            coo.line[i + 1].col = row - 1;
            coo.line[i + 1].row = col - 1;
            coo.line[i + 1].data = val;
        }

        if (debug)
        {
            for (index_t i = 0; i < coo.line.size(); i++)
            {
                std::cout << "row: " << coo.line[i].row << " col: " << coo.line[i].col << " val: " << coo.line[i].data << std::endl;
            }

            std::cout << "round 1" << std::endl;
        }

        coo.line.erase(std::unique(coo.line.begin(), coo.line.end(), [](const MTX_COO_LINE &a, const MTX_COO_LINE &b)
                                   { return (a.col == b.col) && (a.row == b.row); }),
                       coo.line.end());

        if (debug)
        {
            for (index_t i = 0; i < coo.line.size(); i++)
            {
                std::cout << "row: " << coo.line[i].row << " col: " << coo.line[i].col << " val: " << coo.line[i].data << std::endl;
            }
        }

        coo.nnz = coo.line.size();

        if (debug)
            std::cout << "COO matrix read\n"
                      << std::endl;
    }

    MTX_COO read_mtx_coo(std::string filename)
    {
        MTX_COO coo;

        std::ifstream file(filename);
        if (!file.is_open())
        {
            coo.status = -1;
            return coo;
        }

        std::string line;
        std::getline(file, line);
        if (line.find("%%MatrixMarket") == std::string::npos)
        {
            coo.status = -1;
            return coo;
        }

        if (line.find("symmetric") != std::string::npos)
        {
            coo.symmetric = true;
            if (debug)
                std::cout << "symmetric matrix" << std::endl;
        }
        else
        {
            coo.symmetric = false;
        }

        if (line.find("pattern") != std::string::npos)
        {
            coo.pattern = true;
            if (debug)
                std::cout << "pattern matrix" << std::endl;
        }
        else
        {
            coo.pattern = false;
        }

        // Ignore header
        while (file.peek() == '%')
        {
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        if (coo.symmetric)
        {
            read_mtx_symmetric(file, coo);
        }
        else
        {
            read_mtx_generic(file, coo);
        }

        return coo;
    }

    void coo_sort_row(MTX_COO &mtx)
    {
        std::stable_sort(mtx.line.begin(), mtx.line.end(), [](const MTX_COO_LINE &lhs, const MTX_COO_LINE &rhs)
                         { return lhs.row < rhs.row; });
    }

    void coo_sort_col(MTX_COO &mtx)
    {
        std::stable_sort(mtx.line.begin(), mtx.line.end(), [](const MTX_COO_LINE &lhs, const MTX_COO_LINE &rhs)
                         { return lhs.col < rhs.col; });
    }

    MTX_CSX coo_to_csc(MTX_COO coo)
    {
        coo_sort_col(coo);

        if (debug)
            std::cout << "\nCOO matrix sorted by column" << std::endl;

        if (debug)
        {
            for (index_t i = 0; i < coo.line.size(); i++)
            {
                std::cout << "row: " << coo.line[i].row << " col: " << coo.line[i].col << " val: " << coo.line[i].data << std::endl;
            }

            std::cout << "COO matrix sorted\n"
                      << std::endl;
        }

        MTX_CSX csc;
        csc.nrows = coo.nrows;
        csc.ncols = coo.ncols;
        csc.nnz = coo.nnz;

        csc.indices.resize(coo.nnz);
        csc.indptr.resize(coo.ncols + 1);
        csc.data.resize(coo.nnz);

        for (int i = 0; i < coo.nnz; i++)
        {
            csc.indices[i] = coo.line[i].row;
            csc.data[i] = coo.line[i].data;
        }

        csc.indptr[0] = 0;
        /*for (int i = 0; i < coo.ncols; i++)
        {
            csc.indptr[i + 1] = csc.indptr[i] + std::count_if(csc.indices.begin(), csc.indices.end(),
                                                              [i](int j) { return j == i; });
        }*/

        for (int i = 1; i < coo.nnz; i++)
        {
            const int col_prev = coo.line[i - 1].col;
            const int col = coo.line[i].col;
            if (col_prev < col)
            {
                csc.indptr[col] = i;
            }
        }
        csc.indptr[coo.ncols] = coo.nnz;

        return csc;
    }

    MTX_VEC read_mtx_vec(std::string filename)
    {
        MTX_VEC vec;

        std::ifstream file(filename);
        if (!file.is_open())
        {
            vec.status = -1;
            return vec;
        }

        std::string line;
        std::getline(file, line);
        if (line.find("%%MatrixMarket") == std::string::npos)
        {
            vec.status = -1;
            return vec;
        }

        if (line.find("array") == std::string::npos)
        {
            vec.status = -1;
            return vec;
        }

        // Ignore header
        while (file.peek() == '%')
        {
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        line = "";

        std::getline(file, line);
        std::stringstream ss(line);
        int nrows, ncols;
        ss >> nrows >> ncols;

        if (ncols != 1)
        {
            std::cout << "Error: vector matrix must have only one column" << std::endl;
            vec.status = -1;
            return vec;
        }

        vec.data.resize(nrows);

        if (debug)
            std::cout << "nrows: " << nrows << " ncols: " << ncols << std::endl;

        for (int i = 0; i < nrows; i++)
        {
            std::getline(file, line);
            std::stringstream ss(line);
            double val;
            ss >> val;
            vec.data[i] = val;

            if (debug)
                std::cout << "val: " << vec.data[i] << std::endl;
        }

        if (debug)
            std::cout << "vec matrix read\n"
                      << std::endl;

        return vec;
    }

    int write_mtx_vec(std::string filename, MTX_VEC vec)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cout << "Error: could not open file " << filename << std::endl;
            return -1;
        }

        file << "%%MatrixMarket matrix array real general\n";
        file << vec.data.size() << " 1\n";

        for (int i = 0; i < vec.data.size(); i++)
        {
            file << vec.data[i] << "\n";
        }

        file.close();
        return 0;
    }

    MTX_CSX coo_to_csr(MTX_COO coo)
    {
        coo_sort_row(coo);

        if (debug)
            std::cout << "\nCOO matrix sorted by column" << std::endl;

        if (debug)
        {
            for (index_t i = 0; i < coo.line.size(); i++)
            {
                std::cout << "row: " << coo.line[i].row << " col: " << coo.line[i].col << " val: " << coo.line[i].data << std::endl;
            }

            std::cout << "COO matrix sorted\n"
                      << std::endl;
        }

        MTX_CSX csr;
        csr.nrows = coo.nrows;
        csr.ncols = coo.ncols;
        csr.nnz = coo.nnz;

        csr.indices.resize(coo.nnz);
        csr.indptr.resize(coo.nrows + 1);
        csr.data.resize(coo.nnz);

        for (int i = 0; i < coo.nnz; i++)
        {
            csr.indices[i] = coo.line[i].col;
            csr.data[i] = coo.line[i].data;
        }

        csr.indptr[0] = 0;
        /*for (int i = 0; i < coo.ncols; i++)
        {
            csc.indptr[i + 1] = csc.indptr[i] + std::count_if(csc.indices.begin(), csc.indices.end(),
                                                              [i](int j) { return j == i; });
        }*/

        for (int i = 1; i < coo.nnz; i++)
        {
            const int row_prev = coo.line[i - 1].row;
            const int row = coo.line[i].row;
            if (row_prev < row)
            {
                csr.indptr[row] = i;
            }
        }
        csr.indptr[coo.nrows] = coo.nnz;

        return csr;
    }
}
