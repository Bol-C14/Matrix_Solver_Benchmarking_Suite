#!/usr/bin/env python3
"""
superlu_benchmark.py

A flexible and portable Python script to benchmark SciPy's SuperLU for decomposing and solving sparse linear systems.

Usage:
    python superlu_benchmark.py --matrix path/to/matrix.mtx --repetitions 1000 --threads 4

Arguments:
    --matrix: Path to the Matrix Market (.mtx) file containing the sparse matrix.
    --repetitions: Number of times to repeat the factorization and solve for benchmarking.
    --threads: Number of threads to use (affects SciPy's internal OpenMP settings).
"""

import argparse
import os
import sys
import time
import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg
import progressbar
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark SciPy's SuperLU for sparse matrix decomposition and solving.")
    parser.add_argument('--matrix', type=str, required=True, help='Path to the Matrix Market (.mtx) file.')
    parser.add_argument('--repetitions', type=int, default=1000, help='Number of repetitions for factorization and solving.')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use (controls OpenMP threads).')
    parser.add_argument('--output', type=str, default='benchmark_results.csv', help='Path to the output CSV file.')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_num_threads(num_threads):
    """
    Set the number of threads for OpenMP via environment variables.
    """
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

def read_sparse_matrix(file_path):
    """
    Reads a Matrix Market file and returns a SciPy CSR sparse matrix.
    """
    if not os.path.exists(file_path):
        logging.error(f"Matrix file '{file_path}' does not exist.")
        sys.exit(1)
    
    try:
        logging.info(f"Reading matrix from '{file_path}'...")
        matrix = scipy.io.mmread(file_path)
        if not scipy.sparse.issparse(matrix):
            logging.warning("The loaded matrix is not sparse. Converting to CSR format.")
            matrix = scipy.sparse.csr_matrix(matrix)
        else:
            matrix = matrix.tocsr()
        logging.info(f"Matrix shape: {matrix.shape}")
        logging.info(f"Number of non-zeros: {matrix.nnz}")
        return matrix
    except Exception as e:
        logging.error(f"Failed to read the matrix file: {e}")
        sys.exit(1)

def benchmark_superlu(matrix, repetitions):
    """
    Benchmarks LU decomposition and solving using SciPy's SuperLU interface.

    Returns:
        A dictionary containing benchmark results.
    """
    results = {
        'matrix_shape': matrix.shape,
        'num_nonzeros': matrix.nnz,
        'analyze_times': [],
        'decomposition_times': [],
        'solving_times': [],
    }

    # Create the right-hand side vector 'b' as a vector of ones
    b = np.ones(matrix.shape[0])

    # Initialize progress bar
    bar = progressbar.ProgressBar(max_value=repetitions, redirect_stdout=True)

    for i in range(repetitions):
        # Start timing for analyzing (setup phase)
        start_analyze = time.perf_counter()
        # Placeholder for any analysis that might be needed.
        # In practice, this would include reordering or symbolic analysis.
        end_analyze = time.perf_counter()
        analyze_time = end_analyze - start_analyze
        results['analyze_times'].append(analyze_time)

        # Start timing for decomposition
        start_decompose = time.perf_counter()
        try:
            lu = scipy.sparse.linalg.splu(matrix)
        except Exception as e:
            logging.error(f"LU decomposition failed at repetition {i+1}: {e}")
            sys.exit(1)
        end_decompose = time.perf_counter()
        decomposition_time = end_decompose - start_decompose
        results['decomposition_times'].append(decomposition_time)

        # Start timing for solving
        start_solve = time.perf_counter()
        try:
            x = lu.solve(b)
        except Exception as e:
            logging.error(f"Solving failed at repetition {i+1}: {e}")
            sys.exit(1)
        end_solve = time.perf_counter()
        solving_time = end_solve - start_solve
        results['solving_times'].append(solving_time)

        bar.update(i+1)

    bar.finish()

    # Calculate average times
    results['average_analyze_time'] = np.mean(results['analyze_times'])
    results['average_decomposition_time'] = np.mean(results['decomposition_times'])
    results['average_solving_time'] = np.mean(results['solving_times'])
    results['total_analyze_time'] = np.sum(results['analyze_times'])
    results['total_decomposition_time'] = np.sum(results['decomposition_times'])
    results['total_solving_time'] = np.sum(results['solving_times'])

    return results

def save_results(results, output_path):
    """
    Saves the benchmark results to a CSV file.
    """
    import csv

    fieldnames = [
        'matrix_shape',
        'num_nonzeros',
        'repetitions',
        'average_analyze_time_sec',
        'average_decomposition_time_sec',
        'average_solving_time_sec',
        'total_analyze_time_sec',
        'total_decomposition_time_sec',
        'total_solving_time_sec'
    ]

    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({
            'matrix_shape': results['matrix_shape'],
            'num_nonzeros': results['num_nonzeros'],
            'repetitions': len(results['decomposition_times']),
            'average_analyze_time_sec': results['average_analyze_time'],
            'average_decomposition_time_sec': results['average_decomposition_time'],
            'average_solving_time_sec': results['average_solving_time'],
            'total_analyze_time_sec': results['total_analyze_time'],
            'total_decomposition_time_sec': results['total_decomposition_time'],
            'total_solving_time_sec': results['total_solving_time']
        })
    
    logging.info(f"Benchmark results saved to '{output_path}'.")

def main():
    args = parse_arguments()
    setup_logging()
    set_num_threads(args.threads)

    matrix = read_sparse_matrix(args.matrix)
    results = benchmark_superlu(matrix, args.repetitions)
    save_results(results, args.output)

    # Display summary
    logging.info("Benchmark Summary:")
    logging.info(f"Matrix Shape: {results['matrix_shape']}")
    logging.info(f"Number of Non-Zeros: {results['num_nonzeros']}")
    logging.info(f"Repetitions: {args.repetitions}")
    logging.info(f"Average Analyze Time: {results['average_analyze_time']:.6f} seconds")
    logging.info(f"Average Decomposition Time: {results['average_decomposition_time']:.6f} seconds")
    logging.info(f"Average Solving Time: {results['average_solving_time']:.6f} seconds")
    logging.info(f"Total Analyze Time: {results['total_analyze_time']:.6f} seconds")
    logging.info(f"Total Decomposition Time: {results['total_decomposition_time']:.6f} seconds")
    logging.info(f"Total Solving Time: {results['total_solving_time']:.6f} seconds")

if __name__ == "__main__":
    main()
