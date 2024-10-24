import os
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.sparse.linalg
from pathlib import Path
import logging
from tqdm import tqdm
import time


class SuperluBenchmark:
    def __init__(self, database_folder, timeout=100):
        """
        Initialize the SuperluBenchmark class with the database folder and timeout.
        """
        self.database_folder = Path(database_folder)
        self.timeout = timeout
        self.df = pd.DataFrame(columns=[
            "index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze",
            "Factorization", "Read", "RHS_Creation", "Average_Solve"
        ])
        self.mtxs = []

    def find_mtx_files(self):
        """
        Find all matrix (.mtx) files in the database folder.
        """
        if not self.database_folder.exists():
            logging.error(f"Directory {self.database_folder} does not exist!")
            raise FileNotFoundError(f"Directory {self.database_folder} does not exist!")

        logging.info(f"Looking for .mtx files in {self.database_folder}")
        for filename in os.listdir(self.database_folder):
            if filename.endswith('.mtx') and filename.startswith('random_circuit'):
                self.mtxs.append(filename[:-4])

        if not self.mtxs:
            logging.warning("No .mtx files found in the directory!")

    def custom_sort(self, matrix_name):
        """
        Custom sorting function for matrix filenames.
        """
        parts = matrix_name.split('_')
        return int(parts[2]), int(parts[3])

    def run_single(self, matrix_path, reps=100):
        """
        Run a single benchmark for LU decomposition on a given matrix.
        """
        try:
            # Measure the time for reading the matrix
            start_read = time.perf_counter()
            logging.info(f"Reading matrix from {matrix_path}")
            matrix = sio.mmread(matrix_path)
            if not scipy.sparse.issparse(matrix):
                matrix = scipy.sparse.csr_matrix(matrix)
            end_read = time.perf_counter()
            read_time = end_read - start_read
            logging.info(f"Read time for {matrix_path}: {read_time:.6f} seconds")

            nnz = matrix.nnz
            num_rows, _ = matrix.shape

            # Create the right-hand side vector 'b' as a vector of ones
            start_rhs = time.perf_counter()
            b = np.ones(num_rows)
            end_rhs = time.perf_counter()
            rhs_time = end_rhs - start_rhs
            logging.info(f"Right-hand side vector creation time for {matrix_path}: {rhs_time:.6f} seconds")

            # Measure analyze time (symbolic analysis)
            start_analyze = time.perf_counter()
            # Placeholder for analysis - we can assume reading matrix is part of analysis in this case
            end_analyze = time.perf_counter()
            analyze_time = end_analyze - start_analyze
            logging.info(f"Analyze time for {matrix_path}: {analyze_time:.6f} seconds")

            # Measure factorization time and solve the system
            factor_times = []
            solve_times = []
            for _ in range(reps):
                # Measure the time for LU factorization
                start_factor = time.perf_counter()
                lu = scipy.sparse.linalg.splu(matrix)
                end_factor = time.perf_counter()
                factor_time = end_factor - start_factor
                factor_times.append(factor_time)
                logging.info(f"Factorization time for {matrix_path}: {factor_time:.6f} seconds")

                # Measure the time for solving the system
                start_solve = time.perf_counter()
                x = lu.solve(b)
                end_solve = time.perf_counter()
                solve_time = end_solve - start_solve
                solve_times.append(solve_time)
                logging.info(f"Solve time for {matrix_path}: {solve_time:.6f} seconds")

            # Calculate average times for factorization and solving
            average_factor_time = np.mean(factor_times)
            average_solve_time = np.mean(solve_times)
            logging.info(f"Average factorization time for {matrix_path}: {average_factor_time:.6f} seconds")
            logging.info(f"Average solve time for {matrix_path}: {average_solve_time:.6f} seconds")

            return analyze_time, average_factor_time, nnz, num_rows, read_time, rhs_time, average_solve_time

        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            return -1, -1, -1, -1, -1, -1, -1

    def run_benchmark(self):
        """
        Run the benchmark on all found matrix files and save the results to a CSV file.
        """
        self.mtxs = sorted(self.mtxs, key=self.custom_sort)
        reps = np.ones(len(self.mtxs), dtype=int) * 1

        for i in tqdm(range(len(self.mtxs)), desc="Processing Matrices"):
            filepath = str(self.database_folder / (self.mtxs[i] + '.mtx'))

            try:
                t1, t2, nnz, num_rows, read_time, rhs_time, avg_solve_time = self.run_single(filepath, reps[i])

                if t1 >= 0 and t2 >= 0:
                    self.df.loc[i] = [
                        i, self.mtxs[i], 'SUPERLU', 1, nnz, num_rows, t1, t2,
                        read_time, rhs_time, avg_solve_time
                    ]
                    logging.info(f"Finished processing {filepath}: Analyze time = {t1}, Factorization time = {t2}, Read time = {read_time}, RHS time = {rhs_time}, Average solve time = {avg_solve_time}")
                else:
                    logging.warning(f"Skipping {filepath} due to an error during processing.")

            except Exception as e:
                logging.error(f"Error processing matrix {filepath}: {str(e)}")
                continue

        logging.info(f"Final DataFrame: \n{self.df}")
        self.df.to_csv('results_superlu_scipy.csv', index=False)
        logging.info('Results saved to results_superlu_scipy.csv')
        print('done!')


# Example usage of the class
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    database_folder = "/mnt/c/Work/transient-simulation/Random_Circuit_Generator/random_circuit_matrixs"
    superlu_benchmark = SuperluBenchmark(database_folder)
    superlu_benchmark.find_mtx_files()
    superlu_benchmark.run_benchmark()
