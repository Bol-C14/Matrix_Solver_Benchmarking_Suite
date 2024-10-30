import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

class NICSLUTester:
    def __init__(self, database_folder, engine_path='./nicslu_kernel.o', timeout=100):
        # Initialize paths and constants
        self.database_folder = Path(database_folder)
        self.engine = engine_path
        self.timeout = timeout
        self.df = pd.DataFrame(columns=["index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze", "Factorization"])
        
        # Set up logging
        logging.basicConfig(level=logging.DEBUG,  # Set initial level to DEBUG for full logging
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler('nicslu_kernel.log'),
                                logging.StreamHandler()
                            ])
        self.logger = logging.getLogger(__name__)

    def run_single(self, nrhs, filename, bmatrix, reps, threads=1):
        command = [self.engine, str(nrhs), filename, bmatrix, str(reps), str(threads)]
        self.logger.debug(f"Running command: {' '.join(command)}")

        try:
            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout)

            if p.returncode == 0:
                out_lines = (p.stdout).decode('utf-8').split('\n')
                self.logger.debug(f"Command output: {out_lines}")
                time1 = float(out_lines[-3].split(':')[1])
                time2 = float(out_lines[-2].split(':')[1])
            else:
                self.logger.warning(f"Error running command: {p.stderr.decode('utf-8')}")
                time1, time2 = -1, -1

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Command timed out: {' '.join(command)}")
            time1, time2 = -2, -2

        return time1, time2

    def custom_sort(self, matrix_name):
        parts = matrix_name.split('_')
        return int(parts[2]), int(parts[3])

    def process_matrices(self, reps_value=100, thread_values=None):
        if thread_values is None:
            thread_values = [1, 4, 8, 12, 24, 32]

        loc = 0
        mtxs = [f[:-4] for f in os.listdir(self.database_folder) if f.endswith('.mtx') and f.startswith('random_circuit')]

        if not mtxs:
            self.logger.warning("No .mtx files found in the directory!")

        mtxs.sort(key=self.custom_sort)
        reps = np.ones(len(mtxs), dtype=int) * reps_value

        self.logger.info("Starting matrix processing...")

        for threads in thread_values:
            for i in tqdm(range(len(mtxs)), desc=f"Processing Matrices with {threads} threads"):
                filepath = self.database_folder / f"{mtxs[i]}.mtx"
                bmatrix = self.database_folder / mtxs[i] / 'vecb.mtx'

                try:
                    self.logger.debug(f"Reading matrix from {filepath}")
                    matrix = sio.mmread(filepath)
                    nnz = matrix.nnz
                    num_rows, num_cols = matrix.shape

                    self.logger.debug(f"Running analysis on {filepath} with {nnz} non-zero elements and {num_rows} rows")
                    t1, t2 = self.run_single(1, str(filepath), str(bmatrix), reps[i], threads=threads)

                    self.df.loc[loc] = [loc, mtxs[i], 'NICSLU', threads, nnz, num_rows, t1, t2]
                    self.logger.debug(f"Finished processing {filepath}: Analyze time = {t1}, Factorization time = {t2}")

                except Exception as e:
                    self.logger.error(f"Error processing matrix {filepath}: {str(e)}")
                    continue

                loc += 1

        self.logger.info("Matrix processing completed.")
        self.save_results()

    def save_results(self, filename='results_nicslu_kernel.csv'):
        self.logger.info(f"Saving results to {filename}")
        self.df.to_csv(filename, index=False)
        self.logger.info("Results saved.")

    def run(self, reps_value=100, thread_values=None):
        self.logger.info("Starting the NICSLU benchmark...")
        self.process_matrices(reps_value, thread_values)
        self.logger.info("NICSLU Benchmark completed.")

# Wrapper function with argparse for command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Run NICSLU tests on matrix files.")
    parser.add_argument("--database_folder", type=str, default="/mnt/c/Work/transient-simulation/Random_Circuit_Generator/random_circuit_matrixs", help="Path to the database folder with matrix files")
    parser.add_argument("--engine_path", type=str, default="./nicslu_kernel.o", help="Path to the NICSLU engine executable")
    parser.add_argument("--timeout", type=int, default=100, help="Timeout in seconds for each command")
    parser.add_argument("--reps_value", type=int, default=100, help="Number of repetitions for each matrix")
    parser.add_argument("--thread_values", nargs='+', type=int, default=[1, 4, 8, 12, 24, 32], help="List of thread values to use")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    args = parser.parse_args()

    # Set logging level based on the provided log_level argument
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    tester = NICSLUTester(
        database_folder=args.database_folder,
        engine_path=args.engine_path,
        timeout=args.timeout
    )

    tester.run(reps_value=args.reps_value, thread_values=args.thread_values)

if __name__ == "__main__":
    main()
