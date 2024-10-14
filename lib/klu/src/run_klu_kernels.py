import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm  # Import tqdm for the progress bar
import psutil

class KluBenchmark:
    def __init__(self, engine, database_folder, timeout=100):
        """
        Initialize the KluBenchmark class with the engine path, database folder, and timeout.
        """
        self.engine = engine
        self.database_folder = Path(database_folder)
        self.timeout = timeout
        self.df = pd.DataFrame(columns=[
            "index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze", "Factorization"
        ])
        self.mtxs = []

    def run_single(self, engine, nrhs, filename, bmatrix, reps):
        """
        Run a single benchmark command for KLU on the given matrix file.
        """
        try:
            command = [engine, str(nrhs), filename, bmatrix, str(reps)]
            logging.info(f"Running command: {' '.join(command)}")

            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout)

            if p.returncode == 0:
                out_lines = (p.stdout).decode('utf-8').split('\n')
                logging.info(f"Command output: {out_lines}")
                time1 = float(out_lines[-3].split(':')[1])
                time2 = float(out_lines[-2].split(':')[1])
            else:
                logging.warning(f"Error running command: {p.stderr.decode('utf-8')}")
                time1 = -1
                time2 = -1

        except subprocess.TimeoutExpired:
            logging.warning(f"Command timed out: {' '.join(command)}")
            time1 = -2
            time2 = -2

        return time1, time2

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

    def run_benchmark(self):
        """
        Run the benchmark on all found matrix files and save the results to a CSV file.
        """
        self.mtxs = sorted(self.mtxs, key=self.custom_sort)
        reps = np.ones(len(self.mtxs), dtype=int) * 100

        for i in tqdm(range(len(self.mtxs)), desc="Processing Matrices"):
            filepath = str(self.database_folder / (self.mtxs[i] + '.mtx'))
            bmatrix = str(self.database_folder / self.mtxs[i] / 'vecb.mtx')

            try:
                logging.info(f"Reading matrix from {filepath}")
                matrix = sio.mmread(filepath)
                nnz = matrix.nnz
                num_rows, _ = matrix.shape

                logging.info(f"Running analysis on {filepath} with {nnz} non-zero elements and {num_rows} rows")
                t1, t2 = self.run_single(self.engine, 1, filepath, bmatrix, reps[i])

                self.df.loc[i] = [i, self.mtxs[i], 'KLU', 1, nnz, num_rows, t1, t2]
                logging.info(f"Finished processing {filepath}: Analyze time = {t1}, Factorization time = {t2}")

            except Exception as e:
                logging.error(f"Error processing matrix {filepath}: {str(e)}")
                continue

        logging.info(f"Final DataFrame: \n{self.df}")
        self.df.to_csv('results_klu_kernel.csv', index=False)
        logging.info('Results saved to results_klu_kernel.csv')
        print('done!')

# Example usage of the class
if __name__ == "__main__":
    engine = './klu_kernel.o'
    database_folder = "/mnt/c/Work/transient-simulation/Random_Circuit_Generator/random_circuit_matrixs"

    klu_benchmark = KluBenchmark(engine, database_folder)
    klu_benchmark.find_mtx_files()
    klu_benchmark.run_benchmark()
