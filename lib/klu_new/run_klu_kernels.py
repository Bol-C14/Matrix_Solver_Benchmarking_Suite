import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import psutil

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# File handlers for info and debug levels
info_handler = logging.FileHandler('klu_benchmark_info.log')
info_handler.setLevel(logging.INFO)

debug_handler = logging.FileHandler('klu_benchmark_debug.log')
debug_handler.setLevel(logging.DEBUG)

# Console handler for warnings and errors
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Set a common format for all handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(debug_handler)
logger.addHandler(console_handler)


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

    def run_single(self, engine, nrhs, filename, reps, bmatrix=None):
        """
        Run a single benchmark command for KLU on the given matrix file.
        """
        try:
            command = [engine, str(nrhs), filename, bmatrix, str(reps)]
            logging.debug(f"Running command: {' '.join(command)} with timeout {self.timeout}")

            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout)

            if p.returncode == 0:
                out_lines = (p.stdout).decode('utf-8').split('\n')
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
            logger.error(f"Directory {self.database_folder} does not exist!")
            raise FileNotFoundError(f"Directory {self.database_folder} does not exist!")

        logger.info(f"Looking for .mtx files in {self.database_folder}")
        # Iterate through files in the directory
        for filename in os.listdir(self.database_folder):
            if filename.endswith('.mtx'):  # Match all .mtx files
                # Ensure it's a file, not a directory
                file_path = self.database_folder / filename
                if file_path.is_file():
                    self.mtxs.append(filename[:-4])  # Add the filename without the extension
                else:
                    logger.debug(f"Skipping directory named like a .mtx file: {file_path}")

        if not self.mtxs:
            logger.warning("No .mtx files found in the directory!")
        else:
            logger.info(f"Found .mtx files: {self.mtxs}")

    def custom_sort(self, matrix_name):
        """
        Custom sorting function for matrix filenames.
        """
        # Sort alphabetically
        return matrix_name.lower()

    def run_benchmark(self):
        """
        Run the benchmark on all found matrix files and save the results to a CSV file.
        """
        self.mtxs = sorted(self.mtxs, key=self.custom_sort)
        reps = np.ones(len(self.mtxs), dtype=int) * 100

        logger.info(f"Starting benchmark on {len(self.mtxs)} matrices")

        for i in tqdm(range(len(self.mtxs)), desc="KLU - Processing Matrices"):
            filepath = self.database_folder / f"{self.mtxs[i]}.mtx"
            bmatrix = '/home/gushu/work/MLtask/ML_Circuit_Matrix_Analysis/data/vecb.mtx'

            try:
                logger.debug(f"Reading matrix from {filepath}")
                if not filepath.exists():
                    raise FileNotFoundError(f"Matrix file not found: {filepath}")

                matrix = sio.mmread(filepath)
                nnz = matrix.nnz
                num_rows, _ = matrix.shape

                logger.debug(f"Running analysis on {filepath} with {nnz} non-zero elements and {num_rows} rows")
                t1, t2 = self.run_single(self.engine, 1, str(filepath), reps[i], bmatrix)

                # Log completion details at the DEBUG level
                logger.debug(f"Completed {self.mtxs[i]}: Analyze = {t1}, Factorization = {t2}")
                self.df.loc[i] = [i, self.mtxs[i], 'KLU', 1, nnz, num_rows, t1, t2]

            except Exception as e:
                logger.error(f"Error processing matrix {filepath}: {str(e)}")
                continue

        logger.info("Benchmark completed for all matrices")
        logger.debug(f"Final DataFrame: \n{self.df}")
        self.df.to_csv('results_klu_kernel.csv', index=False)
        logger.info('Results saved to results_klu_kernel.csv')
        print('done!')


# Example usage of the class
if __name__ == "__main__":
    # Update the engine path and database folder as per your setup
    engine = './lib/klu_new/src/klu_kernel.o'
    database_folder = './data/ss_organized_data'

    klu_benchmark = KluBenchmark(engine, database_folder)
    klu_benchmark.find_mtx_files()
    klu_benchmark.run_benchmark()
