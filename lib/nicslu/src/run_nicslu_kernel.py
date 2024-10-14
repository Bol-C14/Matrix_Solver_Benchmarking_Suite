import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm  # Import tqdm for the progress bar
import psutil  # Optional for memory tracking if needed


class NicsluProcessor:
    def __init__(self, database_folder=None, engine=None, timeout=100):
        # Set up logging to log both to the console and a file
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler('nicslu_kernel.log'),  # Log to a file
                                logging.StreamHandler()  # Also log to the console
                            ])

        # Define paths and constants using environment variables or defaults
        self.database_folder = Path(database_folder or os.getenv("DATABASE_FOLDER", "./random_circuit_matrixs"))
        self.engine = engine or os.getenv("NICSLU_ENGINE", "./nicslu_kernel.o")
        self.timeout = timeout
        self.df = pd.DataFrame(
            columns=["index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze", "Factorization"])
        self.loc = 0

    # Function to run a single test command
    def run_single(self, nrhs, filename, bmatrix, reps, threads=1):
        try:
            command = [self.engine, str(nrhs), filename, bmatrix, str(reps), str(threads)]
            logging.info(f"Running command: {' '.join(command)}")

            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout)

            if p.returncode == 0:
                outLines = (p.stdout).decode('utf-8').split('\n')
                logging.info(f"Command output: {outLines}")
                time1 = float(outLines[-3].split(':')[1])
                time2 = float(outLines[-2].split(':')[1])
            else:
                logging.warning(f"Error running command: {p.stderr.decode('utf-8')}")
                time1 = -1
                time2 = -1

        except subprocess.TimeoutExpired:
            logging.warning(f"Command timed out: {' '.join(command)}")
            time1 = -2
            time2 = -2

        return time1, time2

    # Helper function to sort matrix files
    def custom_sort(self, matrix_name):
        parts = matrix_name.split('_')
        return int(parts[2]), int(parts[3])

    # Function to process matrices
    def process_matrices(self):
        mtxs = []

        # Iterate through the files and pick the .mtx files
        for filename in os.listdir(self.database_folder):
            if filename.endswith('.mtx') and filename.startswith('random_circuit'):
                mtxs.append(filename[:-4])

        if not mtxs:
            logging.warning("No .mtx files found in the directory!")
            return

        # Sort the list using the custom sort function
        mtxs = sorted(mtxs, key=self.custom_sort)

        # Define reps and threads
        reps = np.ones(len(mtxs), dtype=int) * 100
        threads = [1, 4, 8, 12, 24, 32]

        # Processing loop with progress bar and logging
        for j in range(len(threads)):
            for i in tqdm(range(len(mtxs)), desc=f"Processing Matrices with {threads[j]} threads"):
                filepath = os.path.join(self.database_folder, mtxs[i] + '.mtx')
                bmatrix = os.path.join(self.database_folder, mtxs[i], 'vecb.mtx')

                try:
                    logging.info(f"Reading matrix from {filepath}")
                    matrix = sio.mmread(filepath)
                    nnz = matrix.nnz
                    num_rows, num_cols = matrix.shape

                    logging.info(f"Running analysis on {filepath} with {nnz} non-zero elements and {num_rows} rows")
                    t1, t2 = self.run_single(1, filepath, bmatrix, reps[i], threads=threads[j])

                    # Append the results to the DataFrame
                    self.df.loc[self.loc, 'index'] = self.loc
                    self.df.loc[self.loc, 'mtx'] = mtxs[i]
                    self.df.loc[self.loc, 'algorithmname'] = 'NICSLU'
                    self.df.loc[self.loc, 'threads'] = threads[j]
                    self.df.loc[self.loc, 'nnz'] = nnz
                    self.df.loc[self.loc, 'rows'] = num_rows
                    self.df.loc[self.loc, 'Analyze'] = t1
                    self.df.loc[self.loc, 'Factorization'] = t2
                    logging.info(f"Finished processing {filepath}: Analyze time = {t1}, Factorization time = {t2}")

                except Exception as e:
                    logging.error(f"Error processing matrix {filepath}: {str(e)}")
                    # Continue to the next matrix without terminating the loop
                    continue

                self.loc += 1

        # Save results to CSV
        logging.info("Saving results to results_nicslu_kernel.csv")
        self.df.to_csv('results_nicslu_kernel.csv', index=False)
        logging.info("Results saved.")
        print("Processing complete!")


# Example usage:
if __name__ == "__main__":
    # Initialize the processor
    processor = NicsluProcessor(database_folder="/path/to/matrix/files", engine="./nicslu_kernel.o", timeout=100)

    # Run the processing method
    processor.process_matrices()
