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
    def __init__(self, database_folder, engine_path='./lib/nicslu/src/nicslu_kernel.o', timeout=100):
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

    def run_single(self, nrhs, filename, reps, threads=1, bmatrix=None):
        command = [self.engine, str(nrhs), filename]
        if bmatrix:
            command.append(str(bmatrix))
        command.extend([str(reps), str(threads)])
        self.logger.debug(f"Running command: {' '.join(command)}")

        try:
            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout)

            if p.returncode == 0:
                out_str = p.stdout.decode('utf-8')
                out_lines = out_str.strip().split('\n')
                self.logger.debug(f"Command output:\n{out_str}")

                if len(out_lines) < 3:
                    self.logger.error(f"Unexpected output format from NICSLU executable for file {filename}.")
                    self.logger.error(f"Output received:\n{out_str}")
                    time1, time2 = -3, -3  # Custom error codes
                else:
                    try:
                        time1 = float(out_lines[-3].split(':')[1].strip())
                        time2 = float(out_lines[-2].split(':')[1].strip())
                    except (IndexError, ValueError) as e:
                        self.logger.error(f"Error parsing output for file {filename}: {e}")
                        self.logger.error(f"Output received:\n{out_str}")
                        time1, time2 = -4, -4  # Custom error codes
            else:
                error_output = p.stderr.decode('utf-8')
                self.logger.warning(f"Error running command: {error_output}")
                time1, time2 = -1, -1

        except subprocess.TimeoutExpired:
            command_str = ' '.join(command)
            self.logger.warning(f"Command timed out: {command_str}")
            time1, time2 = -2, -2

        except Exception as e:
            self.logger.error(f"Unexpected error running command {' '.join(command)}: {e}")
            time1, time2 = -5, -5  # Custom error codes

        return time1, time2

    def custom_sort(self, matrix_name):
        # Sort alphabetically
        return matrix_name.lower()

    def process_matrices(self, reps_value=100, thread_values=None):
        """
        Process all .mtx files in the database folder using specified thread values.
        """
        if thread_values is None:
            thread_values = [1, 4, 8, 12, 24, 32]

        loc = 0

        # Include all .mtx files in the directory
        mtxs = [f[:-4] for f in os.listdir(self.database_folder) if f.endswith('.mtx')]

        if not mtxs:
            self.logger.warning("No .mtx files found in the directory!")
            return

        mtxs.sort(key=self.custom_sort)  # Sort the files for consistent processing
        reps = np.ones(len(mtxs), dtype=int) * reps_value

        self.logger.info("Starting matrix processing...")

        for threads in thread_values:
            for i in tqdm(range(len(mtxs)), desc=f"Processing Matrices with {threads} threads"):
                filepath = self.database_folder / f"{mtxs[i]}.mtx"
                bmatrix = '/home/gushu/work/MLtask/ML_Circuit_Matrix_Analysis/data/vecb.mtx'

                try:
                    self.logger.debug(f"Reading matrix from {filepath}")
                    
                    # Read the .mtx file
                    if not filepath.exists():
                        raise FileNotFoundError(f"Matrix file not found: {filepath}")
                    
                    matrix = sio.mmread(filepath)  # Assumes scipy.io.mmread is used for reading
                    nnz = matrix.nnz
                    num_rows, num_cols = matrix.shape

                    self.logger.debug(f"Running analysis on {filepath} with {nnz} non-zero elements and {num_rows} rows")

                    # Run the analysis
                    t1, t2 = self.run_single(1, str(filepath), reps[i], threads=threads, bmatrix=bmatrix)

                    # Save results
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
    parser.add_argument("--database_folder", type=str, default="./data/ss_organized_data", help="Path to the database folder with matrix files")
    parser.add_argument("--engine_path", type=str, default="./lib/nicslu/src/nicslu_kernel.o", help="Path to the NICSLU engine executable")
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
