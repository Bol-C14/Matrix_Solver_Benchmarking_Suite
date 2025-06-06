import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import psutil
import argparse

class PardisoKernelBenchmark:
    def __init__(self, engine='./pardiso_kernel.o', database_folder="/mnt/c/Work/transient-simulation/Random_Circuit_Generator/random_circuit_matrixs", timeout=100, log_level=logging.INFO, output_dir="."):
        """
        Initialize the PardisoKernelBenchmark class with paths, timeout, and logging configurations.
        """
        self.engine = engine
        self.database_folder = Path(database_folder)
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self.df = pd.DataFrame(columns=["index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze", "Factorization"])

        # Set up logging based on log level
        self.setup_logging(log_level)

    def setup_logging(self, log_level):
        """
        Configures logging with the specified log level.
        """
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler('pardiso_kernel.log'),
                                logging.StreamHandler()
                            ])

    def run_single(self, nrhs, filename, bmatrix, reps, threads=1):
        """
        Runs a single benchmark command for PARDISO with specified threads.
        """
        try:
            command = [self.engine, str(nrhs), filename, bmatrix, str(reps), str(threads)]
            logging.debug(f"Running command: {' '.join(command)}")
            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout)

            if p.returncode == 0:
                out_lines = p.stdout.decode('utf-8').split('\n')
                time1 = float(out_lines[-3].split(':')[1])
                time2 = float(out_lines[-2].split(':')[1])
            else:
                logging.error(f"Error running command: {p.stderr.decode('utf-8')}")
                time1 = time2 = -1

        except subprocess.TimeoutExpired:
            logging.warning(f"Command timed out: {' '.join(command)}")
            time1 = time2 = -2

        return time1, time2

    @staticmethod
    def custom_sort(matrix_name):
        """
        Sort helper function based on matrix file naming conventions.
        """
        parts = matrix_name.split('_')
        return int(parts[2]), int(parts[3])

    def process_matrices(self, repetitions=100, thread_values=[1, 4, 8, 12, 24, 32]):
        """
        Processes all matrix files in the database, running benchmarks with various threads.
        """
        loc = 0
        mtxs = [filename[:-4] for filename in os.listdir(self.database_folder) if filename.endswith('.mtx')]

        if not mtxs:
            logging.warning("No .mtx files found in the directory!")
            return

        mtxs = sorted(mtxs)
        reps = np.ones(len(mtxs), dtype=int) * repetitions

        logging.info(f"Starting PARDISO benchmarks on {len(mtxs)} matrices")

        for j, thread_count in enumerate(thread_values):
            for i in tqdm(range(len(mtxs)), desc=f"Processing with {thread_count} threads"):
                filepath = os.path.join(self.database_folder, mtxs[i] + '.mtx')
                bmatrix = os.path.join(self.database_folder, mtxs[i], 'vecb.mtx')

                try:
                    logging.debug(f"Reading matrix from {filepath}")

                    # Attempt to read the matrix and log errors if encountered
                    try:
                        matrix = sio.mmread(filepath)
                        nnz = matrix.nnz
                        num_rows, _ = matrix.shape
                        logging.debug(f"Successfully read {mtxs[i]}: {nnz} non-zero elements, {num_rows} rows")
                    except Exception as e:
                        logging.error(f"Error reading matrix {filepath}: {str(e)}")
                        continue

                    t1, t2 = self.run_single(1, filepath, bmatrix, reps[i], threads=thread_count)

                    # Append results to DataFrame
                    self.df.loc[loc] = [loc, mtxs[i], 'PARDISO', thread_count, nnz, num_rows, t1, t2]
                    logging.debug(f"Finished processing {filepath}: Analyze time = {t1}, Factorization time = {t2}")

                except Exception as e:
                    logging.error(f"Error processing matrix {filepath}: {str(e)}")
                    continue

                loc += 1
                logging.debug(f"Memory usage: {psutil.virtual_memory().percent}%")

        logging.info("PARDISO benchmarks completed.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_file = self.output_dir / 'results_pardiso_kernel.csv'
        self.df.to_csv(out_file, index=False)
        logging.info(f"Results saved to {out_file}")

# Wrapper function for external usage
def run_pardiso_benchmark(engine='./pardiso_kernel.o', database_folder="./data/ss_organized_data", timeout=100, log_level=logging.INFO, repetitions=100, thread_values=[1, 4, 8, 12, 24, 32], output_dir="."):
    """
    Run the PARDISO kernel benchmark with adjustable parameters.
    """
    pardiso_benchmark = PardisoKernelBenchmark(
        engine=engine,
        database_folder=database_folder,
        timeout=timeout,
        log_level=log_level,
        output_dir=output_dir
    )
    pardiso_benchmark.process_matrices(repetitions=repetitions, thread_values=thread_values)

# Run the main processing function only if executed directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PARDISO kernel benchmark")
    parser.add_argument("--engine", type=str, default="./pardiso_kernel.o", help="Path to PARDISO executable")
    parser.add_argument("--database_folder", type=str, default="./data/ss_organized_data", help="Directory with matrix files")
    parser.add_argument("--output_dir", type=str, default="./data/results/pardiso", help="Directory to store result CSV")
    parser.add_argument("--timeout", type=int, default=100, help="Timeout for each run in seconds")
    parser.add_argument("--repetitions", type=int, default=100, help="Repetitions for each matrix")
    parser.add_argument("--thread_values", nargs='+', type=int, default=[1,4,8,12,24,32], help="Thread counts")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    run_pardiso_benchmark(engine=args.engine, database_folder=args.database_folder, timeout=args.timeout,
                          log_level=log_level, repetitions=args.repetitions, thread_values=args.thread_values,
                          output_dir=args.output_dir)
