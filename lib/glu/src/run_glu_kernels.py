import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import psutil


class GluKernelBenchmark:
    def __init__(self, engine='./glu_kernel', database_folder="/mnt/c/Work/transient-simulation/Random_Circuit_Generator/random_circuit_matrixs", timeout=100, log_level=logging.INFO):
        """
        Initialize the GluKernelBenchmark class with the engine path, database folder, timeout, and log level.
        """
        self.engine = engine
        self.database_folder = Path(database_folder)
        self.timeout = timeout
        self.df = pd.DataFrame(columns=["index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze", "Factorization_1", "Factorization_2"])
        
        # Set up logging
        self.set_logging(log_level)

    def set_logging(self, log_level):
        """
        Configure logging with the specified log level.
        """
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler('glu_kernel.log'),
                                logging.StreamHandler()
                            ])

    def run_single(self, nrhs, filename, bmatrix):
        """
        Run a single benchmark command.
        """
        try:
            command = [self.engine, str(nrhs), filename, bmatrix]
            logging.debug(f"Running command: {' '.join(command)}")
            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout)

            if p.returncode == 0:
                outLines = p.stdout.decode('utf-8').split('\n')
                time1 = float(outLines[-3].split(':')[1])
                time2 = float(outLines[-2].split(':')[1])
            else:
                logging.error(f"Error running command: {p.stderr.decode('utf-8')}")
                time1 = time2 = -1

        except subprocess.TimeoutExpired:
            logging.warning(f"Command timed out: {' '.join(command)}")
            time1 = time2 = -2

        return time1, time2

    def run_double(self, nrhs, filename, bmatrix):
        """
        Run two benchmark commands in parallel.
        """
        command_1 = [self.engine, str(nrhs), filename, bmatrix]
        command_2 = [self.engine, str(nrhs), filename, bmatrix]
        logging.debug(f"Running parallel commands: {' '.join(command_1)} & {' '.join(command_2)}")

        p = subprocess.Popen(command_1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        q = subprocess.Popen(command_2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            out_p, err_p = p.communicate(timeout=self.timeout)
            out_q, err_q = q.communicate(timeout=self.timeout)

            if p.returncode == 0 and q.returncode == 0:
                outLines_1 = out_p.decode('utf-8').split('\n')
                outLines_2 = out_q.decode('utf-8').split('\n')
                time1 = float(outLines_1[-3].split(':')[1])
                time2 = float(outLines_1[-2].split(':')[1])
                time3 = float(outLines_2[-3].split(':')[1])
                time4 = float(outLines_2[-2].split(':')[1])
            else:
                logging.error(f"Error running commands in parallel. Stderr: {err_p.decode('utf-8')}, {err_q.decode('utf-8')}")
                time1 = time2 = time3 = time4 = -1

        except subprocess.TimeoutExpired:
            logging.warning("One of the processes took too long! Terminating...")
            p.terminate()
            q.terminate()
            time1 = time2 = time3 = time4 = -2

        return time1, time2, time3, time4

    def process_matrices(self):
        """
        Process all matrix files in the database folder, run benchmarks, and save results.
        """
        loc = 0
        mtxs = [filename[:-4] for filename in os.listdir(self.database_folder) if filename.endswith('.mtx') and filename.startswith('random_circuit')]

        if not mtxs:
            logging.warning("No .mtx files found in the directory!")
            return

        mtxs = sorted(mtxs, key=self.custom_sort)
        reps = np.ones(len(mtxs), dtype=int) * 1

        logging.info(f"Starting benchmark for {len(mtxs)} matrices")

        for i in tqdm(range(len(mtxs)), desc="Processing Matrices"):
            filepath = os.path.join(self.database_folder, mtxs[i] + '.mtx')
            bmatrix = os.path.join(self.database_folder, mtxs[i], 'vecb.mtx')

            try:
                logging.debug(f"Reading matrix from {filepath}")
                matrix = sio.mmread(filepath)
                nnz = matrix.nnz
                num_rows, _ = matrix.shape
                logging.info(f"Processing matrix {mtxs[i]} with {nnz} non-zero elements and {num_rows} rows")

                total_t1 = total_t2 = total_t3 = total_t4 = 0

                for _ in range(reps[i]):
                    t1, t2, t3, t4 = self.run_double(1, filepath, bmatrix)
                    total_t1 += t1
                    total_t2 += t2
                    total_t3 += t3
                    total_t4 += t4

                t1 = total_t1 / reps[i]
                t2 = total_t2 / reps[i]
                t3 = total_t3 / reps[i]
                t4 = total_t4 / reps[i]

                # Append results to DataFrame
                self.df.loc[loc] = [loc, mtxs[i], 'GLU', 1, nnz, num_rows, t1, t2, t4]
                logging.debug(f"Completed matrix {mtxs[i]}: Analyze time = {t1}, Factorization 1 = {t2}, Factorization 2 = {t4}")

            except Exception as e:
                logging.error(f"Error processing matrix {filepath}: {str(e)}")
                continue

            loc += 1
            logging.debug(f"Memory usage: {psutil.virtual_memory().percent}%")

        logging.info("Benchmark completed for all matrices")
        self.df.to_csv('results_glu_kernel.csv', index=False)
        logging.info("Results saved to results_glu_kernel.csv")

    def custom_sort(self, matrix_name):
        parts = matrix_name.split('_')
        return int(parts[2]), int(parts[3])

# Wrapper function for external usage
def run_benchmark(engine='./glu_kernel', database_folder="/mnt/c/Work/transient-simulation/Random_Circuit_Generator/random_circuit_matrixs", timeout=100, log_level=logging.INFO):
    """
    Run the GLU kernel benchmark with adjustable parameters.
    """
    benchmark = GluKernelBenchmark(engine=engine, database_folder=database_folder, timeout=timeout, log_level=log_level)
    benchmark.process_matrices()

# Example usage of the class in a standalone script
if __name__ == "__main__":
    run_benchmark()
