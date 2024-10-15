import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm  # Progress bar
import psutil  # Memory tracking (optional)

# Set up logging to log both to the console and a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('pardiso_kernel.log'),  # Log to a file
                        logging.StreamHandler()  # Also log to the console
                    ])

databaseFolder = Path("/mnt/c/Work/transient-simulation/Random_Circuit_Generator/random_circuit_matrixs")
engine = './pardiso_kernel.o'
timeout = 100

# Initialize the DataFrame
df = pd.DataFrame(columns=["index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze", "Factorization"])


# Function to run a single command
def runSingle(engine, nrhs, filename, bmatrix, reps, threads=1):
    try:
        command = [engine, str(nrhs), filename, bmatrix, str(reps), str(threads)]
        logging.info(f"Running command: {' '.join(command)}")

        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

        if p.returncode == 0:
            outLines = (p.stdout).decode('utf-8').split('\n')
            logging.info(f"Command output: {outLines}")
            time1 = float(outLines[-3].split(':')[1])
            time2 = float(outLines[-2].split(':')[1])
        else:
            logging.error(f"Error running command: {p.stderr.decode('utf-8')}")
            time1 = -1
            time2 = -1

    except subprocess.TimeoutExpired:
        logging.warning(f"Command timed out: {' '.join(command)}")
        time1 = -2
        time2 = -2

    return time1, time2


# Sorting helper function
def custom_sort(matrix_name):
    parts = matrix_name.split('_')
    return int(parts[2]), int(parts[3])


# Main processing loop with tqdm and logging
def process_matrices():
    loc = 0
    mtxs = []

    # Get list of matrix files
    for filename in os.listdir(databaseFolder):
        if filename.endswith('.mtx') and filename.startswith('random_circuit'):
            mtxs.append(filename[:-4])

    if not mtxs:
        logging.warning("No .mtx files found in the directory!")
        return

    # Sort the matrix files
    mtxs = sorted(mtxs, key=custom_sort)

    # Set repetitions and thread counts
    reps = np.ones(len(mtxs), dtype=int) * 100
    threads = [1, 4, 8, 12, 24, 32]

    # Processing loop with tqdm progress bar
    for j in range(len(threads)):
        for i in tqdm(range(len(mtxs)), desc=f"Processing Matrices with {threads[j]} threads"):
            filepath = os.path.join(databaseFolder, mtxs[i] + '.mtx')
            bmatrix = os.path.join(databaseFolder, mtxs[i], 'vecb.mtx')

            try:
                logging.info(f"Reading matrix from {filepath}")
                matrix = sio.mmread(filepath)
                nnz = matrix.nnz
                num_rows, num_cols = matrix.shape
                logging.info(f"Matrix info: {nnz} non-zero elements, {num_rows} rows")

                # Run the PARDISO kernel with the specified number of threads
                t1, t2 = runSingle(engine, 1, filepath, bmatrix, reps[i], threads=threads[j])

                # Append the results to the DataFrame
                df.loc[loc, 'index'] = loc
                df.loc[loc, 'mtx'] = mtxs[i]
                df.loc[loc, 'algorithmname'] = 'PARDISO'
                df.loc[loc, 'threads'] = threads[j]
                df.loc[loc, 'nnz'] = nnz
                df.loc[loc, 'rows'] = num_rows
                df.loc[loc, 'Analyze'] = t1
                df.loc[loc, 'Factorization'] = t2

                logging.info(f"Finished processing {filepath}: Analyze time = {t1}, Factorization time = {t2}")

            except Exception as e:
                logging.error(f"Error processing matrix {filepath}: {str(e)}")
                continue

            loc += 1

            # Optionally track memory usage
            logging.info(f"Memory usage: {psutil.virtual_memory().percent}%")

    # Save results to CSV
    logging.info("Saving results to results_pardiso_kernel.csv")
    df.to_csv('results_pardiso_kernel.csv', index=False)
    logging.info("Results saved successfully.")
    print("Processing complete!")


# Run the main processing function
if __name__ == "__main__":
    process_matrices()
