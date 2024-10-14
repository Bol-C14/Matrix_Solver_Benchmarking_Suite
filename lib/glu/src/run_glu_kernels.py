import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm  # Import tqdm for the progress bar
import psutil  # For memory tracking

# Set up logging to log both to the console and a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('glu_kernel.log'),  # Log to a file
                        logging.StreamHandler()  # Also log to the console
                    ])

databaseFolder = Path("/mnt/c/Work/transient-simulation/Random_Circuit_Generator/random_circuit_matrixs")
engine = './glu_kernel'
timeout = 100

# Initialize the DataFrame
df = pd.DataFrame(columns=["index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze", "Factorization_1", "Factorization_2"])

# Function to run a single command
def runSingle(engine, nrhs, filename, bmatrix):
    try:
        command = [engine, str(nrhs), filename, bmatrix]
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

# Function to run two commands in parallel
def runDouble(engine, nrhs, filename, bmatrix):
    command_1 = [engine, str(nrhs), filename, bmatrix]
    command_2 = [engine, str(nrhs), filename, bmatrix]
    logging.info(f"Running parallel commands: {' '.join(command_1)} & {' '.join(command_2)}")

    # Start the processes in parallel
    p = subprocess.Popen(command_1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    q = subprocess.Popen(command_2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Wait for the processes to complete
        out_p, err_p = p.communicate(timeout=timeout)
        out_q, err_q = q.communicate(timeout=timeout)

        if p.returncode == 0:
            outLines_1 = out_p.decode('utf-8').split('\n')
            outLines_2 = out_q.decode('utf-8').split('\n')
            logging.info(f"Command 1 output: {outLines_1}")
            logging.info(f"Command 2 output: {outLines_2}")
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

    # Set repetitions for each matrix
    reps = np.ones(len(mtxs), dtype=int) * 20

    # Processing loop with tqdm progress bar
    for i in tqdm(range(len(mtxs)), desc="Processing Matrices"):
        filepath = os.path.join(databaseFolder, mtxs[i] + '.mtx')
        bmatrix = os.path.join(databaseFolder, mtxs[i], 'vecb.mtx')

        try:
            logging.info(f"Reading matrix from {filepath}")
            matrix = sio.mmread(filepath)
            nnz = matrix.nnz
            num_rows, num_cols = matrix.shape
            logging.info(f"Matrix info: {nnz} non-zero elements, {num_rows} rows")

            Total_t1 = Total_t2 = Total_t3 = Total_t4 = 0

            for j in range(reps[i]):
                t1, t2, t3, t4 = runDouble(engine, 1, filepath, bmatrix)
                Total_t1 += t1
                Total_t2 += t2
                Total_t3 += t3
                Total_t4 += t4

            t1 = Total_t1 / reps[i]
            t2 = Total_t2 / reps[i]
            t3 = Total_t3 / reps[i]
            t4 = Total_t4 / reps[i]

            # Append results to DataFrame
            df.loc[loc, 'index'] = loc
            df.loc[loc, 'mtx'] = mtxs[i]
            df.loc[loc, 'algorithmname'] = 'GLU'
            df.loc[loc, 'nnz'] = nnz
            df.loc[loc, 'rows'] = num_rows
            df.loc[loc, 'Analyze'] = t1
            df.loc[loc, 'Factorization_1'] = t2
            df.loc[loc, 'Factorization_2'] = t4

            logging.info(f"Finished processing {filepath}: Analyze time = {t1}, Factorization 1 = {t2}, Factorization 2 = {t4}")

        except Exception as e:
            logging.error(f"Error processing matrix {filepath}: {str(e)}")
            continue

        loc += 1

        # Optionally track memory usage
        logging.info(f"Memory usage: {psutil.virtual_memory().percent}%")

    # Save results to CSV
    logging.info("Saving results to results_glu_kernel.csv")
    df.to_csv('results_glu_kernel.csv', index=False)
    logging.info("Results saved successfully.")
    print("Processing complete!")

# Run the main processing function
if __name__ == "__main__":
    process_matrices()
