import os
import pandas as pd
import subprocess
import scipy.io as sio
from scipy.sparse import coo_matrix
import numpy as np
from pathlib import Path
import logging

# Determine the directory of the script
script_dir = Path(__file__).resolve().parent

# Define the base directory and relative paths
base_dir = script_dir.parents[3]
default_database_folder = base_dir / 'Circuit_Matrix_Analysis' / 'generated_circuits'
default_engine = base_dir / 'Circuit_Matrix_Analysis' / 'LU_Algorithm_Kernels' / 'myKLU' / 'klu_kernel.o'
default_timeout = 100

# Initialize the DataFrame to store results
def initialize_dataframe():
    df = pd.DataFrame(columns=["index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze", "Factorization"])
    df['Analyze'] = df['Analyze'].astype(float)
    df['Factorization'] = df['Factorization'].astype(float)
    return df

def setup_logging(debug_info):
    """
    Set up logging based on the debug information level.

    Parameters:
    - debug_info: Level of debug information.
    """
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if debug_info == 0:
        logger.setLevel(logging.ERROR)
    elif debug_info == 1:
        logger.setLevel(logging.INFO)
    elif debug_info == 2:
        logger.setLevel(logging.DEBUG)

def run_single(engine, nrhs, filename, bmatrix, reps, threads, timeout, debug_info):
    """
    Run the external command for matrix analysis and factorization.

    Parameters:
    - engine: Path to the executable engine.
    - nrhs: Number of right-hand sides.
    - filename: Path to the matrix file.
    - bmatrix: Path to the B matrix file.
    - reps: Number of repetitions.
    - threads: Number of threads to use.
    - timeout: Timeout for the subprocess.
    - debug_info: Level of debug information.

    Returns:
    - time1: Analysis time.
    - time2: Factorization time.
    """
    try:
        command = [engine, str(nrhs), filename, bmatrix, str(reps), str(threads)]
        logging.debug(f"Running command: {command}")
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if p.returncode == 0:
            out_lines = p.stdout.decode('utf-8').split('\n')
            logging.debug(out_lines)
            time1 = float(out_lines[-3].split(':')[1])
            time2 = float(out_lines[-2].split(':')[1])
        else:
            stderr_output = p.stderr.decode('utf-8')
            if debug_info == 2:
                logging.error(f"Command failed with return code {p.returncode}: {stderr_output}")
            time1 = -1
            time2 = -1
    except subprocess.TimeoutExpired:
        if debug_info > 0:
            logging.error(f"Command timed out: {command}")
        time1 = -2
        time2 = -2

    return time1, time2

def custom_sort(matrix_name):
    """
    Custom sorting function for matrix filenames.

    Parameters:
    - matrix_name: The name of the matrix file.

    Returns:
    - A tuple containing the sorting keys.
    """
    parts = matrix_name.split('_')
    return int(parts[2]), int(parts[3])

def process_matrices(database_folder, engine, timeout, threads, debug_info):
    loc = 0
    mtxs = []
    df = initialize_dataframe()

    # Change directory to the database folder
    logging.debug(f"Changing directory to: {database_folder}")
    os.chdir(database_folder)

    # Collect all matrix filenames
    logging.debug("Collecting matrix filenames...")
    for filename in os.listdir():
        if filename.endswith('.mtx') and filename.startswith('random_circuit'):
            mtxs.append(filename[:-4])
            logging.debug(f"Found matrix file: {filename}")

    # Check if any matrix files were found
    if not mtxs:
        logging.info("No matrix files found. Exiting...")
        return df

    # Sort the list using the custom sort function
    mtxs = sorted(mtxs, key=custom_sort)
    logging.debug(f"Sorted matrix files: {mtxs}")

    # Initialize repetitions array
    reps = np.ones(len(mtxs), dtype=int) * 100

    # Process each matrix file
    for j in range(len(threads)):
        for i, mtx in enumerate(mtxs):
            filepath = mtx + '.mtx'
            bmatrix = './' + mtx + '/vecb.mtx'

            if not os.path.exists(filepath):
                logging.info(f"File not found: {filepath}")
                continue

            try:
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    header = lines[0].strip()
                    if not header.startswith('%%MatrixMarket'):
                        logging.info(f"Invalid Matrix Market file header: {filepath}")
                        continue

                    num_rows, num_cols, _ = map(int, lines[1].strip().split())
                    valid_data = []
                    for line in lines[2:]:
                        if line.strip():
                            x, y, value = line.strip().split()
                            x, y = int(x), int(y)  # 1-based index
                            valid_data.append((x-1, y-1, float(value)))  # Convert to 0-based index

                if not valid_data:
                    logging.info(f"No valid data found in {filename}. Skipping.")
                    continue

                x, y, values = zip(*valid_data)
                matrix = coo_matrix((values, (x, y)), shape=(max(max(x) + 1, num_rows), max(max(y) + 1, num_cols)))
                nnz = matrix.nnz
            except Exception as e:
                logging.info(f"Error reading matrix {filepath}: {e}")
                continue

            t1, t2 = run_single(engine, 1, filepath, bmatrix, reps[i], threads[j], timeout, debug_info)

            df.loc[loc] = [loc, mtx, 'NICSLU', threads[j], nnz, num_rows, t1, t2]
            logging.debug(f"Stored results for matrix: {mtx}")
            logging.info(f"NICSLU - Processed {loc + 1}/{len(mtxs) * len(threads)} matrices.")
            loc += 1

    return df

def nicslu_solve(database_folder=default_database_folder, engine=default_engine, timeout=default_timeout, debug_info=1, threads=[1, 4, 8, 12, 24, 32]):
    """
    Solve the circuit matrix using the NICSLU algorithm.

    Parameters:
    - database_folder (str): The path to the folder containing the circuit matrices. Default is `default_database_folder`.
    - engine (str): The engine to use for matrix processing. Default is `default_engine`.
    - timeout (int): The maximum time (in seconds) to wait for matrix processing. Default is `default_timeout`.
    - debug_info (int): The level of debug information to print. 0 = no debug info, 1 = only showing crucial info, 2 = showing all info. Default is 1.
    - threads (list of int): List of thread counts to use. Default is [1, 4, 8, 12, 24, 32].

    Returns:
    - df (pandas.DataFrame): The final DataFrame containing the results of the circuit matrix analysis.
    """
    setup_logging(debug_info)

    # Check if the engine file exists; if not, run 'make' to compile it
    if not os.path.isfile(engine):
        logging.info(f"{engine} not found. Attempting to compile...")
        try:
            subprocess.run(["make"], check=True)
            logging.info("Compilation successful.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Compilation failed: {e}")
            return initialize_dataframe()
    
    else :
        logging.info(f"{engine} found. Proceeding...")

    df = process_matrices(database_folder, engine, timeout, threads, debug_info)

    # Print the final DataFrame
    if debug_info > 0:
        logging.info("Final DataFrame:")
        logging.info(df)

    # Save results to CSV
    df.to_csv('results_nicslu_kernel.csv', index=False)
    if debug_info > 0:
        logging.info('done!')

    return df

if __name__ == "__main__":
    nicslu_solve()
