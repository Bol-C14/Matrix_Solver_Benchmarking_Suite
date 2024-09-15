import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
from scipy.sparse import coo_matrix
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading  # Import threading to log thread information

# Determine the directory of the script
script_dir = Path(__file__).resolve().parent

# Define the base directory and relative paths
base_dir = script_dir.parents[2]
default_database_folder = base_dir / 'Circuit_Matrix_Analysis' / 'generated_circuits'
default_engine = base_dir / 'Circuit_Matrix_Analysis' / 'LU_Algorithm_Kernels' / 'myPARDISO' / 'pardiso_kernel.o'
default_timeout = 100
default_parallel_jobs = 4  # Default to 4 parallel jobs

# Initialize the DataFrame to store results
def initialize_dataframe():
    df = pd.DataFrame(columns=[
        "index", "mtx", "algorithmname", "threads", "nnz", "rows", 
        "Analyze", "Factorization"
    ])
    df['Analyze'] = df['Analyze'].astype(float)
    df['Factorization'] = df['Factorization'].astype(float)
    return df

def setup_logging(debug_info):
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
    parts = matrix_name.split('_')
    return int(parts[2]), int(parts[3])

def validate_matrix(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip()
        if not header.startswith('%%MatrixMarket'):
            raise ValueError('Invalid Matrix Market file header')

        num_rows, num_cols, _ = map(int, lines[1].strip().split())
        valid_data = []
        for line in lines[2:]:
            if line.strip():
                x, y, value = line.strip().split()
                x, y = int(x) - 1, int(y) - 1  # Convert to zero-based index
                valid_data.append((x, y, float(value)))

    return num_rows, num_cols, valid_data

def process_single_matrix(mtx, engine, reps, threads, timeout, debug_info):
    """
    Function to process a single matrix file. Used for parallel processing.
    Logs which thread is solving the matrix.
    """
    loc = mtx['loc']
    mtx_name = mtx['name']
    filepath = mtx['filepath']
    thread_name = threading.current_thread().name  # Get current thread name

    # Log which thread is solving which matrix
    logging.info(f"Thread {thread_name} is solving matrix {mtx_name}")

    try:
        num_rows, num_cols, valid_data = validate_matrix(filepath)
        if not valid_data:
            logging.info(f"No valid data found in {mtx_name}. Skipping.")
            return None

        x, y, values = zip(*valid_data)
        matrix = coo_matrix((values, (x, y)), shape=(max(max(x) + 1, num_rows), max(max(y) + 1, num_cols)))
        nnz = matrix.nnz
        num_rows, num_cols = matrix.shape
        bmatrix = './' + mtx_name + '/vecb.mtx'
        t1, t2 = run_single(engine, 1, filepath, bmatrix, reps, threads, timeout, debug_info)
        logging.debug(f"Thread {thread_name} stored results for matrix: {mtx_name}")
        return (loc, mtx_name, 'PARDISO', threads, nnz, num_rows, t1, t2)
    except Exception as e:
        logging.info(f"Error processing matrix {filepath} by thread {thread_name}: {e}")
        return None


def process_matrices(database_folder, engine, timeout, threads, debug_info, parallel_jobs):
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

    # Prepare data for parallel processing
    matrix_data = []
    for i, mtx in enumerate(mtxs):
        filepath = mtx + '.mtx'
        matrix_data.append({'loc': loc + i, 'name': mtx, 'filepath': filepath})

    # Process each matrix exactly once in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = [executor.submit(process_single_matrix, mtx, engine, reps[i], threads[0], timeout, debug_info)
                   for i, mtx in enumerate(matrix_data)]  # Ensure each matrix is submitted only once
                   
        for future in as_completed(futures):
            result = future.result()
            if result:
                df.loc[len(df)] = result
                logging.info(f"PARDISO - Processed {len(df)}/{len(matrix_data)} matrices.")

    return df


def pardiso_solve(database_folder=default_database_folder, engine=default_engine, timeout=default_timeout, debug_info=1, threads=[1, 4, 8, 12, 24, 32], parallel_jobs=default_parallel_jobs):
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

    df = process_matrices(database_folder, engine, timeout, threads, debug_info, parallel_jobs)

    # Print the final DataFrame
    if debug_info > 0:
        logging.info("Final DataFrame:")
        logging.info(df)

    # Save results to CSV
    df.to_csv('results_pardiso_kernel.csv', index=False)
    if debug_info > 0:
        logging.info('done!')

    return df

if __name__ == "__main__":
    pardiso_solve()
