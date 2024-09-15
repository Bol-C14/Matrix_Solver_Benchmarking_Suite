import os
import pandas as pd
import subprocess
import scipy.io as sio
import numpy as np
from scipy.sparse import coo_matrix
from pathlib import Path
import logging

# Determine the directory of the script
script_dir = Path(__file__).resolve().parent

# Define the base directory and relative paths
base_dir = script_dir.parents[3]
default_database_folder = base_dir / 'Circuit_Matrix_Analysis' / 'generated_circuits'
default_engine = base_dir / 'Circuit_Matrix_Analysis' / 'LU_Algorithm_Kernels' / 'myGLU' / 'src' / 'glu_kernel'
default_timeout = 100

# Initialize the DataFrame to store results
def initialize_dataframe():
    df = pd.DataFrame(columns=[
        "index", "mtx", "algorithmname", "threads", "nnz", "rows", 
        "Analyze", "Factorization_1", "Factorization_2"
    ])
    df['Analyze'] = df['Analyze'].astype(float)
    df['Factorization_1'] = df['Factorization_1'].astype(float)
    df['Factorization_2'] = df['Factorization_2'].astype(float)
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

def run_single(engine, nrhs, filename, bmatrix, timeout, debug_info):
    """
    Run the external command for single matrix analysis and factorization.

    Parameters:
    - engine: Path to the executable engine.
    - nrhs: Number of right-hand sides.
    - filename: Path to the matrix file.
    - bmatrix: Path to the B matrix file.
    - timeout: Timeout for the subprocess.
    - debug_info: Level of debug information.

    Returns:
    - time1: Analysis time.
    - time2: Factorization time.
    """
    try:
        command = [engine, str(nrhs), filename, bmatrix]
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

def run_double(engine, nrhs, filename, bmatrix, timeout, debug_info):
    """
    Run the external command for double matrix analysis and factorization in parallel.

    Parameters:
    - engine: Path to the executable engine.
    - nrhs: Number of right-hand sides.
    - filename: Path to the matrix file.
    - bmatrix: Path to the B matrix file.
    - timeout: Timeout for the subprocess.
    - debug_info: Level of debug information.

    Returns:
    - time1: First analysis time.
    - time2: First factorization time.
    - time3: Second analysis time.
    - time4: Second factorization time.
    """
    command_1 = [engine, str(nrhs), filename, bmatrix]
    command_2 = [engine, str(nrhs), filename, bmatrix]

    logging.debug(f"Running commands in parallel: {command_1} and {command_2}")

    p = subprocess.Popen(command_1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    q = subprocess.Popen(command_2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        out_p, err_p = p.communicate(timeout=timeout)
        out_q, err_q = q.communicate(timeout=timeout)

        if p.returncode == 0:
            out_lines_1 = out_p.decode('utf-8').split('\n')
            out_lines_2 = out_q.decode('utf-8').split('\n')
            logging.debug(out_lines_1)
            logging.debug(out_lines_2)
            time1 = float(out_lines_1[-3].split(':')[1])
            time2 = float(out_lines_1[-2].split(':')[1])
            time3 = float(out_lines_2[-3].split(':')[1])
            time4 = float(out_lines_2[-2].split(':')[1])
        else:
            logging.error(f"Command failed with return code {p.returncode}: {err_p.decode('utf-8')}")
            time1 = -1
            time2 = -1
            time3 = -1
            time4 = -1
    except subprocess.TimeoutExpired:
        logging.error("One of the processes took too long!")
        p.terminate()
        q.terminate()
        time1 = -2
        time2 = -2
        time3 = -2
        time4 = -2

    return time1, time2, time3, time4

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

def validate_matrix(filepath):
    """
    Validate the matrix indices from the Matrix Market file.

    Parameters:
    - filepath: Path to the Matrix Market file.

    Returns:
    - tuple: (num_rows, num_cols, valid_data)
    """
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

def process_matrices(database_folder, engine, timeout, debug_info):
    """
    Process all matrices in the specified database folder.

    Parameters:
    - database_folder: Path to the folder containing the circuit matrices.
    - engine: Path to the engine executable.
    - timeout: Timeout for the subprocesses.
    - debug_info: Level of debug information.

    Returns:
    - df: DataFrame containing the results of the matrix analysis and factorization.
    """
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
    reps = np.ones(len(mtxs), dtype=int) * 20

    # Process each matrix file
    for i in range(len(mtxs)):
        filename = mtxs[i] + '.mtx'
        filepath = os.path.join(database_folder, filename)

        try:
            num_rows, num_cols, valid_data = validate_matrix(filepath)
            if not valid_data:
                logging.info(f"No valid data found in {filename}. Skipping.")
                continue

            x, y, values = zip(*valid_data)
            matrix = coo_matrix((values, (x, y)), shape=(max(max(x) + 1, num_rows), max(max(y) + 1, num_cols)))
        except Exception as e:
            logging.info(f"Error reading matrix {filename}: {e}")
            continue

        nnz = matrix.nnz
        bmatrix = './' + mtxs[i] + '/vecb.mtx'
        total_t1 = 0
        total_t2 = 0
        total_t3 = 0
        total_t4 = 0

        for j in range(reps[i]):
            t1, t2, t3, t4 = run_double(engine, 1, filepath, bmatrix, timeout, debug_info)
            total_t1 += t1
            total_t2 += t2
            total_t3 += t3
            total_t4 += t4

        t1 = total_t1 / reps[i]
        t2 = total_t2 / reps[i]
        t3 = total_t3 / reps[i]
        t4 = total_t4 / reps[i]
        df.loc[loc] = [loc, mtxs[i], 'GLU', 0, nnz, num_rows, t1, t2, t4]
        logging.debug(f"Stored results for matrix: {mtxs[i]}")
        logging.info(f"GLU Kernel - Processed {loc + 1}/{len(mtxs)} matrices.")
        loc += 1

    return df

def glu_solve(database_folder=default_database_folder, engine=default_engine, timeout=default_timeout, debug_info=1):
    """
    Solve the circuit matrix using the GLU algorithm.

    Parameters:
    - database_folder (str): The path to the folder containing the circuit matrices. Default is `default_database_folder`.
    - engine (str): The engine to use for matrix processing. Default is `default_engine`.
    - timeout (int): The maximum time (in seconds) to wait for matrix processing. Default is `default_timeout`.
    - debug_info (int): The level of debug information to print. 0 = no debug info, 1 = only showing crucial info, 2 = showing all info. Default is 1.

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
            logging.error(f"Have you installed the ")
            return initialize_dataframe()

    df = process_matrices(database_folder, engine, timeout, debug_info)

    # Print the final DataFrame
    if debug_info > 0:
        logging.info("Final DataFrame:")
        logging.info(df)

    # Save results to CSV
    df.to_csv('results_glu_kernel.csv', index=False)
    if debug_info > 0:
        logging.info('done!')

    return df

if __name__ == "__main__":
    glu_solve(debug_info=2)
