import os
import pandas as pd
import subprocess
import scipy.io as sio
from scipy.sparse import coo_matrix
import numpy as np
from pathlib import Path
import logging


class CircuitMatrixAnalyzer:
    def __init__(self, database_folder, engine, timeout, debug_info=1, boundary_option=1):
        """
        Initializes the CircuitMatrixAnalyzer with given configurations.

        Parameters:
        - database_folder (str): Path to the folder containing the matrices.
        - engine (str): Path to the executable engine for matrix analysis.
        - timeout (int): Timeout for the subprocess in seconds.
        - debug_info (int): Level of debug logging (0=ERROR, 1=INFO, 2=DEBUG).
        - boundary_option (int): Boundary handling option (1=strict, 2=ignore limit, 3=ignore out-of-bound items).
        """
        self.database_folder = Path(database_folder)
        self.engine = engine
        self.timeout = timeout
        self.debug_info = debug_info
        self.boundary_option = boundary_option
        self._setup_logging()

        if self.boundary_option == 2 or self.boundary_option == 3:
            logging.warning(f"Boundary option {self.boundary_option} selected. This may affect matrix data integrity.")

    def _setup_logging(self):
        """
        Set up logging based on the debug information level.
        """
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if self.debug_info == 0:
            logger.setLevel(logging.ERROR)
        elif self.debug_info == 1:
            logger.setLevel(logging.INFO)
        elif self.debug_info == 2:
            logger.setLevel(logging.DEBUG)

    def _initialize_dataframe(self):
        """
        Initialize the pandas DataFrame to store the results.

        Returns:
        - df (pandas.DataFrame): Empty DataFrame with predefined columns.
        """
        df = pd.DataFrame(
            columns=["index", "mtx", "algorithmname", "threads", "nnz", "rows", "Analyze", "Factorization"])
        df['Analyze'] = df['Analyze'].astype(float)
        df['Factorization'] = df['Factorization'].astype(float)
        return df

    def _run_single(self, nrhs, filename, bmatrix, reps):
        """
        Runs the external command for matrix analysis and factorization.
        """
        # Ensure all elements in the command are strings
        command = [str(self.engine), str(nrhs), str(filename), str(bmatrix), str(reps)]

        logging.debug(f"Running command: {command}")

        try:
            # Log the command being executed
            logging.error(f"Command to be executed: {' '.join(command)}")

            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout)
            if p.returncode == 0:
                out_lines = p.stdout.decode('utf-8').split('\n')
                time1 = float(out_lines[-3].split(':')[1])
                time2 = float(out_lines[-2].split(':')[1])
            else:
                logging.error(f"Command failed: {p.stderr.decode('utf-8')}")
                time1, time2 = -1, -1
        except subprocess.TimeoutExpired:
            logging.error(f"Command timed out: {command}")
            time1, time2 = -2, -2

        return time1, time2

    def _custom_sort(self, matrix_name):
        """
        Custom sorting function for matrix filenames.

        Parameters:
        - matrix_name (str): The name of the matrix file.

        Returns:
        - A tuple containing sorting keys for the matrix file.
        """
        parts = matrix_name.split('_')
        return int(parts[2]), int(parts[3])

    def process_matrices(self):
        """
        Process the matrices in the database folder using the KLU engine.

        Returns:
        - df (pandas.DataFrame): DataFrame containing the analysis and factorization results.
        """
        loc = 0
        mtxs = []
        df = self._initialize_dataframe()

        os.chdir(self.database_folder)
        logging.debug(f"Changed directory to: {self.database_folder}")

        for filename in os.listdir():
            if filename.endswith('.mtx') and filename.startswith('random_circuit'):
                mtxs.append(filename[:-4])

        if not mtxs:
            logging.info("No matrix files found. Exiting...")
            return df

        mtxs = sorted(mtxs, key=self._custom_sort)

        reps = np.ones(len(mtxs), dtype=int) * 100

        for i, mtx in enumerate(mtxs):
            filepath = self.database_folder / f"{mtx}.mtx"
            bmatrix = self.database_folder / f"{mtx}/vecb.mtx"

            if not filepath.exists():
                logging.info(f"File not found: {filepath}")
                continue

            try:
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    num_rows, num_cols = map(int, lines[1].strip().split()[:2])

                    valid_data = []
                    max_row, max_col = num_rows, num_cols

                    for line in lines[2:]:
                        if line.strip():
                            x, y, value = line.strip().split()
                            x, y = int(x) - 1, int(y) - 1  # Convert 1-based to 0-based index

                            if self.boundary_option == 1:
                                # Strict limit: Error if index exceeds boundaries
                                if x >= num_rows or y >= num_cols:
                                    logging.error(f"Entry ({x+1}, {y+1}) exceeds matrix dimensions ({num_rows}, {num_cols}) in {filename}. Skipping.")
                                    raise ValueError(f"Entry ({x+1}, {y+1}) exceeds matrix dimensions ({num_rows}, {num_cols})")
                            elif self.boundary_option == 2:
                                # Ignore boundary limit: Allow indices to exceed dimensions
                                max_row = max(max_row, x + 1)
                                max_col = max(max_col, y + 1)
                            elif self.boundary_option == 3:
                                # Ignore out-of-bound items: Skip invalid entries
                                if x >= num_rows or y >= num_cols:
                                    logging.warning(f"Entry ({x+1}, {y+1}) exceeds matrix dimensions ({num_rows}, {num_cols}) in {filename}. Skipping.")
                                    continue

                            valid_data.append((x, y, float(value)))

                    if not valid_data:
                        logging.info(f"No valid data found in {filename}. Skipping.")
                        continue

                    # Create the sparse matrix with dynamically determined size for option 2
                    matrix_shape = (max_row, max_col) if self.boundary_option == 2 else (num_rows, num_cols)
                    x, y, values = zip(*valid_data)
                    matrix = coo_matrix((values, (x, y)), shape=matrix_shape)
                    nnz = matrix.nnz

            except Exception as e:
                logging.error(f"Error processing file {filepath}: {e}")
                logging.error(f"If this error is caused by out-of-index, you may use default_boundary_option parameter to force loading matrices.")
                continue

            t1, t2 = self._run_single(1, str(filepath), str(bmatrix), reps[i])
            df.loc[loc] = [loc, mtx, 'KLU', 1, nnz, num_rows, t1, t2]
            loc += 1

        return df


def klu_solve(database_folder, engine, timeout, debug_info=1, boundary_option=1):
    """
    Function to be called externally to perform KLU analysis on matrices.

    Parameters:
    - database_folder (str): Folder path for the matrices.
    - engine (str): Path to the engine for analysis.
    - timeout (int): Maximum time allowed for the analysis.
    - debug_info (int): Debug information level.
    - boundary_option (int): Boundary handling option (1=strict, 2=ignore limit, 3=ignore out-of-bound items).

    Returns:
    - df (pandas.DataFrame): DataFrame with the analysis results.
    """
    analyzer = CircuitMatrixAnalyzer(database_folder, engine, timeout, debug_info, boundary_option)
    df = analyzer.process_matrices()

    df.to_csv('results_klu_kernel.csv', index=False)
    return df


if __name__ == "__main__":
    # Default paths and engine configurations, can be changed for the script execution
    default_database_folder = Path(__file__).resolve().parents[3] / 'random_circuit_generator' / 'generated_circuits'
    default_engine = Path(__file__).resolve().parents[1] / 'myKLU' / 'klu_kernel.o'
    default_timeout = 100
    default_debug_info = 1
    default_boundary_option = 3  # 1: strict, 2: ignore limit, 3: ignore out-of-bound items

    # Call the function with default values or custom ones
    klu_solve(default_database_folder, default_engine, default_timeout, default_debug_info, default_boundary_option)
