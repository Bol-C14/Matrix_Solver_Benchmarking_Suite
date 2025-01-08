import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from pathlib import Path
import logging
from scipy.sparse.linalg import eigs, svds, norm
from scipy.sparse import csr_matrix
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging to log both to the console and a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('matrix_features.log'),  # Log to a file
                        logging.StreamHandler()  # Also log to the console
                    ])

# Input and output paths
databaseFolder = Path("./data/circuit_random/matrices")  # Path to the folder containing the .mtx files
outputFolder = Path("./output")
outputFolder.mkdir(parents=True, exist_ok=True)

# Initialize the DataFrame with additional columns
df = pd.DataFrame(columns=[
    "Matrix", "Locality", "Sparsity", "Ncol", "Nrow", "NNZ", "NNZL", "NNZU",
    "NNZmax", "NNZavg", "DEmax", "DEmin", "bwL", "bwU", "NO", "NDDrow",
    "OneNorm", "FroNorm", "SpectralRadius", "SecondLargestEigenvalue",
    "SmallestNonZeroEigenvalue", "Lambda1_Lambda2", "Peak",
    "ConditionNumber", "SqrtConditionNumber",
    "s90", "s10", "smid",
    "Ndim", "Meshdo", "Meshrl",
    "Kappa_hmin", "Kappa_hmax",
    "sp_all", "n2_structural", "sp_nz", "n2_numerical",
    "psym", "nnz_structural_symmetry", "nsym_numerical_symmetry",
    "Bandwidth", "BandWidthTotal", "CoefficientOfVariation"
])

# Define functions for computing the required features

def read_mtx_file(file_path):
    '''
    This function reads a sparse matrix from a file and returns a CSR matrix.

    @param file_path: the path to the file containing the matrix
    @return: a CSR sparse matrix
    '''
    # Load the sparse matrix from the file_path
    sparse_mtx = mmread(file_path)

    # Convert the sparse matrix to compressed sparse row (CSR) format
    csr_matrix_ = sparse_mtx.tocsr()
    return csr_matrix_

def compute_locality(matrix):
    coo = matrix.tocoo()
    row_diffs = np.diff(coo.row)
    col_diffs = np.diff(coo.col)
    logging.info(f"Locality - row_diffs shape: {row_diffs.shape}, col_diffs shape: {col_diffs.shape}")
    locality = np.mean(np.abs(row_diffs) + np.abs(col_diffs))
    return locality

def compute_sparsity(matrix):
    nnz = matrix.nnz
    total_elements = matrix.shape[0] * matrix.shape[1]
    logging.info(f"Sparsity - NNZ: {nnz}, total elements: {total_elements}")
    sparsity = (total_elements - nnz) / total_elements
    return sparsity

def compute_bandwidth(matrix):
    coo = matrix.tocoo()
    diag_diffs = coo.row - coo.col
    logging.info(f"Bandwidth - diag_diffs shape: {diag_diffs.shape}")
    bwL = np.abs(np.min(diag_diffs))  # Lower triangle bandwidth
    bwU = np.abs(np.max(diag_diffs))  # Upper triangle bandwidth
    return bwL, bwU

def compute_nnz_triangle(matrix):
    coo = matrix.tocoo()
    nnzL = np.sum(coo.row > coo.col)
    nnzU = np.sum(coo.row < coo.col)
    logging.info(f"NNZ Triangle - NNZL: {nnzL}, NNZU: {nnzU}")
    return nnzL, nnzU

def compute_nnz_stats(matrix):
    nnz_per_row = np.diff(matrix.indptr)
    logging.info(f"NNZ Stats - nnz_per_row shape: {nnz_per_row.shape}")
    nnz_max = np.max(nnz_per_row)
    nnz_avg = np.mean(nnz_per_row)
    return nnz_max, nnz_avg

def compute_diagonal_extremes(matrix):
    diag_elements = matrix.diagonal()
    logging.info(f"Diagonal Extremes - diag_elements shape: {diag_elements.shape}")
    de_max = np.max(np.abs(diag_elements))
    non_zero_diag = diag_elements[np.nonzero(diag_elements)]
    if non_zero_diag.size == 0:
        de_min = 0
        logging.warning("No non-zero diagonal elements found.")
    else:
        de_min = np.min(np.abs(non_zero_diag))  # Exclude zeros
    return de_max, de_min

def compute_diagonally_dominant_rows(matrix):
    abs_matrix = np.abs(matrix)
    diag_abs = np.abs(matrix.diagonal())

    # Sum all off-diagonal elements in each row by subtracting diagonal from row sum
    sum_off_diag = np.array(abs_matrix.sum(axis=1)).flatten() - diag_abs
    logging.info(f"Diagonal Dominance - diag_abs shape: {diag_abs.shape}, sum_off_diag shape: {sum_off_diag.shape}")

    # Count rows that are strictly diagonally dominant
    ndd_row = np.sum(2 * diag_abs > sum_off_diag)
    return ndd_row

def count_ones(matrix):
    ones_count = np.sum(matrix.data == 1)
    logging.info(f"Number of ones: {ones_count}")
    return ones_count

# Additional Feature Computation Functions

def compute_one_norm(matrix):
    one_norm = norm(matrix, 1)
    logging.info(f"1-Norm: {one_norm}")
    return one_norm

def compute_frobenius_norm(matrix):
    fro_norm = norm(matrix, 'fro')
    logging.info(f"Frobenius Norm: {fro_norm}")
    return fro_norm

def compute_spectral_radius(matrix, k=1):
    try:
        eigenvalues, _ = eigs(matrix, k=k, which='LR')  # Largest magnitude
        spectral_radius = max(abs(eigenvalues))
        logging.info(f"Spectral Radius: {spectral_radius}")
        return spectral_radius, eigenvalues
    except Exception as e:
        logging.error(f"Error computing spectral radius: {e}")
        return 0, np.array([])

def compute_second_largest_eigenvalue(matrix, spectral_radius, eigenvalues):
    try:
        if eigenvalues.size < 2:
            # Compute more eigenvalues if needed
            eigenvalues, _ = eigs(matrix, k=2, which='LR')
        eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
        if len(eigenvalues_sorted) >= 2:
            second_largest = eigenvalues_sorted[1]
            logging.info(f"Second Largest Eigenvalue: {second_largest}")
            return second_largest
        else:
            logging.warning("Not enough eigenvalues to compute the second largest.")
            return 0
    except Exception as e:
        logging.error(f"Error computing second largest eigenvalue: {e}")
        return 0

def compute_smallest_nonzero_eigenvalue(matrix, k=10):
    try:
        # Avoid requesting nearly all eigenvalues
        k = min(k, matrix.shape[0] - 2)
        eigenvalues, _ = eigs(matrix, k=k, which='SM')
        eigenvalues = np.abs(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if (eigenvalues.size == 0):
            logging.warning("No non-zero eigenvalues found.")
            return 0
        smallest_nonzero = np.min(eigenvalues)
        return smallest_nonzero
    except Exception as e:
        logging.error(f"Error computing smallest non-zero eigenvalue: {e}")
        return 0

def compute_lambda1_lambda2(spectral_radius, second_largest):
    if second_largest == 0:
        logging.warning("Second largest eigenvalue is zero. Cannot compute Lambda1/Lambda2.")
        return 0
    ratio = spectral_radius / second_largest
    logging.info(f"Lambda1/Lambda2: {ratio}")
    return ratio

def compute_peak(lambda1, P=1):  # Assuming P=1 if not defined
    if P == 0:
        logging.warning("Parameter P is zero. Cannot compute peak.")
        return 0
    peak = lambda1 / P
    logging.info(f"Peak: {peak}")
    return peak

def compute_condition_number(matrix):
    try:
        # Only singular values are returned
        s = svds(matrix, k=2, return_singular_vectors=False)
        sigma_max = np.max(s)
        sigma_min = np.min(s)
        if sigma_min == 0:
            kappa = np.inf
            logging.warning("Smallest singular value is zero. Condition number is infinity.")
        else:
            kappa = sigma_max / sigma_min
        return kappa
    except Exception as e:
        logging.error(f"Error computing condition number: {e}")
        return np.inf

def compute_sqrt_condition_number(kappa):
    if kappa == np.inf:
        sqrt_kappa = np.inf
    else:
        sqrt_kappa = np.sqrt(kappa)
    logging.info(f"Sqrt Condition Number: {sqrt_kappa}")
    return sqrt_kappa

def timeout_handler(signum, frame):
    raise TimeoutException

class TimeoutException(Exception):
    pass

def compute_singular_value_statistics(matrix):
    try:
        # Set the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # Set timeout to 10 seconds

        # Compute all singular values
        s = svds(matrix, k=min(matrix.shape)-1, return_singular_vectors=False)
        sqrt_s = np.sqrt(s)
        sqrt_s_max = np.max(s)
        sqrt_s_min = np.min(s)
        threshold_90 = 0.9 * (sqrt_s_max - sqrt_s_min) + sqrt_s_min
        threshold_10 = 0.1 * (sqrt_s_max - sqrt_s_min) + sqrt_s_min

        s90 = np.sum(sqrt_s > threshold_90) / len(sqrt_s)
        s10 = np.sum(sqrt_s <= threshold_10) / len(sqrt_s)
        smid = 1 - (s90 + s10)

        logging.info(f"s90: {s90}, s10: {s10}, smid: {smid}")

        # Disable the alarm
        signal.alarm(0)

        return s90, s10, smid
    except TimeoutException:
        logging.error("compute_singular_value_statistics timed out after 10 seconds.")
        return 0, 0, 1
    except Exception as e:
        logging.error(f"Error computing singular value statistics: {e}")
        return 0, 0, 1

def compute_kappa_approx(Nrow, Ndim):
    try:
        kappa = (Nrow ** 2) / Ndim if Ndim != 0 else np.inf
        sqrt_kappa = np.sqrt(kappa)
        logging.info(f"Approximate Kappa: {kappa}, Sqrt Kappa: {sqrt_kappa}")
        return sqrt_kappa
    except Exception as e:
        logging.error(f"Error computing approximate kappa: {e}")
        return np.inf

def compute_kappa_hmin_hmax(h_values):
    try:
        min_h = np.min(h_values)
        max_h = np.max(h_values)
        kappa_hmin = (1 / min_h) ** 2
        kappa_hmax = (1 / max_h) ** 2
        logging.info(f"kappa_hmin: {kappa_hmin}, kappa_hmax: {kappa_hmax}")
        return kappa_hmin, kappa_hmax
    except Exception as e:
        logging.error(f"Error computing kappa_hmin and kappa_hmax: {e}")
        return np.inf, np.inf

def compute_coefficient_of_variation(matrix):
    try:
        # Compute coefficient of variation for rows and columns
        row_means = matrix.mean(axis=1).A1
        row_std = matrix.std(axis=1).A1
        cv_rows = row_std / row_means
        col_means = matrix.mean(axis=0).A1
        col_std = matrix.std(axis=0).A1
        cv_cols = col_std / col_means
        cv = np.mean(cv_rows) + np.mean(cv_cols)
        logging.info(f"Coefficient of Variation: {cv}")
        return cv
    except Exception as e:
        logging.error(f"Error computing coefficient of variation: {e}")
        return 0

def process_single_file(filepath):
    filename = filepath.name
    try:
        logging.info(f"Processing {filename}...")
        matrix = read_mtx_file(filepath)
        logging.info(f"Matrix shape: {matrix.shape}, NNZ: {matrix.nnz}")

        if matrix.shape[0] != matrix.shape[1]:
            logging.error(f"Matrix {filename} is not square. Skipping.")
            return None

        nrow, ncol = matrix.shape
        nnz = matrix.nnz

        # Compute features with detailed logging
        locality = compute_locality(matrix)
        sparsity = compute_sparsity(matrix)
        nnzl, nnzU = compute_nnz_triangle(matrix)
        nnz_max, nnz_avg = compute_nnz_stats(matrix)
        de_max, de_min = compute_diagonal_extremes(matrix)
        bwl, bwu = compute_bandwidth(matrix)
        no = count_ones(matrix)
        ndd_row = compute_diagonally_dominant_rows(matrix)
        one_norm = compute_one_norm(matrix)
        fro_norm = compute_frobenius_norm(matrix)
        spectral_radius, eigenvalues_lr = compute_spectral_radius(matrix, k=2)
        second_largest = compute_second_largest_eigenvalue(matrix, spectral_radius, eigenvalues_lr)
        smallest_nonzero_eigenvalue = compute_smallest_nonzero_eigenvalue(matrix)
        lambda1_lambda2 = compute_lambda1_lambda2(spectral_radius, second_largest)
        peak = compute_peak(spectral_radius, P=1)
        condition_number = compute_condition_number(matrix)
        sqrt_condition_number = compute_sqrt_condition_number(condition_number)

        # Example placeholders
        Ndim = 3
        Meshdo = 1
        Meshrl = 1
        h_values = np.random.rand(10) + 0.1
        kappa_hmin, kappa_hmax = compute_kappa_hmin_hmax(h_values)
        sp_all = nnz
        n2_structural = 0
        sp_nz = nnz
        n2_numerical = 0
        psym = 0
        nnz_structural_symmetry = 0
        nsym_numerical_symmetry = 0
        bandwidth_total = bwl + bwu
        coefficient_of_variation = compute_coefficient_of_variation(matrix)

        return {
            "Matrix": filename,
            "Locality": locality,
            "Sparsity": sparsity,
            "Ncol": ncol,
            "Nrow": nrow,
            "NNZ": nnz,
            "NNZL": nnzl,
            "NNZU": nnzU,
            "NNZmax": nnz_max,
            "NNZavg": nnz_avg,
            "DEmax": de_max,
            "DEmin": de_min,
            "bwL": bwl,
            "bwU": bwu,
            "NO": no,
            "NDDrow": ndd_row,
            "OneNorm": one_norm,
            "FroNorm": fro_norm,
            "SpectralRadius": spectral_radius,
            "SecondLargestEigenvalue": second_largest,
            "SmallestNonZeroEigenvalue": smallest_nonzero_eigenvalue,
            "Lambda1_Lambda2": lambda1_lambda2,
            "Peak": peak,
            "ConditionNumber": condition_number,
            "SqrtConditionNumber": sqrt_condition_number,
            # "s90": s90,
            # "s10": s10,
            # "smid": smid,
            "Ndim": Ndim,
            "Meshdo": Meshdo,
            "Meshrl": Meshrl,
            "Kappa_hmin": kappa_hmin,
            "Kappa_hmax": kappa_hmax,
            "sp_all": sp_all,
            "n2_structural": n2_structural,
            "sp_nz": sp_nz,
            "n2_numerical": n2_numerical,
            "psym": psym,
            "nnz_structural_symmetry": nnz_structural_symmetry,
            "nsym_numerical_symmetry": nsym_numerical_symmetry,
            "Bandwidth": bwl,
            "BandWidthTotal": bandwidth_total,
            "CoefficientOfVariation": coefficient_of_variation
        }
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return None

if __name__ == "__main__":
    rows = []
    mt_files = [f for f in databaseFolder.iterdir() if f.suffix == '.mtx']
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, f): f for f in mt_files}

        for future in as_completed(future_to_file):
            result = future.result()
            if result is not None:
                rows.append(result)

    df = pd.DataFrame(rows)
    output_csv = outputFolder / "matrix_features.csv"
    df.to_csv(output_csv, index=False)
    logging.info(f"Results saved to {output_csv}")
