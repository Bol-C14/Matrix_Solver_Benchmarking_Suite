import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from pathlib import Path
import logging

# Set up logging to log both to the console and a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('matrix_features.log'),  # Log to a file
                        logging.StreamHandler()  # Also log to the console
                    ])

# Input and output paths
databaseFolder = Path("/home/gushu/work/MLtask/ML_Circuit_Matrix_Analysis/data/circuit_data")
outputFolder = Path("./output")
outputFolder.mkdir(parents=True, exist_ok=True)

# Initialize the DataFrame
df = pd.DataFrame(columns=["Matrix", "Locality", "Sparsity", "Ncol", "Nrow", "NNZ", "NNZL", "NNZU",
                           "NNZmax", "NNZavg", "DEmax", "DEmin", "bwL", "bwU", "NO", "NDDrow"])


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
    csr_matrix = sparse_mtx.tocsr()
    return csr_matrix


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
    de_min = np.min(np.abs(diag_elements[np.nonzero(diag_elements)]))  # Exclude zeros
    return de_max, de_min


def compute_diagonally_dominant_rows(matrix):
    abs_matrix = np.abs(matrix)
    diag_abs = np.abs(matrix.diagonal())

    # Sum all off-diagonal elements in each row by subtracting diagonal from row sum
    sum_off_diag = np.sum(abs_matrix, axis=1) - diag_abs
    logging.info(f"Diagonal Dominance - diag_abs shape: {diag_abs.shape}, sum_off_diag shape: {sum_off_diag.shape}")

    # Count rows that are strictly diagonally dominant
    ndd_row = np.sum(2 * diag_abs > sum_off_diag)

    return ndd_row


def count_ones(matrix):
    ones_count = np.sum(matrix.data == 1)
    logging.info(f"Number of ones: {ones_count}")
    return ones_count

# Initialize the list to collect rows for the DataFrame
rows = []

# Iterate through each .mtx file in the directory
for filename in os.listdir(databaseFolder):
    if filename.endswith('.mtx'):
        filepath = databaseFolder / filename

        try:
            logging.info(f"Processing {filename}...")

            # Load the matrix and log its shape
            try:
                matrix = read_mtx_file(filepath)
                logging.info(f"Matrix shape: {matrix.shape}, NNZ: {matrix.nnz}")
            except ValueError as e:
                logging.error(f"Error reading matrix {filename}: {str(e)}")
                continue  # Skip matrices with read errors

            # Check for shape mismatch
            if matrix.shape[0] != matrix.shape[1]:
                logging.error(f"Matrix {filename} is not square. Skipping.")
                continue

            # Matrix properties
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

            # Add the result as a dictionary to the list `rows`
            rows.append({
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
                "NDDrow": ndd_row
            })

        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            continue

# Convert the list of rows to a DataFrame
df = pd.DataFrame(rows)

# Save the DataFrame to a CSV file
output_csv = outputFolder / "matrix_features.csv"
df.to_csv(output_csv, index=False)
logging.info(f"Results saved to {output_csv}")
