import os
import zipfile
import pandas as pd
from utils.result_schema import STANDARD_COLUMNS, normalize_result

def load_and_merge_data(zip_dir, file_groups, source_labels):
    """
    1) For each zip file:
       - Read matrix_features.csv
       - Read each kernel result file:
         * result_glu_kernel.csv
         * result_klu_kernel.csv
         * result_nicslu_kernel.csv
         * result_pardiso_kernel.csv
       - Merge kernel CSV with matrix_features on 'mtx' vs 'Matrix'
       - Append to a master list of dataframes
    2) Concatenate all dataframes from all groups
    3) Return the combined dataframe
    """
    all_data = []

    # Map kernel filenames to how they should be labeled
    kernel_map = {
        'results_glu_kernel.csv': 'glu',
        'results_klu_kernel.csv': 'klu',
        'results_nicslu_kernel.csv': 'nicslu',
        'results_pardiso_kernel.csv': 'pardiso'
    }

    for source_label, zip_file in zip(source_labels, file_groups):
        zip_path = os.path.join(zip_dir, zip_file)
        if not os.path.exists(zip_path):
            print(f"[Warning] {zip_path} does not exist. Skipping.")
            continue

        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Check that matrix_features.csv is present
            if 'matrix_features.csv' not in zf.namelist():
                print(f"[Warning] 'matrix_features.csv' not found in {zip_file}. Skipping.")
                continue

            # Load matrix features
            with zf.open('matrix_features.csv') as mf:
                matrix_features = pd.read_csv(mf)
                # Rename 'Matrix' => 'mtx' for consistent merging
                matrix_features.rename(columns={'Matrix': 'mtx'}, inplace=True)

            # Process each known kernel file
            for filename in kernel_map:
                if filename not in zf.namelist():
                    print(f"[Warning] '{filename}' not found in {zip_file}. Skipping this kernel.")
                    continue

                kernel_name = kernel_map[filename]
                with zf.open(filename) as f:
                    raw_df = pd.read_csv(f)
                    kernel_df = normalize_result(raw_df, kernel_name)
                    kernel_df['source'] = source_label

                    # Merge with matrix features
                    merged_df = pd.merge(
                        kernel_df, 
                        matrix_features, 
                        left_on='mtx', 
                        right_on='mtx', 
                        how='left'
                    )

                    # Append to list
                    all_data.append(merged_df)

    if not all_data:
        raise ValueError("No data loaded. Please verify ZIP files and their contents.")

    # Concatenate everything
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def preprocess_data(combined_data):
    """
    1) Compute effective_time as the min of Factorization_1, Factorization_2, Factorization
    2) Mark NICSLU "failures"
    3) Identify fastest_kernel for each (source, mtx)
    4) Filter invalid rows
    """
    # Ensure columns exist
    for col in ['Factorization_1', 'Factorization_2', 'Factorization']:
        if col not in combined_data.columns:
            combined_data[col] = None

    # Compute effective_time
    combined_data['effective_time'] = combined_data[['Factorization_1','Factorization_2','Factorization']].min(axis=1, skipna=True)

    # Identify NICSLU failures
    combined_data['special_label'] = None
    nicslu_failed = (
        (combined_data['kernel'] == 'nicslu') &
        (combined_data['nnz'] > 1e3) &
        (combined_data['effective_time'] < 100)
    )
    combined_data.loc[nicslu_failed, 'special_label'] = 'nicslu_failed'

    # Determine the fastest kernel, ignoring failed NICSLU
    valid_data = combined_data[combined_data['special_label'] != 'nicslu_failed']
    # Group by (source, mtx), find row with min effective_time
    fastest_idx = valid_data.groupby(['source','mtx'])['effective_time'].idxmin()
    combined_data['fastest_kernel'] = None
    combined_data.loc[fastest_idx, 'fastest_kernel'] = combined_data.loc[fastest_idx, 'kernel']

    # Filter out rows that are missing a fastest_kernel => means factorization was invalid
    processed_data = combined_data.dropna(subset=['fastest_kernel'])
    # Also filter out negative or missing effective_time
    processed_data = processed_data[processed_data['effective_time'] >= 0]

    return processed_data


def append_results(normalized_df: pd.DataFrame, processed_file: str) -> pd.DataFrame:
    """Append new normalized results to processed_file, avoiding duplicates."""
    if os.path.exists(processed_file):
        existing = pd.read_csv(processed_file)
        combined = pd.concat([existing, normalized_df], ignore_index=True)
        combined.drop_duplicates(subset=["mtx", "kernel"], keep="last", inplace=True)
    else:
        combined = normalized_df.copy()
    combined.to_csv(processed_file, index=False)
    return combined

def main():
    # 1) Define the data directory
    zip_dir = os.path.join('..','..','data','ml_data')  # Adjust to your environment

    # 2) Four data groups
    file_groups = [
        'data_group_01_cr.zip',
        'data_group_02_csw.zip',
        'data_group_03_pr.zip',
        'data_group_04_ss.zip'
    ]
    source_labels = ['group1','group2','group3','group4']

    # 3) Load + Merge data
    combined_data = load_and_merge_data(zip_dir, file_groups, source_labels)

    # 4) Preprocess
    processed_data = preprocess_data(combined_data)

    # 5) Save final CSV
    output_dir = os.path.join('..','..','data')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'processed_data.csv')
    processed_data.to_csv(output_csv, index=False)
    print(f"Data processing complete. Saved to '{output_csv}'.")

if __name__ == "__main__":
    main()
