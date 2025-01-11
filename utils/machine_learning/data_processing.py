import pandas as pd
import os

def load_and_merge_data(files, source_labels):
    dataframes = []
    for source, file_group in zip(source_labels, files):
        for kernel, path in file_group.items():
            df = pd.read_csv(path)
            df['kernel'] = kernel
            df['source'] = source
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def preprocess_data(combined_data):
    # Calculate the effective time (factorization only)
    combined_data['effective_time'] = combined_data[['Factorization_1', 'Factorization_2', 'Factorization']].min(axis=1, skipna=True)

    # Identify NICSLU failures
    combined_data['special_label'] = None
    combined_data.loc[
        (combined_data['kernel'] == 'nicslu') & 
        (combined_data['nnz'] > 1e3) & 
        (combined_data['effective_time'] < 100),
        'special_label'
    ] = 'nicslu_failed'

    # Determine the fastest kernel, ignoring failed NICSLU cases
    valid_cases = combined_data[combined_data['special_label'] != 'nicslu_failed']
    fastest_kernel_indices = valid_cases.groupby(['source', 'mtx'])['effective_time'].idxmin()
    combined_data['fastest_kernel'] = None
    combined_data.loc[fastest_kernel_indices, 'fastest_kernel'] = combined_data.loc[fastest_kernel_indices, 'kernel']

    # Filter out invalid rows (e.g., failed factorization)
    combined_data = combined_data.dropna(subset=['fastest_kernel'])
    combined_data = combined_data[combined_data['effective_time'] >= 0]

    return combined_data

if __name__ == "__main__":
    # Define your file paths and source labels
    files = [
        {'glu': 'path_to_glu_group1.csv', 'klu': 'path_to_klu_group1.csv', 'nicslu': 'path_to_nicslu_group1.csv', 'pardiso': 'path_to_pardiso_group1.csv'},
        # Add other file groups here
    ]
    source_labels = ['group1', 'group2', 'group3', 'group4']  # Adjust to match your groups
    
    # Load and preprocess data
    combined_data = load_and_merge_data(files, source_labels)
    processed_data = preprocess_data(combined_data)
    processed_data.to_csv('processed_data.csv', index=False)
    print("Data processing complete. Saved to 'processed_data.csv'.")
