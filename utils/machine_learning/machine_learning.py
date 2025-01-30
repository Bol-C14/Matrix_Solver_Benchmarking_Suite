import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def process_zip_files(zip_files, extraction_dir):
    os.makedirs(extraction_dir, exist_ok=True)
    group_data = []
    matrix_features_data = []

    for group, zip_file in zip_files.items():
        # Create a group-specific directory for extraction
        group_dir = os.path.join(extraction_dir, group)
        os.makedirs(group_dir, exist_ok=True)

        # Extract files from the zip
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(group_dir)

        # Process each file in the extracted directory
        for file_name in os.listdir(group_dir):
            file_path = os.path.join(group_dir, file_name)

            # Process kernel result files
            if file_name.startswith('results') and file_name.endswith('.csv'):
                kernel_name = file_name.split('_')[1]  # Extract kernel name
                df = pd.read_csv(file_path)
                df['kernel'] = kernel_name
                df['source'] = group
                group_data.append(df)

            # Process matrix_features.csv
            elif file_name == 'matrix_features.csv':
                matrix_features_df = pd.read_csv(file_path)
                matrix_features_df['source'] = group
                matrix_features_data.append(matrix_features_df)

    # Combine all datasets
    combined_data = pd.concat(group_data, ignore_index=True)
    all_matrix_features = pd.concat(matrix_features_data, ignore_index=True)

    # Standardize matrix names in features
    all_matrix_features['Matrix'] = all_matrix_features['Matrix'].str.replace('.mtx', '', regex=False)

    # Merge datasets on matrix and source
    merged_data = pd.merge(
        combined_data,
        all_matrix_features.rename(columns={'Matrix': 'mtx'}),
        on=['mtx', 'source'],
        how='inner'
    )

    return merged_data

def label_and_clean_data(merged_data):
    # Add a "nicslu_failed" label for NICSLU-specific conditions
    merged_data['special_label'] = None
    nicslu_failed_condition = (merged_data['kernel'] == 'nicslu') & \
                               (merged_data['Factorization'] < 100) & \
                               (merged_data['nnz'] > 1e3)
    merged_data.loc[nicslu_failed_condition, 'special_label'] = 'nicslu_failed'

    # Ensure "nicslu_failed" cases are not selected as the fastest kernel
    merged_data['effective_factorization'] = merged_data['Factorization']
    merged_data.loc[nicslu_failed_condition, 'effective_factorization'] = float('inf')

    # Determine the fastest kernel based on effective_factorization time
    fastest_kernel_indices = merged_data.groupby(['source', 'mtx'])['effective_factorization'].idxmin()
    merged_data['fastest_kernel'] = None
    merged_data.loc[fastest_kernel_indices, 'fastest_kernel'] = merged_data.loc[fastest_kernel_indices, 'kernel']

    # Drop rows without a fastest kernel
    cleaned_data = merged_data.dropna(subset=['fastest_kernel'])

    return cleaned_data

def prepare_data_for_ml(cleaned_data, feature_columns):
    # Extract features and labels
    features = cleaned_data[feature_columns]
    labels = cleaned_data['fastest_kernel']

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def clean_features(dataframe):
    # Convert complex numbers to their magnitudes (absolute values) and handle non-numeric values
    return dataframe.applymap(lambda x: abs(x) if isinstance(x, complex) else (float(x) if isinstance(x, (int, float)) else np.nan))

def train_models(X_train, X_test, y_train, y_test, models):
    # Initialize results dictionary
    results = {}

    # Train and evaluate models
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=y_train.unique())
        results[model_name] = {"accuracy": acc, "report": report}

    return results

if __name__ == "__main__":
    # Define zip file paths and output directory
    zip_files = {
        'group1_cr': 'path_to/data_group_01_cr.zip',
        'group2_csw': 'path_to/data_group_02_csw.zip',
        'group3_pr': 'path_to/data_group_03_pr.zip',
        'group4_ss': 'path_to/data_group_04_ss.zip'
    }
    extraction_dir = 'path_to/extracted_datasets'

    # Feature columns for the model
    feature_columns = [
        'nnz', 'rows', 'threads', 'Locality', 'Sparsity', 'Ncol', 'Nrow', 'NNZL', 'NNZU', 
        'NNZmax', 'NNZavg', 'DEmax', 'DEmin', 'bwL', 'bwU', 'NO', 'NDDrow', 'OneNorm', 
        'FroNorm', 'Ndim', 'Meshdo', 'Meshrl', 'Kappa_hmin', 'Kappa_hmax', 'sp_all', 
        'n2_structural', 'sp_nz', 'n2_numerical', 'psym', 'nnz_structural_symmetry', 
        'nsym_numerical_symmetry', 'Bandwidth', 'BandWidthTotal', 'CoefficientOfVariation'
    ]

    # Process datasets
    merged_data = process_zip_files(zip_files, extraction_dir)

    # Add labels and clean data
    cleaned_data = label_and_clean_data(merged_data)

    # Prepare data for ML
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_ml(cleaned_data, feature_columns)

    # Clean features
    X_train = clean_features(X_train).dropna()
    X_val = clean_features(X_val).dropna()
    X_test = clean_features(X_test).dropna()

    # Match labels after cleaning
    y_train = y_train.loc[X_train.index]
    y_val = y_val.loc[X_val.index]
    y_test = y_test.loc[X_test.index]

    # Initialize models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Bayesian Network": GaussianNB(),
    }

    # Train models
    results = train_models(X_train, X_test, y_train, y_test, models)

    # Display results
    for model_name, result in results.items():
        print(f"Model: {model_name}")
        print(f"Accuracy: {result['accuracy']}")
        print("Classification Report:")
        print(result['report'])
