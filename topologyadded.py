import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

# File paths
file_paths = {
    "glu": "results_glu_kernel.csv",
    "klu": "results_klu_kernel.csv",
    "nicslu": "results_nicslu_kernel.csv",
    "pardiso": "results_pardiso_kernel.csv",
    "superlu": "results_superlu_scipy.csv",
    "features": "matrix_features.csv"
}

# Load kernel timing files and features file
data = {name: pd.read_csv(path) for name, path in file_paths.items() if name != "features"}
matrix_features = pd.read_csv(file_paths["features"])

# Filter and calculate Total_Time for each kernel dataset
filtered_data = {}
for name, df in data.items():
    if 'Factorization' in df.columns:
        valid_df = df[(df['Analyze'] > 0) & (df['Factorization'] > 0)].copy()
        valid_df.loc[:, 'Total_Time'] = valid_df['Analyze'] + valid_df['Factorization']
    elif 'Factorization_1' in df.columns and 'Factorization_2' in df.columns:
        valid_df = df[(df['Analyze'] > 0) & (df['Factorization_1'] > 0) & (df['Factorization_2'] > 0)]
        valid_df['Total_Time'] = valid_df['Analyze'] + valid_df['Factorization_1'] + valid_df['Factorization_2']
    filtered_data[name] = valid_df

# Combine all data into one DataFrame
time_data = []
for name, df in filtered_data.items():
    for _, row in df.iterrows():
        time_data.append({
            'Matrix': row['mtx'],
            'Kernel': name,
            'Analyze_Time': row['Analyze'],
            'Factorization_Time': row.get('Factorization', row.get('Factorization_1', 0) + row.get('Factorization_2', 0)),
            'Total_Time': row['Total_Time']
        })
time_df = pd.DataFrame(time_data)

# Clean matrix names for merging
matrix_features['Matrix'] = matrix_features['Matrix'].str.replace('.mtx', '', regex=False)

# Merge timing data with matrix features
time_features_df = pd.merge(matrix_features, time_df, on="Matrix")

# Calculate topology features for each matrix
def matrix_topology_features(sparse_matrix):
    graph = nx.from_scipy_sparse_array(sparse_matrix)
    features = {
        'bandedness': np.mean(np.abs(sparse_matrix.diagonal())),  # Example bandedness
        'average_clustering': nx.average_clustering(graph),
        'avg_degree': np.mean([d for _, d in graph.degree()]),
        'max_degree': max(dict(graph.degree()).values()),
        'std_degree': np.std([d for _, d in graph.degree()])
    }
    return features

topology_data = []
for matrix_name in time_features_df['Matrix'].unique():
    matrix = mmread(f"./data/circuit_data/{matrix_name}.mtx")  # Adjust path accordingly
    sparse_matrix = sp.csr_matrix(matrix)
    features = matrix_topology_features(sparse_matrix)
    features['Matrix'] = matrix_name
    topology_data.append(features)

# Convert topology features to DataFrame and merge with timing and matrix features
topology_df = pd.DataFrame(topology_data)
combined_df = pd.merge(time_features_df, topology_df, on="Matrix")

# Select only numeric columns for correlation calculation
numeric_combined_df = combined_df.select_dtypes(include=[np.number])

# Calculate correlation on numeric data
correlation_data = numeric_combined_df.corr()[['Analyze_Time', 'Factorization_Time', 'Total_Time']]


# Visualize correlations with a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm')
plt.title("Overall Correlation Heatmap of Features with Computation Times")

plt.savefig('correlation_heatmap.png')
