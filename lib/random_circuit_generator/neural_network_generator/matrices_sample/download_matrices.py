import os
import ssgetpy

# Create a directory to store downloaded matrices
base_dir = 'SuiteSparse_Matrices'
os.makedirs(base_dir, exist_ok=True)

# Fetch all matrices with a very high limit
print("Fetching all matrices...")
matrices = ssgetpy.search(limit=1000000)  # Use a very large limit instead of None

print(f"Total matrices found: {len(matrices)}")

# Iterate through each matrix
for matrix in matrices:
    try:
        # Replace spaces with underscores for valid directory names
        kind = matrix.kind.replace(' ', '_') if matrix.kind else "unknown_kind"
        kind_dir = os.path.join(base_dir, kind)
        os.makedirs(kind_dir, exist_ok=True)

        # Define the path for the Matrix Market file
        mtx_path = os.path.join(kind_dir, f"{matrix.name}.mtx")

        # Download the matrix in Matrix Market format if not already downloaded
        if not os.path.exists(mtx_path):
            print(f"Downloading {matrix.name} of kind {matrix.kind}...")
            matrix.download(format='MM', destpath=mtx_Zpath)
        else:
            print(f"{matrix.name} already exists. Skipping download.")
    except Exception as e:
        print(f"Error processing {matrix.name}: {e}")
