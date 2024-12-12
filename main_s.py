import numpy as np
import logging
import sys
from pathlib import Path
import os
import tarfile
import shutil  # Added for file copying
from lib.random_circuit_generator import run_circuit_generation
from lib.klu_new.run_klu_kernels import KluBenchmark
from lib.nicslu.src.run_nicslu_kernel import NICSLUTester
from lib.glu.src.run_glu_kernels import GluKernelBenchmark
from lib.pardiso.run_pardiso_kernel import PardisoKernelBenchmark
from lib.superlu.run_superlu_kernels import SuperluBenchmark  # Import the SuperLUBenchmark class
import argparse

# Set up logging with multiple levels
def setup_logging(log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Console handler for log output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler for DEBUG logs, if needed
    debug_handler = logging.FileHandler('debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(debug_handler)

    return logger

# Define directories
WORKSPACE_PATH = Path.cwd()  # Adjust as necessary
SUITE_SPARSE_DATA_DIR = WORKSPACE_PATH / "data" / "suite_sparse_data"
CIRCUIT_OUTPUT_DIR = WORKSPACE_PATH / "data" / "circuit_data"
SS_ORGANIZED_DATA_DIR = WORKSPACE_PATH / "data" / "ss_organized_data"  # New directory for organized .mtx files
DATABASE_FOLDER = SS_ORGANIZED_DATA_DIR  # Update DATABASE_FOLDER to ss_organized_data

# Ensure output directories exist
CIRCUIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUITE_SPARSE_DATA_DIR.mkdir(parents=True, exist_ok=True)
SS_ORGANIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure ss_organized_data exists

# Function to extract all tar.gz files under suite_sparse_data
def extract_all_tar_gz(logger):
    logger.info(f"Starting extraction of .tar.gz files in {SUITE_SPARSE_DATA_DIR}")
    # Use glob to find all .tar.gz files recursively
    tar_gz_files = list(SUITE_SPARSE_DATA_DIR.rglob("*.tar.gz"))
    logger.info(f"Found {len(tar_gz_files)} .tar.gz files to extract.")

    for tar_gz_path in tar_gz_files:
        # Define the extraction directory (same as tar.gz file without .tar.gz)
        extraction_dir = tar_gz_path.parent / tar_gz_path.stem
        if not extraction_dir.exists():
            extraction_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Extracting {tar_gz_path} to {extraction_dir}")
            try:
                with tarfile.open(tar_gz_path, "r:gz") as tar:
                    tar.extractall(path=extraction_dir)
                logger.info(f"Extracted {tar_gz_path} successfully.")
            except (tarfile.TarError, EOFError) as e:
                logger.error(f"Failed to extract {tar_gz_path}: {e}")
                logger.info(f"Extracted {tar_gz_path} successfully.")
            except (tarfile.TarError, EOFError) as e:
                logger.error(f"Failed to extract {tar_gz_path}: {e}")
        else:
            logger.debug(f"Extraction directory {extraction_dir} already exists. Skipping extraction.")

    logger.info("All .tar.gz files have been processed.")

# Function to verify that all expected .mtx files are present after extraction
def verify_mtx_files(logger):
    logger.info(f"Verifying .mtx files in {SUITE_SPARSE_DATA_DIR}")
    mtx_files = list(SUITE_SPARSE_DATA_DIR.rglob("*.mtx"))
    logger.info(f"Found {len(mtx_files)} .mtx files.")
    if not mtx_files:
        logger.warning("No .mtx files found. Please check the extraction process.")
    else:
        logger.info("All .mtx files are present.")
    return mtx_files

def remove_mtx_header(mtx_file_path):
    """
    Remove headers from the .mtx file, retain necessary metadata, 
    and return the cleaned content with a Matrix Market header.
    """
    with open(mtx_file_path, 'r') as file:
        lines = file.readlines()

    # Identify and retain the first line starting with '%%MatrixMarket'
    matrix_market_header = None
    for line in lines:
        if line.startswith('%%MatrixMarket'):
            matrix_market_header = line.strip()
            break

    # If no Matrix Market header is found, set a default
    if not matrix_market_header:
        matrix_market_header = "%%MatrixMarket matrix coordinate real general"

    # Skip all comment lines (starting with '%') and retain data lines
    data_lines = [line for line in lines if not line.startswith('%')]

    # Combine the retained Matrix Market header with the data lines
    cleaned_content = f"{matrix_market_header}\n" + ''.join(data_lines)

    return cleaned_content


# Function to organize .mtx files by copying main .mtx files to ss_organized_data
def organize_mtx_files(logger):
    """
    Organize .mtx files by removing headers and copying them to ss_organized_data.
    """
    logger.info(f"Organizing .mtx files into {SS_ORGANIZED_DATA_DIR}")
    
    # Find all extracted directories under suite_sparse_data
    extracted_dirs = [p for p in SUITE_SPARSE_DATA_DIR.rglob("*") if p.is_dir()]
    logger.info(f"Found {len(extracted_dirs)} directories to search for .mtx files.")

    # Iterate through directories to find the main .mtx files
    for directory in extracted_dirs:
        # Search for .mtx files in the directory
        mtx_files = list(directory.glob("*.mtx"))
        logger.debug(f"Found {len(mtx_files)} .mtx files in {directory}")

        for mtx_file in mtx_files:
            # Skip files with '_x.mtx' or '_b.mtx' suffix
            if mtx_file.name.endswith(('_x.mtx', '_b.mtx')):
                logger.debug(f"Skipping auxiliary file: {mtx_file}")
                continue

            # Prepare the destination path
            destination = SS_ORGANIZED_DATA_DIR / mtx_file.name
            if not destination.exists():
                try:
                    # Clean the .mtx file by removing its header
                    logger.debug(f"Cleaning headers for {mtx_file}")
                    cleaned_data = remove_mtx_header(mtx_file)

                    # Write the cleaned data to the destination
                    with open(destination, 'w') as cleaned_file:
                        cleaned_file.write(cleaned_data)
                    
                    logger.info(f"Cleaned and copied {mtx_file} to {destination}")
                except IOError as e:
                    logger.error(f"Failed to process {mtx_file}: {e}")
            else:
                logger.debug(f"File {destination} already exists. Skipping copy.")

    logger.info(f"Organized .mtx files are available in {SS_ORGANIZED_DATA_DIR}")


# Step 1: Generate matrices and save to `data/circuit_data`
def generate_matrices(logger):
    node_iterations = 1  # Number of times to iterate for each node count
    node_numbers = np.concatenate((np.arange(5, 100, 5), np.arange(20, 1000, 80)))

    logger.info(f"Generating matrices in {CIRCUIT_OUTPUT_DIR}")
    run_circuit_generation(node_iterations, node_numbers, output_directory=CIRCUIT_OUTPUT_DIR, verbose=False)
    logger.info("Matrix generation completed.")

# Step 2.1: Solve and benchmark matrices using KLU
def run_klu(logger):
    # Set library paths for KLU dependencies
    lib_paths = [
        "./lib/klu_new/src/SuiteSparse-stable/lib",
        "./lib/klu_new/src/SuiteSparse-stable/KLU/build"
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_paths + [os.environ.get("LD_LIBRARY_PATH", "")])

    # Check if the LD_LIBRARY_PATH includes the necessary paths for libklu.so.2
    logger.info(f"LD_LIBRARY_PATH set to: {os.environ['LD_LIBRARY_PATH']}")

    # Set the path to the KLU kernel executable
    engine_path = "./lib/klu_new/src/klu_kernel.o"  # Ensure this path is correct
    if not os.path.exists(engine_path):
        logger.error(f"KLU engine not found at {engine_path}. Please compile it if necessary.")
        return

    # Initialize the benchmark
    logger.info(f"Initializing KLU benchmark with engine: {engine_path}")
    klu_benchmark = KluBenchmark(engine_path, DATABASE_FOLDER)

    # Find all .mtx files in the database folder
    klu_benchmark.find_mtx_files()

    # Run the benchmark on each matrix file and save results
    klu_benchmark.run_benchmark()
    logger.info("KLU Benchmarking completed.")

# Step 2.2: Solve and benchmark matrices using NICSLU
def run_nicslu(logger):
    # Set the path to NICSLU shared library directory relative to the repo root
    lib_path = str(Path("./lib/nicslu/src/linux/lib_centos6_x64_gcc482_fma/int32").resolve())

    # Add the NICSLU library directory to LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{lib_path}"
    
    engine_path = "./lib/nicslu/src/nicslu_kernel.o"
    logger.info(f"Initializing NICSLU benchmark with engine: {engine_path}")
    
    # Create an instance of NICSLUTester
    nicslu_tester = NICSLUTester(
        database_folder=DATABASE_FOLDER,
        engine_path=engine_path,
        timeout=100
    )

    # Define parameters for the benchmark
    reps_value = 100
    thread_values = [1, 4, 8, 12, 24, 32]
    
    # Run the NICSLU benchmark
    nicslu_tester.run(reps_value=reps_value, thread_values=thread_values)
    logger.info("NICSLU Benchmarking completed.")

# Step 2.3: Solve and benchmark matrices using GLU
def run_glu(logger):
    engine_path = "./lib/glu/src/glu_kernel"  # Ensure this is correctly built if necessary
    logger.info(f"Initializing GLU benchmark with engine: {engine_path}")

    # Create an instance of GluKernelBenchmark
    glu_benchmark = GluKernelBenchmark(
        engine=engine_path,
        database_folder=DATABASE_FOLDER,
        timeout=100,
        log_level=logging.DEBUG  # Change this as needed
    )

    # Run the benchmark
    glu_benchmark.process_matrices()
    logger.info("GLU Benchmarking completed.")

# Step 2.4: Solve and benchmark matrices using PARDISO
def run_pardiso(logger):
    engine_path = "./lib/pardiso/pardiso_kernel.o"  # Ensure this is correctly compiled as an executable
    database_folder = DATABASE_FOLDER
    timeout = 120
    log_level = logging.INFO
    repetitions = 50
    thread_values = [1, 2, 4, 8, 16]

    # Initialize the PardisoKernelBenchmark instance
    pardiso_benchmark = PardisoKernelBenchmark(
        engine=engine_path,
        database_folder=database_folder,
        timeout=timeout,
        log_level=log_level
    )

    # Run the process_matrices method with specified repetitions and threads
    pardiso_benchmark.process_matrices(repetitions=repetitions, thread_values=thread_values)
    logger.info("PARDISO Benchmarking completed.")

# Step 2.5: Solve and benchmark matrices using SuperLU
# SuperLU benchmark function
def run_superlu(logger):
    database_folder = DATABASE_FOLDER  # Updated to use ss_organized_data

    logger.info("Initializing SuperLU benchmark.")
    superlu_benchmark = SuperluBenchmark(database_folder, timeout=100)

    # Find .mtx files in the database folder
    superlu_benchmark.find_mtx_files()

    # Run the benchmark and save results
    superlu_benchmark.run_benchmark()
    logger.info("SuperLU Benchmarking completed.")
    
def main():
    parser = argparse.ArgumentParser(description="Run matrix generation and benchmarks for KLU, NICSLU, GLU, PARDISO, and SuperLU.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--use_suite_sparse", action='store_true', help="Use suite_sparse_data instead of generating new matrices")
    args = parser.parse_args()

    # Set up logging based on the command-line argument
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(log_level)

    if args.use_suite_sparse:
        # Step A: Extract all tar.gz files in suite_sparse_data
        extract_all_tar_gz(logger)

        # Step B: Verify that .mtx files are present
        mtx_files = verify_mtx_files(logger)

        if not mtx_files:
            logger.error("No .mtx files found after extraction. Exiting.")
            sys.exit(1)

        # Step C: Organize .mtx files by copying main .mtx files to ss_organized_data
        organize_mtx_files(logger)
    else:
        # Step 1: Generate matrices
        # generate_matrices(logger)
        pass
        # If you still want to use generated matrices, you might need to adjust DATABASE_FOLDER accordingly
        # For now, it's set to ss_organized_data which uses suite_sparse_data

    # # Benchmark matrices with KLU
    # run_klu(logger)
    
    # # Benchmark matrices with NICSLU
    # run_nicslu(logger)

    # # Benchmark matrices with GLU
    # run_glu(logger)  # Enabled

    # Benchmark matrices with PARDISO
    run_pardiso(logger)
    
    # Benchmark matrices with SuperLU
    # run_superlu(logger)

if __name__ == "__main__":
    main()
