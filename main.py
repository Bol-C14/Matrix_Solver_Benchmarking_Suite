import numpy as np
import logging
import sys
from pathlib import Path
import os
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

# Main script settings
CIRCUIT_OUTPUT_DIR = Path("data/circuit_data")
DATABASE_FOLDER = CIRCUIT_OUTPUT_DIR

# Ensure output directories exist
CIRCUIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Generate matrices and save to `data/circuit_data`
def generate_matrices(logger):
    node_iterations = 20  # Number of times to iterate for each node count
    node_numbers = np.concatenate((np.arange(5, 100, 5), np.arange(100, 1000, 50)))

    logger.info(f"Generating matrices in {CIRCUIT_OUTPUT_DIR}")
    run_circuit_generation(node_iterations, node_numbers, output_directory=CIRCUIT_OUTPUT_DIR, verbose=False)
    logger.info("Matrix generation completed.")

# Step 2.1: Solve and benchmark matrices using KLU
def run_klu(logger):
    engine_path = "./lib/klu_new/src/klu_kernel.o"   # if not found, go compile it

    logger.info(f"Initializing KLU benchmark with engine: {engine_path}")
    klu_benchmark = KluBenchmark(engine_path, DATABASE_FOLDER)

    # Find all generated .mtx files in the database folder
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
    engine_path = "./lib/glu/src/glu_kernel.o"  # Ensure this is correctly built if necessary
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
    database_folder = "./data/circuit_data"
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
    database_folder = "./data/circuit_data"  # Example path; update as needed

    logger.info("Initializing SuperLU benchmark.")
    superlu_benchmark = SuperluBenchmark(database_folder, timeout=100)

    # Find .mtx files in the database folder
    superlu_benchmark.find_mtx_files()

    # Run the benchmark and save results
    superlu_benchmark.run_benchmark()
    logger.info("SuperLU Benchmarking completed.")
    
    
def main():
    parser = argparse.ArgumentParser(description="Run matrix generation and benchmarks for KLU, NICSLU, and GLU.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    args = parser.parse_args()

    # Set up logging based on the command-line argument
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(log_level)

    # Generate matrices
    generate_matrices(logger)

    # Benchmark matrices with KLU
    run_klu(logger)
    
    # Benchmark matrices with NICSLU
    run_nicslu(logger)

    # # Benchmark matrices with GLU
    # run_glu(logger)  # Disabled if running directly

    # Benchmark matrices with PARDISO
    run_pardiso(logger)
    
    # Benchmark matrices with SuperLU
    run_superlu(logger)

if __name__ == "__main__":
    main()
