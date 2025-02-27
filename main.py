# main.py

import numpy as np
import logging
import sys
from pathlib import Path
import os
import argparse
import shutil  # Added import for file operations

from utils.suitesparse_handler import SuiteSparseManager
from lib.random_circuit_generator import run_circuit_generation
from lib.klu_new.run_klu_kernels import KluBenchmark
from lib.nicslu.src.run_nicslu_kernel import NICSLUTester
from lib.glu.src.run_glu_kernels import GluKernelBenchmark
from lib.pardiso.run_pardiso_kernel import PardisoKernelBenchmark
from lib.superlu.run_superlu_kernels import SuperluBenchmark

# Import the pure_random_generator module
from lib.neural_network_generator.pure_random_generator import (
    MatrixGenerator,
    MatrixGeneratorConfig,
    SizeConfig,  # Add this import
    SparsityConfig,
    ShapeConfig,
    ValueConfig,
    OutputConfig,
    MatrixSaver,
    MatrixShape,
    generate_matrix_batch
)


def setup_logging(log_level=logging.INFO):
    logger = logging.getLogger()
    
    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # Console handler for log output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
     
    # File handler for DEBUG logs
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
SS_ORGANIZED_DATA_DIR = WORKSPACE_PATH / "data" / "ss_organized_data"  # Directory for organized .mtx files
GENERATED_MATRICES_DIR = WORKSPACE_PATH / "data" / "generated_matrices"  # Directory for pure_random_generator's output


# Ensure output directories exist
CIRCUIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUITE_SPARSE_DATA_DIR.mkdir(parents=True, exist_ok=True)
SS_ORGANIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure ss_organized_data exists
GENERATED_MATRICES_DIR.mkdir(parents=True, exist_ok=True)  # Ensure generated_matrices exists


# Benchmarking Functions
def run_klu(logger, database_folder):
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
    klu_benchmark = KluBenchmark(engine_path, database_folder)

    # Find all .mtx files in the database folder
    klu_benchmark.find_mtx_files()

    # Run the benchmark on each matrix file and save results
    klu_benchmark.run_benchmark()
    logger.info("KLU Benchmarking completed.")


def run_nicslu(logger, database_folder):
    # Set the path to NICSLU shared library directory relative to the repo root
    lib_path = str(Path("./lib/nicslu/src/linux/lib_centos6_x64_gcc482_fma/int32").resolve())

    # Add the NICSLU library directory to LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{lib_path}"
    
    engine_path = "./lib/nicslu/src/nicslu_kernel.o"
    logger.info(f"Initializing NICSLU benchmark with engine: {engine_path}")
    
    # Create an instance of NICSLUTester
    nicslu_tester = NICSLUTester(
        database_folder=database_folder,
        engine_path=engine_path,
        timeout=100
    )

    # Define parameters for the benchmark
    reps_value = 100
    thread_values = [1, 4, 8, 12, 24, 32]
    
    # Run the NICSLU benchmark
    nicslu_tester.run(reps_value=reps_value, thread_values=thread_values)
    logger.info("NICSLU Benchmarking completed.")


def run_glu(logger, database_folder):
    engine_path = "./lib/glu/src/glu_kernel"  # Ensure this is correctly built if necessary
    logger.info(f"Initializing GLU benchmark with engine: {engine_path}")

    # Create an instance of GluKernelBenchmark
    glu_benchmark = GluKernelBenchmark(
        engine=engine_path,
        database_folder=database_folder,
        timeout=100,
        log_level=logging.DEBUG  # Change this as needed
    )

    # Run the benchmark
    glu_benchmark.process_matrices()
    logger.info("GLU Benchmarking completed.")


def run_pardiso(logger, database_folder):
    engine_path = "./lib/pardiso/pardiso_kernel.o"  # Ensure this is correctly compiled as an executable
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


def run_superlu(logger, database_folder):
    logger.info("Initializing SuperLU benchmark.")
    superlu_benchmark = SuperluBenchmark(database_folder, timeout=100)

    # Find .mtx files in the database folder
    superlu_benchmark.find_mtx_files()

    # Run the benchmark and save results
    superlu_benchmark.run_benchmark()
    logger.info("SuperLU Benchmarking completed.")


def run_random_circuit_generation(logger):
    node_iterations = 3  # Number of times to iterate for each node count
    node_numbers = np.concatenate((np.arange(5, 100, 5), np.arange(20, 2000, 80)))

    logger.info(f"Generating matrices in {CIRCUIT_OUTPUT_DIR}")
    run_circuit_generation(node_iterations, node_numbers, output_directory=CIRCUIT_OUTPUT_DIR, verbose=False, connection_method = "small_world")
    logger.info("Matrix generation completed.")
    
    benchmark_database_folder = CIRCUIT_OUTPUT_DIR
    
    return benchmark_database_folder


def run_pure_random_generation_and_benchmark(logger):
    """Generates matrices using pure_random_generator and runs benchmarks on them."""
    logger.info("Starting pure random matrix generation and benchmarking.")

    # Configure the generator
    generator_config = MatrixGeneratorConfig(
        size=SizeConfig(min_size=100, max_size=800, size_step=100, random_size=False),
        sparsity=SparsityConfig(
            min_sparsity=0.8,
            max_sparsity=0.95,
            random_sparsity=False,
            enable_range_generation=True,
            range_start=0.8,  # Refined range for better control
            range_end=0.95,
            num_steps=10,  # Increased steps for finer granularity
            decrease_with_size=True,
            random_decrease=False,
            decrease_rate=0.05  # Adjusted decrease rate for smoother transitions
        ),
        shape=ShapeConfig(
            shapes=[
                MatrixShape.RANDOM,
                MatrixShape.DIAGONAL,
                MatrixShape.BANDED,
                MatrixShape.SPARSELY_RANDOM,
                MatrixShape.POSITIVE_DEFINITE
            ],
            probabilities=[0.3, 0.2, 0.2, 0.2, 0.1],
            attempts_per_shape=3
        ),
        value=ValueConfig(min_val=-10, max_val=10),
        seed=42,
        repeat=1  # Generate matrices for one iteration
    )

    # Configure the output
    output_config = OutputConfig(
        output_dir=GENERATED_MATRICES_DIR,
        file_prefix="matrix",
        file_extension="mtx"
    )

    # Initialize the generator and saver
    generator = MatrixGenerator(generator_config)
    saver = MatrixSaver(output_config)

    # Generate matrices
    saved_matrix_paths = generate_matrix_batch(generator, saver)

    logger.info(f"Generated and saved {len(saved_matrix_paths)} matrices using pure_random_generator.")

    benchmark_database_folder = GENERATED_MATRICES_DIR

    # Log the database folder being used for benchmarks
    logger.info(f"Benchmarking will use matrices from: {benchmark_database_folder}")

    return benchmark_database_folder



def main():
    parser = argparse.ArgumentParser(description="Run matrix generation and benchmarks for KLU, NICSLU, GLU, PARDISO, and SuperLU.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--use_suite_sparse", action='store_true', help="Use suite_sparse_data instead of generating new matrices")
    group.add_argument("--use_circuit", action='store_true', help="Use circuit generator to create matrices")
    group.add_argument("--use_pure_random", action='store_true', help="Use pure random generator to create matrices")
    parser.add_argument("--download_only", action='store_true', help="Only download and organize SuiteSparse matrices")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    # Future argument for neural network generator can be added here
    args = parser.parse_args()

    # Set up logging based on the command-line argument
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(log_level)

    if args.use_suite_sparse:
        # Initialize SuiteSparseManager
        suite_sparse_manager = SuiteSparseManager(
            suite_sparse_dir=SUITE_SPARSE_DATA_DIR,
            organized_dir=SS_ORGANIZED_DATA_DIR,
            logger=logger
        )
        
        # Perform download and organization
        suite_sparse_manager.prepare_matrices()
        
        database_folder = SS_ORGANIZED_DATA_DIR
    elif args.use_circuit:
        # Step 1: Generate matrices using random circuit generator
        database_folder = run_random_circuit_generation(logger)
        database_folder = CIRCUIT_OUTPUT_DIR
    elif args.use_pure_random:
        # Step 3: Generate matrices using pure random generator and benchmark
        run_pure_random_generation_and_benchmark(logger)
        database_folder = GENERATED_MATRICES_DIR

    if args.download_only:
        logger.info("Download and organization completed. Exiting as per --download_only flag.")
        sys.exit(0)

    # Benchmarking routines
    # Uncomment the benchmarks you wish to run if not already included in the routines above
    # For example, if run_pure_random_generation_and_benchmark runs benchmarks, you don't need to run them here
    # Alternatively, if you wish to run additional benchmarks, add here
    run_klu(logger, database_folder)
    run_nicslu(logger, database_folder)
    run_glu(logger, database_folder)
    run_pardiso(logger, database_folder)
    # run_superlu(logger, database_folder)


if __name__ == "__main__":
    main()
