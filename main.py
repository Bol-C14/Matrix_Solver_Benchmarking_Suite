import numpy as np
import logging
from lib.random_circuit_generator import run_circuit_generation
from lib.klu_new.run_klu_kernels import KluBenchmark
import sys
from pathlib import Path

# Add lib/random_circuit_generator to sys.path
sys.path.append(str(Path(__file__).resolve().parent / "lib" / "random_circuit_generator"))

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Console handler for INFO and higher levels
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# Optional debug handler (writes DEBUG logs to a file, if needed)
debug_handler = logging.FileHandler('debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(debug_handler)

# Main script settings
CIRCUIT_OUTPUT_DIR = Path("data/circuit_data")

DATABASE_FOLDER = CIRCUIT_OUTPUT_DIR

# Ensure output directories exist
CIRCUIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Generate matrices and save to `data/circuit_data`
def generate_matrices():
    node_iterations = 10  # Number of times to iterate for each node count
    node_numbers = np.concatenate((np.arange(5, 100, 5), np.arange(100, 1000, 50)))

    logger.info(f"Generating matrices in {CIRCUIT_OUTPUT_DIR}")
    run_circuit_generation(node_iterations, node_numbers, output_directory=CIRCUIT_OUTPUT_DIR, verbose=False)
    logger.info("Matrix generation completed.")

# Step 2.1: Solve and benchmark matrices using KLU
def run_klu():
    engine_path = "./lib/klu_new/src/klu_kernel.o"   # if not found, go compile it

    logger.info(f"Initializing KLU benchmark with engine: {engine_path}")
    klu_benchmark = KluBenchmark(engine_path, DATABASE_FOLDER)

    # Find all generated .mtx files in the database folder
    klu_benchmark.find_mtx_files()

    # Run the benchmark on each matrix file and save results
    klu_benchmark.run_benchmark()
    logger.info("Benchmarking completed.")

# Step 2.2 Solve and benchmark matrices using NICSLU
def run_nicslu():
    engine_path = "./lib/nicslu/src/nicslu_kernel.o"  # if not found, go compile it

    logger.info(f"Initializing NICSLU benchmark with engine: {engine_path}")

if __name__ == "__main__":
    # Generate matrices
    generate_matrices()

    # Benchmark matrices
    run_klu()
