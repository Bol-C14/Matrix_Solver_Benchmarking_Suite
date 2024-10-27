import numpy as np
import logging
from lib.random_circuit_generator import run_circuit_generation
from lib.klu_new.run_klu_kernels import KluBenchmark
import sys
from pathlib import Path

# Add lib/random_circuit_generator to sys.path
sys.path.append(str(Path(__file__).resolve().parent / "lib" / "random_circuit_generator"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Main script settings
CIRCUIT_OUTPUT_DIR = Path("data/circuit_data")
ENGINE_PATH = "./lib/klu_new/src/klu_kernel.o"
DATABASE_FOLDER = CIRCUIT_OUTPUT_DIR

# Ensure output directories exist
CIRCUIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Generate matrices and save to `data/circuit_data`
def generate_matrices():
    node_iterations = 10  # Number of times to iterate for each node count
    node_numbers = np.concatenate((np.arange(5, 100, 5), np.arange(100, 1000, 50)))

    logger.info(f"Generating matrices in {CIRCUIT_OUTPUT_DIR}")
    run_circuit_generation(node_iterations, node_numbers, output_directory=CIRCUIT_OUTPUT_DIR, verbose=True)
    logger.info("Matrix generation completed.")

# Step 2: Solve and benchmark matrices using KLU
def benchmark_matrices():
    logger.info(f"Initializing KLU benchmark with engine: {ENGINE_PATH}")
    klu_benchmark = KluBenchmark(ENGINE_PATH, DATABASE_FOLDER)

    # Find all generated .mtx files in the database folder
    klu_benchmark.find_mtx_files()

    # Run the benchmark on each matrix file and save results
    klu_benchmark.run_benchmark()
    logger.info("Benchmarking completed.")

if __name__ == "__main__":
    # Generate matrices
    generate_matrices()

    # Benchmark matrices
    benchmark_matrices()
