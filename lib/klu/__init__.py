import subprocess
import logging
from pathlib import Path

# Import KluBenchmark from the correct path
try:
    from .src.run_klu_kernels import KluBenchmark
except ImportError as e:
    logging.error(f"Error importing KluBenchmark: {e}")
    raise


def compile_cpp():
    """
    Compiles the C++ code using the Makefile in the src directory.
    """
    # Construct the correct path to the Makefile
    makefile_path = Path(__file__).parent / "src/Makefile"

    # Check if the Makefile exists
    if not makefile_path.is_file():
        logging.error(f"Makefile not found at {makefile_path}")
        raise FileNotFoundError(f"Makefile not found at {makefile_path}")

    try:
        logging.info(f"Compiling C++ code using Makefile at {makefile_path}")
        # Run 'make klu_kernel' in the src directory
        subprocess.run(["make", "klu_kernel", "-C", str(makefile_path.parent)], check=True)
        logging.info("Compilation successful.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Compilation failed: {e}")
        raise


# Automatically compile the C++ code when the package is imported
compile_cpp()


# Example of initializing the processor
def initialize_klu_benchmark(database_folder=None, engine=None, timeout=100):
    """
    Initializes the KluBenchmark with default or custom parameters and returns the instance.
    """
    engine = engine or './klu_kernel.o'
    database_folder = database_folder or "/mnt/c/Work/transient-simulation/Random_Circuit_Generator/random_circuit_matrixs"

    logging.info(f"Initializing KLU Benchmark with parameters: \n"
                 f"Database Folder: {database_folder}\n"
                 f"Engine: {engine}\n"
                 f"Timeout: {timeout}")

    return KluBenchmark(engine=engine, database_folder=database_folder, timeout=timeout)
