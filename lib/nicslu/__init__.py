import os
import subprocess
import logging
from pathlib import Path
from .src.run_nicslu_kernel import NicsluProcessor

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('nicslu_init.log'),  # Log to a file
                        logging.StreamHandler()  # Also log to the console
                    ])

# Default parameters
DEFAULT_DATABASE_FOLDER = os.getenv("DATABASE_FOLDER", "./random_circuit_matrixs")
DEFAULT_ENGINE = os.getenv("NICSLU_ENGINE", "./nicslu_kernel.o")
DEFAULT_TIMEOUT = int(os.getenv("TIMEOUT", 100))
MAKEFILE_DIR = Path("lib/nicslu/src")  # Path to the Makefile directory

def run_makefile():
    """
    Runs the makefile in the specified src directory.
    """
    makefile_path = MAKEFILE_DIR / "Makefile"

    if not makefile_path.exists():
        logging.error(f"Makefile not found at {makefile_path}")
        raise FileNotFoundError(f"Makefile not found at {makefile_path}")

    logging.info(f"Running makefile at {makefile_path}...")
    try:
        # Run the 'make' command in the specified directory
        subprocess.run(["make", "-C", str(MAKEFILE_DIR)], check=True)
        logging.info("Makefile executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running Makefile: {e}")
        raise

def initialize_nicslu(database_folder=None, engine=None, timeout=None):
    """
    Initializes the NICSLU Processor with default or custom parameters
    and runs the matrix processing.
    """
    # Set parameters, use defaults if not provided
    database_folder = database_folder or DEFAULT_DATABASE_FOLDER
    engine = engine or DEFAULT_ENGINE
    timeout = timeout or DEFAULT_TIMEOUT

    logging.info(f"Initializing NICSLU Processor with parameters:\n"
                 f"Database Folder: {database_folder}\n"
                 f"Engine: {engine}\n"
                 f"Timeout: {timeout}")

    # Create the processor instance
    processor = NicsluProcessor(database_folder=database_folder, engine=engine, timeout=timeout)

    # Return the initialized processor
    return processor

# Run makefile when the package is imported
try:
    run_makefile()
except Exception as e:
    logging.error(f"Failed to run makefile: {str(e)}")

# Example of use: Automatically initialize the processor with defaults
nicslu_processor = initialize_nicslu()

# Now the processor can be used elsewhere in the code
