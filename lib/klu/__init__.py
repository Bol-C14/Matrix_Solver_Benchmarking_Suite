# klu/__init__.py
import subprocess
import logging
from pathlib import Path

def compile_cpp():
    makefile_path = Path(__file__).parent / "src/Makefile"  # Path to your Makefile
    try:
        logging.info(f"Compiling C++ code using Makefile at {makefile_path}")
        # Run 'make klu_kernel' command
        subprocess.run(["make", "klu_kernel", "-C", makefile_path.parent], check=True)
        logging.info("Compilation successful.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Compilation failed: {e}")
        raise

# Automatically compile the C++ code when the package is imported
compile_cpp()

from .src.run_klu_kernels import KluBenchmark
