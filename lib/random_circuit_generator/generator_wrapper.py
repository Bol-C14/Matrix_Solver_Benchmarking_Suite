import os
import argparse
import subprocess
import logging
import shutil
import sys

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('circuit_generator.log'),
                        logging.StreamHandler()
                    ])


def check_dependencies():
    """
    Check if necessary dependencies are installed (such as matplotlib, numpy, networkx).
    If not, prompt the user to install them.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import networkx as nx
        import scipy
    except ImportError as e:
        logging.error(f"Missing dependency: {e}. Please install the required libraries.")
        sys.exit(1)


def create_virtual_environment(venv_dir='venv'):
    """
    Creates and activates a virtual environment, and installs necessary dependencies.
    """
    if not os.path.exists(venv_dir):
        logging.info(f"Creating virtual environment in {venv_dir}...")
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
        activate_script = os.path.join(venv_dir, 'Scripts', 'activate' if os.name == 'nt' else 'bin/activate')
        install_dependencies(activate_script)
    else:
        logging.info(f"Using existing virtual environment in {venv_dir}...")


def install_dependencies(activate_script):
    """
    Install the necessary dependencies into the virtual environment.
    """
    logging.info("Installing dependencies...")
    requirements = ["matplotlib", "numpy", "networkx", "scipy"]
    subprocess.run(f"{activate_script} && pip install " + " ".join(requirements), shell=True, check=True)


def run_circuit_generator_script(script_path, iterations, node_range_1, node_range_2, output_dir):
    """
    Run the circuit generation script with the specified parameters.
    """
    cmd = [
        sys.executable, script_path,
        '--iterations', str(iterations),
        '--node_range_1', *map(str, node_range_1),
        '--node_range_2', *map(str, node_range_2),
        '--output_dir', output_dir
    ]
    logging.info(f"Running circuit generator with command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logging.info("Circuit generation completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during circuit generation: {e}")
        sys.exit(1)


def setup_output_directory(output_dir):
    """
    Sets up the output directory by ensuring it exists or creating it.
    If the directory already exists, prompt the user for action.
    """
    if os.path.exists(output_dir):
        user_input = input(f"Output directory {output_dir} already exists. Do you want to clear it? [y/N]: ").strip().lower()
        if user_input == 'y':
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            logging.info("Using the existing output directory.")
    else:
        os.makedirs(output_dir)


def parse_arguments():
    """
    Parse command-line arguments for configuring the circuit generation process.
    """
    parser = argparse.ArgumentParser(description="Portable Circuit Generator Wrapper")
    parser.add_argument('--iterations', type=int, default=2, help='Number of iterations for circuit generation')
    parser.add_argument('--node_range_1', type=int, nargs=3, default=[5, 100, 5], help='Node range 1 (start, end, step)')
    parser.add_argument('--node_range_2', type=int, nargs=3, default=[100, 2000, 20], help='Node range 2 (start, end, step)')
    parser.add_argument('--output_dir', type=str, default="generated_circuits", help='Output directory for matrix files')
    parser.add_argument('--script_path', type=str, default="lib/random_circuit_generator/utils/circuit_generation_script.py",
                        help='Path to the original circuit generation script')

    return parser.parse_args()


def main():
    """
    Main function that orchestrates the circuit generation process.
    """
    # Parse arguments
    args = parse_arguments()

    # Check if all dependencies are installed
    check_dependencies()

    # Setup output directory
    setup_output_directory(args.output_dir)

    # Create and activate virtual environment (optional but recommended for portability)
    create_virtual_environment()

    # Run the original circuit generation script with provided parameters
    run_circuit_generator_script(
        args.script_path,
        iterations=args.iterations,
        node_range_1=args.node_range_1,
        node_range_2=args.node_range_2,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
