import os
from lib.random_circuit_generator.generator_wrapper import run_circuit_generator_script, setup_output_directory


def main():
    # Define parameters for the circuit generation
    iterations = 10000  # Set to 10,000 to generate 10,000 matrices
    node_range_1 = [5, 100, 5]  # Customize these ranges as needed
    node_range_2 = [100, 2000, 20]  # Customize these ranges as needed
    output_dir = "data/circuit_data/raw"

    # Ensure the output directory exists
    setup_output_directory(output_dir)

    # Path to the circuit generation script (modify if the path is different)
    script_path = "lib/random_circuit_generator/utils/circuit_generation_script.py"

    # Run the circuit generator with the specified parameters
    run_circuit_generator_script(script_path, iterations, node_range_1, node_range_2, output_dir)


if __name__ == "__main__":
    main()
