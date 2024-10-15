import os
import logging
from lib.random_circuit_generator.circuit_matrix_generate import run_circuit_generation, setup_logging

def main():
    # Define the parameters for circuit generation
    iterations = 10  # Increase iterations to cover more matrices
    node_range_1 = (5, 200, 5)  # First node range: (start, end, step)
    node_range_2 = (200, 10000, 100)  # Second node range: (start, end, step)

    # Set the output directory to save the generated matrices
    output_dir = "data/circuit_data/raw"

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging configuration
    log_file = "circuit_generation.log"
    setup_logging(log_file)  # Set up logging to file and console

    # Run the circuit generation process with 8 threads
    run_circuit_generation(
        node_of_iterations=iterations,
        node_range_1=node_range_1,
        node_range_2=node_range_2,
        output_dir=output_dir,
    )




    # using klu to solve the matrices

if __name__ == "__main__":
    main()
