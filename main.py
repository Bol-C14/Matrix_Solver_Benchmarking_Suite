import os
from lib.random_circuit_generator.circuit_matrix_generate import run_circuit_generation


def main():
    # Define the parameters for circuit generation
    iterations = 2  # Number of iterations for circuit generation
    node_range_1 = (5, 100, 5)  # First node range: (start, end, step)
    node_range_2 = (100, 5000, 50)  # Second node range: (start, end, step)

    # Set the output directory to save the generated matrices
    output_dir = "data/circuit_data/raw"

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Run the circuit generation process
    run_circuit_generation(
        node_of_iterations=iterations,
        node_range_1=node_range_1,
        node_range_2=node_range_2,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
