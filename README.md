# Matrix Benchmarking Suite

This repository provides a framework for automatically generating or downloading matrix datasets and benchmarking various numerical solvers. The suite supports multiple solvers, including KLU, NICSLU, GLU, PARDISO, and SuperLU, and allows users to test their performance on different types of matrices.

## Features
- **Matrix Sources**:
  - **SuiteSparse**: Download and organize matrices from the SuiteSparse collection.
  - **Random Circuit Generator**: Generate matrices based on circuit simulation principles.
  - **Pure Random Generator**: Create structured and sparse matrices with customizable parameters.
- **Benchmarking**:
  - Supports multiple solvers including KLU, NICSLU, GLU, PARDISO, and SuperLU.
  - Customizable matrix generation settings for benchmarking experiments.
  - Logs benchmarking results for performance evaluation.
- **Logging**:
  - Console and file logging for process tracking and debugging.
  - Supports configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.7+
- NumPy
- SuiteSparse (if using SuiteSparse matrices)
- Required solver dependencies (KLU, NICSLU, GLU, PARDISO, SuperLU)

### Setup
Clone the repository and install dependencies:
```sh
$ git clone <repository-url>
$ cd <repository-folder>
$ pip install -r requirements.txt
```

## Compilation
Although the solvers are designed to be pipelined, compilation may fail due to environmental differences. If this occurs, you may need to manually compile the C++ library for each solver. Navigate to `lib/{target_solver}` and manually compile the solver using the provided `Makefile` or relevant build scripts.

## Benchmark Results
Benchmark results will be stored as CSV files. The structure of the CSV output varies based on the solver used, as different solvers may support multi-threading and have unique performance metrics. An example CSV output is shown below:

```
index,mtx,algorithmname,threads,nnz,rows,Analyze,Factorization
0,random_circuit_5_1,KLU,1,315,69,10.5,18.26
1,random_circuit_5_2,KLU,1,464,98,15.77,28.03
2,random_circuit_5_3,KLU,1,297,65,10.0,18.55
3,random_circuit_5_4,KLU,1,417,87,16.34,25.62
4,random_circuit_5_5,KLU,1,224,55,7.31,13.32
5,random_circuit_5_6,KLU,1,204,45,6.63,11.65
6,random_circuit_5_7,KLU,1,363,77,14.77,22.84
7,random_circuit_5_8,KLU,1,250,54,8.25,15.26
```

Different solvers may have additional or different columns depending on their capabilities and execution modes.

## Usage
### Generating or Downloading Matrices
Run the main script with the desired matrix source:

- **Use SuiteSparse Data**:
  ```sh
  python main.py --use_suite_sparse
  ```
- **Generate Matrices using Circuit Generator**:
  ```sh
  python main.py --use_circuit
  ```
- **Generate Matrices using Pure Random Generator**:
  ```sh
  python main.py --use_pure_random
  ```
- **Download Only (SuiteSparse)**:
  ```sh
  python main.py --use_suite_sparse --download_only
  ```

### Running Benchmarks
Benchmarks will automatically run after generating/downloading matrices. To run a specific solver, modify `main.py` or call:

```sh
python main.py --use_circuit
python main.py --use_pure_random
python main.py --use_suite_sparse
```

This will execute all solvers: KLU, NICSLU, GLU, PARDISO, and SuperLU.

## Directory Structure
```
├── main.py                  # Entry point for matrix generation and benchmarking
├── debug.log                # Debug logs
├── data/                    # Directory for all generated/downloaded matrices
│   ├── suite_sparse_data/   # Matrices from SuiteSparse
│   ├── circuit_data/        # Generated circuit matrices
│   ├── ss_organized_data/   # Processed SuiteSparse matrices
│   ├── generated_matrices/  # Randomly generated matrices
├── lib/                     # Libraries for solvers and generators
│   ├── klu_new/             # KLU solver
│   ├── nicslu/              # NICSLU solver
│   ├── glu/                 # GLU solver
│   ├── pardiso/             # PARDISO solver
│   ├── superlu/             # SuperLU solver
│   ├── neural_network_generator/  # Neural network-based generator (future work)
├── utils/                   # Utility scripts
│   ├── suitesparse_handler.py # SuiteSparse manager
│   ├── ...                  # Additional utility files
```

## Future Work
- Implement a neural network-based matrix generator.
- Enhance matrix organization and metadata storage.
- Extend benchmarking support to additional solvers.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.

