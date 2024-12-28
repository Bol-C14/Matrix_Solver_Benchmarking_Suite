import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional
import numpy as np
import random
from scipy.sparse import csr_matrix, diags
from scipy.io import mmwrite
import math
import time

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("matrix_generator.log")
    ]
)
logger = logging.getLogger(__name__)

class MatrixShape(Enum):
    """Supported matrix shape types."""
    RANDOM = auto()
    DIAGONAL = auto()
    BANDED = auto()
    SPARSELY_RANDOM = auto()
    POSITIVE_DEFINITE = auto()  # New shape for positive definite matrices

@dataclass
class SizeConfig:
    """Configuration for matrix size generation."""
    min_size: int
    max_size: int
    size_step: Optional[int] = None
    random_size: bool = False

@dataclass
class SparsityConfig:
    """Configuration for matrix sparsity."""
    min_sparsity: float
    max_sparsity: float
    min_density: float = 0.1  # New parameter to set a minimum density
    random_sparsity: bool = False
    enable_range_generation: bool = False
    range_start: float = 0.8  # Align with min_sparsity
    range_end: float = 0.95   # Align with max_sparsity
    num_steps: int = 5
    decrease_with_size: bool = True
    random_decrease: bool = False
    decrease_rate: float = 0.1

@dataclass
class ShapeConfig:
    """Configuration for matrix shape generation."""
    shapes: List[MatrixShape]
    probabilities: Optional[List[float]] = None
    attempts_per_shape: int = 1

@dataclass
class ValueConfig:
    """Configuration for matrix values."""
    min_val: int
    max_val: int

@dataclass
class MatrixGeneratorConfig:
    """Main configuration for matrix generation."""
    size: SizeConfig
    sparsity: SparsityConfig
    shape: ShapeConfig
    value: ValueConfig
    seed: Optional[int] = None
    repeat: int = 1  # New parameter to define repeat count
    max_attempts: int = 5000
    attempt_logging_frequency: int = 100

@dataclass
class OutputConfig:
    """Configuration for output handling."""
    output_dir: Path
    file_prefix: str = "matrix"
    file_extension: str = "mtx"

class MatrixGenerationError(Exception):
    """Custom exception for matrix generation errors."""
    pass

class SparsityCalculator:
    """Handles sparsity calculations and adjustments."""

    @staticmethod
    def calculate_sparsity_range(config: SparsityConfig) -> List[float]:
        if not config.enable_range_generation:
            return []
        linear_space = np.linspace(config.range_start, config.range_end, config.num_steps)
        return list(linear_space)

    @staticmethod
    def adjust_sparsity_for_size(base_sparsity: float, initial_size: int, current_size: int, config: SparsityConfig) -> float:
        if not config.decrease_with_size:
            return base_sparsity
        size_ratio = current_size / initial_size
        decrease_rate = config.decrease_rate * (0.5 + random.random()) if config.random_decrease else config.decrease_rate
        # Using linear adjustment instead of logarithmic
        adjusted_sparsity = base_sparsity - (decrease_rate * size_ratio)
        return max(config.range_start, min(adjusted_sparsity, config.range_end))

class MatrixValidator:
    """Handles matrix validation logic."""

    @staticmethod
    def is_valid(matrix: np.ndarray) -> bool:
        try:
            rank = np.linalg.matrix_rank(matrix)
            if rank < matrix.shape[0]:
                logger.debug(f"Matrix rank {rank} is less than size {matrix.shape[0]}.")
                return False
            cond_number = np.linalg.cond(matrix)
            if cond_number > 1e10:  # Example threshold for condition number
                logger.debug(f"Matrix condition number {cond_number} exceeds threshold.")
                return False
            return True
        except np.linalg.LinAlgError as e:
            logger.debug(f"LinAlgError during validation: {e}")
            return False

class MatrixGenerator:
    """Handles generation of matrices with various properties."""

    def __init__(self, config: MatrixGeneratorConfig):
        self.config = config
        self._setup_random_seed()
        self._current_step = 0
        self._initial_size = config.size.min_size
        self.sparsity_calculator = SparsityCalculator()

    def _setup_random_seed(self, seed: Optional[int] = None):
        """Setup random seed for reproducibility."""
        actual_seed = seed if seed is not None else self.config.seed
        if actual_seed is not None:
            np.random.seed(actual_seed)
            random.seed(actual_seed)


    def get_next_size(self) -> int:
        if self.config.size.random_size:
            return random.randint(self.config.size.min_size, self.config.size.max_size)
        if self.config.size.size_step:
            size = self.config.size.min_size + self._current_step * self.config.size.size_step
            if size > self.config.size.max_size:
                raise StopIteration
            self._current_step += 1
            return size
        return self.config.size.min_size

    def get_sparsity(self, size: int) -> List[float]:
        if self.config.sparsity.enable_range_generation:
            base_sparsities = self.sparsity_calculator.calculate_sparsity_range(self.config.sparsity)
        elif self.config.sparsity.random_sparsity:
            base_sparsities = [random.uniform(self.config.sparsity.min_sparsity, self.config.sparsity.max_sparsity)]
        else:
            base_sparsities = [self.config.sparsity.min_sparsity]

        adjusted_sparsities = [
            self.sparsity_calculator.adjust_sparsity_for_size(s, self._initial_size, size, self.config.sparsity)
            for s in base_sparsities
        ]
        # Ensure sparsity is within the specified range
        adjusted_sparsities = [max(self.config.sparsity.min_sparsity, min(s, self.config.sparsity.max_sparsity)) for s in adjusted_sparsities]

        logger.debug(f"Sparsity values for size {size}: {adjusted_sparsities}")
        return adjusted_sparsities


    def choose_shape(self) -> MatrixShape:
        if len(self.config.shape.shapes) == 1:
            return self.config.shape.shapes[0]
        return random.choices(self.config.shape.shapes, weights=self.config.shape.probabilities, k=1)[0]

    def _generate_random_matrix(self, size: int, density: float) -> np.ndarray:
        matrix = np.zeros((size, size), dtype=float)
        num_elements = int(density * size * size)
        # Randomly choose positions for non-zero off-diagonal elements
        rows = np.random.randint(0, size, num_elements)
        cols = np.random.randint(0, size, num_elements)
        mask = rows != cols  # Exclude diagonal
        rows, cols = rows[mask], cols[mask]
        values = np.random.randint(self.config.value.min_val, self.config.value.max_val, size=rows.size)
        matrix[rows, cols] = values
        # Set diagonal elements to ensure diagonal dominance
        for i in range(size):
            row_sum = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
            matrix[i, i] = row_sum + np.random.randint(1, self.config.value.max_val)
        return matrix

    def _generate_diagonal_matrix(self, size: int, density: float) -> np.ndarray:
        # Diagonal matrix inherently has density of (1 / size)
        matrix = np.zeros((size, size), dtype=float)
        diagonal_values = np.random.randint(max(1, self.config.value.min_val), self.config.value.max_val, size)
        np.fill_diagonal(matrix, diagonal_values)
        # Optionally add a few off-diagonal elements to increase density slightly
        if density > (1.0 / size):
            num_off_diag = int(density * size) - size  # Approximate number of off-diagonal elements
            if num_off_diag > 0:
                for _ in range(num_off_diag):
                    i, j = random.randint(0, size-1), random.randint(0, size-1)
                    if i != j:
                        matrix[i, j] = random.randint(self.config.value.min_val, self.config.value.max_val)
                        # Adjust diagonal to maintain diagonal dominance
                        matrix[i, i] += abs(matrix[i, j])
        return matrix

    def _generate_banded_matrix(self, size: int, density: float) -> np.ndarray:
        bandwidth = max(1, int(density * size / 2))  # Adjust bandwidth based on density
        matrix = np.zeros((size, size), dtype=float)
        for i in range(size):
            lower = max(0, i - bandwidth)
            upper = min(size, i + bandwidth + 1)
            for j in range(lower, upper):
                if i != j:
                    matrix[i, j] = random.randint(self.config.value.min_val, self.config.value.max_val)
            # Set diagonal element to ensure diagonal dominance
            row_sum = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
            matrix[i, i] = row_sum + random.randint(1, self.config.value.max_val)
        return matrix

    def _generate_positive_definite_matrix(self, size: int, density: float) -> np.ndarray:
        # Generate a random sparse matrix B
        B = np.zeros((size, size), dtype=float)
        num_elements = int(density * size * size)
        rows = np.random.randint(0, size, num_elements)
        cols = np.random.randint(0, size, num_elements)
        mask = rows != cols  # Exclude diagonal for B
        rows, cols = rows[mask], cols[mask]
        values = np.random.randint(self.config.value.min_val, self.config.value.max_val, size=rows.size)
        B[rows, cols] = values
        # Construct A = B^T B + D to ensure positive definiteness
        A = np.dot(B.T, B)
        # Add a diagonal matrix D to ensure numerical stability and positive definiteness
        diagonal = np.random.randint(1, self.config.value.max_val, size)
        np.fill_diagonal(A, A.diagonal() + diagonal)
        return A

    def _generate_sparcely_random_matrix(self, size: int, density: float) -> np.ndarray:
        # Similar to random matrix but with much lower density
        return self._generate_random_matrix(size, density * 0.1)

    def make_diagonally_dominant(self, matrix: np.ndarray) -> np.ndarray:
        """Modify the matrix to make it diagonally dominant."""
        for i in range(matrix.shape[0]):
            row_sum = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
            if row_sum >= np.abs(matrix[i, i]):
                matrix[i, i] = row_sum + 1  # Make diagonal element larger than the sum of the row
        return matrix

    def generate_matrix(self, size: int, sparsity: float, shape: MatrixShape) -> np.ndarray:
        density = 1.0 - sparsity
        generators = {
            MatrixShape.RANDOM: lambda: self._generate_random_matrix(size, density),
            MatrixShape.DIAGONAL: lambda: self._generate_diagonal_matrix(size, density),
            MatrixShape.BANDED: lambda: self._generate_banded_matrix(size, density),
            MatrixShape.SPARSELY_RANDOM: lambda: self._generate_sparcely_random_matrix(size, density),
            MatrixShape.POSITIVE_DEFINITE: lambda: self._generate_positive_definite_matrix(size, density)
        }
        generator = generators.get(shape)
        if not generator:
            raise MatrixGenerationError(f"Unsupported shape: {shape}")
        return generator()

    def generate_valid_matrix(self, size: int, sparsity: float, shape: MatrixShape, attempt_count: int = 1) -> Optional[np.ndarray]:
        best_matrix = None
        best_condition_number = float('inf')
        for attempt in range(attempt_count):
            start_time = time.time()
            for attempt_num in range(1, self.config.max_attempts + 1):
                try:
                    if attempt_num % self.config.attempt_logging_frequency == 0:
                        logger.info(f"Attempt {attempt_num}: Generating matrix with size={size}, sparsity={sparsity:.6f}, shape={shape.name}")
                    matrix = self.generate_matrix(size, sparsity, shape)
                    # For positive definite matrices, they are already non-singular
                    if shape != MatrixShape.POSITIVE_DEFINITE:
                        matrix = self.make_diagonally_dominant(matrix)
                    if MatrixValidator.is_valid(matrix):
                        condition_number = np.linalg.cond(matrix)
                        logger.info(f"Generated valid matrix (condition number={condition_number:.2f}).")
                        if condition_number < best_condition_number:
                            best_matrix = matrix
                            best_condition_number = condition_number
                        break  # Valid matrix found, proceed to next
                    else:
                        logger.debug("Matrix failed validation.")
                except np.linalg.LinAlgError as e:
                    logger.debug(f"LinAlgError during generation: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
            elapsed_time = time.time() - start_time
            logger.info(f"Time taken for attempt {attempt + 1}: {elapsed_time:.2f} seconds.")
        if best_matrix is None:
            logger.error(f"Failed to generate valid matrix after {attempt_count} attempts (size={size}, sparsity={sparsity:.2f}, shape={shape.name})")
        return best_matrix

class MatrixSaver:
    """Handles saving matrices to files."""

    def __init__(self, config: OutputConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 1

    def save_matrix(self, matrix: np.ndarray, seed: int, iteration: int) -> Optional[Path]:
        if matrix is None:
            return None
        sparse_matrix = csr_matrix(matrix)
        filepath = self.config.output_dir / f"{self.config.file_prefix}_{self._counter}_seed{seed}_iter{iteration}.{self.config.file_extension}"
        mmwrite(filepath, sparse_matrix)
        logger.info(f"Saved matrix {self._counter} to {filepath}")
        self._counter += 1
        return filepath


def generate_matrix_batch(generator: MatrixGenerator, saver: MatrixSaver) -> List[Path]:
    """Generates matrices for the specified size range and repeats."""
    saved_paths = []
    initial_seed = generator.config.seed

    for iteration in range(generator.config.repeat):
        logger.info(f"Starting iteration {iteration + 1}/{generator.config.repeat}")
        generator._current_step = 0  # Reset size iteration for each repeat

        # Update seed for each iteration
        new_seed = initial_seed + iteration if initial_seed is not None else None
        generator._setup_random_seed(new_seed)

        try:
            while True:
                try:
                    size = generator.get_next_size()
                    sparsities = generator.get_sparsity(size)
                    shape = generator.choose_shape()
                    for sparsity in sparsities:
                        logger.info(f"Generating matrix with size={size}, sparsity={sparsity:.6f}, shape={shape.name}")
                        matrix = generator.generate_valid_matrix(
                            size, sparsity, shape, generator.config.shape.attempts_per_shape
                        )
                        if matrix is not None:
                            saved_path = saver.save_matrix(matrix, seed=new_seed, iteration=iteration + 1)
                            if saved_path:
                                saved_paths.append(saved_path)
                except StopIteration:
                    logger.info("Reached maximum size limit for this iteration.")
                    break
        except Exception as e:
            logger.error(f"Unexpected error during iteration {iteration + 1}: {e}")
    return saved_paths


if __name__ == "__main__":
    generator_config = MatrixGeneratorConfig(
        size=SizeConfig(min_size=100, max_size=500, size_step=100, random_size=False),
        sparsity=SparsityConfig(
            min_sparsity=0.8,
            max_sparsity=0.95,
            random_sparsity=False,
            enable_range_generation=True,
            range_start=0.8,    # Aligned with min_sparsity
            range_end=0.95,     # Aligned with max_sparsity
            num_steps=10,       # Increased for finer granularity
            decrease_with_size=True,
            random_decrease=False,
            decrease_rate=0.05   # Adjusted rate for smoother changes
        ),
        shape=ShapeConfig(
            shapes=[
                MatrixShape.RANDOM,
                MatrixShape.DIAGONAL,
                MatrixShape.BANDED,
                MatrixShape.SPARSELY_RANDOM,
                MatrixShape.POSITIVE_DEFINITE
            ],
            probabilities=[0.3, 0.2, 0.2, 0.2, 0.1],
            attempts_per_shape=3
        ),
        value=ValueConfig(min_val=-10, max_val=10),
        # seed=42,
        repeat=4  # Generate matrices for the range 4 times
    )

    output_config = OutputConfig(output_dir=Path("generated_matrices"), file_prefix="matrix", file_extension="mtx")
    generator = MatrixGenerator(generator_config)
    saver = MatrixSaver(output_config)
    saved_matrix_paths = generate_matrix_batch(generator, saver)
    logger.info(f"Generated and saved {len(saved_matrix_paths)} matrices.")
