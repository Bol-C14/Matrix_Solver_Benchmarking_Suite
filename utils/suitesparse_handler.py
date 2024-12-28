import os
import tarfile
import logging
from pathlib import Path
import shutil
import ssgetpy
import sys
import threading


class SuiteSparseManager:
    def __init__(self, suite_sparse_dir: Path, organized_dir: Path, logger: logging.Logger):
        """
        Initialize the SuiteSparseManager with directories and logger.

        :param suite_sparse_dir: Path to the directory where SuiteSparse data is downloaded.
        :param organized_dir: Path to the directory where organized .mtx files will be stored.
        :param logger: Logger instance for logging information.
        """
        self.suite_sparse_dir = suite_sparse_dir
        self.organized_dir = organized_dir
        self.logger = logger

    def prepare_matrices(self):
        """
        Complete routine to download, extract, verify, and organize SuiteSparse matrices.
        """
        self.logger.info("Starting SuiteSparse matrices preparation process.")

        if not self.check_existing_data():
            self.download_matrices()
            self.extract_all_tar_gz()
            self.verify_mtx_files()
            self.organize_mtx_files()
        else:
            self.logger.info("Existing SuiteSparse data detected. Skipping download and extraction.")

        self.logger.info("SuiteSparse matrices preparation completed successfully.")

    def check_existing_data(self) -> bool:
        """
        Check if the target directory already contains SuiteSparse data.

        :return: True if data exists and user chooses to skip downloading; False otherwise.
        """
        if any(self.suite_sparse_dir.iterdir()):
            self.logger.warning(f"The directory {self.suite_sparse_dir} already contains files/folders.")
            # List some files/folders
            existing_items = list(self.suite_sparse_dir.iterdir())
            items_to_show = existing_items[:5]  # Show up to 5 items
            self.logger.info("Existing items:")
            for item in items_to_show:
                self.logger.info(f" - {item.name}")
            if len(existing_items) > 5:
                self.logger.info(f"and {len(existing_items) - 5} more...")

            # Prompt user
            user_input = self.prompt_user(
                prompt=f"Do you want to skip downloading SuiteSparse matrices? [Y/n]: ",
                timeout=10
            )
            if user_input.lower() in ['y', 'yes', '']:
                self.logger.info("Skipping download and extraction of SuiteSparse matrices.")
                return True
            else:
                self.logger.info("Proceeding with download and extraction of SuiteSparse matrices.")
                return False
        else:
            self.logger.info(f"The directory {self.suite_sparse_dir} is empty. Proceeding with download.")
            return False

    def prompt_user(self, prompt: str, timeout: int) -> str:
        """
        Prompt the user for input with a timeout.

        :param prompt: The prompt message to display.
        :param timeout: Timeout in seconds.
        :return: User input as a string, or empty string if timeout occurs.
        """
        # Use print for interactive prompt messages
        print(prompt, end='', flush=True)
        user_input = []

        def get_input():
            try:
                inp = input()
                user_input.append(inp)
            except EOFError:
                pass  # Handle end-of-file condition

        thread = threading.Thread(target=get_input)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            print("\nNo input received within timeout. Proceeding with default action (Skip).")
            return ''
        return user_input[0] if user_input else ''


    def download_matrices(self):
        """
        Download SuiteSparse matrices using ssgetpy and store them in suite_sparse_dir.
        """
        self.logger.info("Starting download of SuiteSparse matrices.")
        # Create a directory to store downloaded matrices
        self.suite_sparse_dir.mkdir(parents=True, exist_ok=True)

        # Fetch all matrices with a very high limit
        self.logger.info("Fetching all matrices from SuiteSparse...")
        try:
            matrices = ssgetpy.search(limit=1000000)  # Use a very large limit instead of None
        except Exception as e:
            self.logger.error(f"Failed to fetch matrices using ssgetpy: {e}")
            sys.exit(1)
        
        self.logger.info(f"Total matrices found: {len(matrices)}")

        # Iterate through each matrix
        for idx, matrix in enumerate(matrices, 1):
            try:
                # Replace spaces with underscores for valid directory names
                kind = matrix.kind.replace(' ', '_') if matrix.kind else "unknown_kind"
                kind_dir = self.suite_sparse_dir / kind
                kind_dir.mkdir(parents=True, exist_ok=True)

                # Define the path for the Matrix Market file
                mtx_path = kind_dir / f"{matrix.name}.mtx"

                # Download the matrix in Matrix Market format if not already downloaded
                if not mtx_path.exists():
                    self.logger.info(f"Downloading ({idx}/{len(matrices)}): {matrix.name} of kind {matrix.kind}...")
                    try:
                        matrix.download(format='MM', destpath=str(kind_dir))
                        self.logger.debug(f"Downloaded {matrix.name} to {mtx_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to download {matrix.name}: {e}")
                else:
                    self.logger.debug(f"{matrix.name} already exists. Skipping download.")
            except Exception as e:
                self.logger.error(f"Error processing {matrix.name}: {e}")

        self.logger.info("SuiteSparse matrices download completed.")

    def extract_all_tar_gz(self):
        """
        Extract all .tar.gz files in the suite_sparse_dir.
        """
        self.logger.info(f"Starting extraction of .tar.gz files in {self.suite_sparse_dir}")
        # Use glob to find all .tar.gz files recursively
        tar_gz_files = list(self.suite_sparse_dir.rglob("*.tar.gz"))
        self.logger.info(f"Found {len(tar_gz_files)} .tar.gz files to extract.")

        for tar_gz_path in tar_gz_files:
            # Define the extraction directory (same as tar.gz file without .tar.gz)
            extraction_dir = tar_gz_path.parent / tar_gz_path.stem
            if not extraction_dir.exists():
                extraction_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Extracting {tar_gz_path} to {extraction_dir}")
                try:
                    with tarfile.open(tar_gz_path, "r:gz") as tar:
                        tar.extractall(path=extraction_dir)
                    self.logger.info(f"Extracted {tar_gz_path} successfully.")
                except (tarfile.TarError, EOFError) as e:
                    self.logger.error(f"Failed to extract {tar_gz_path}: {e}")
            else:
                self.logger.debug(f"Extraction directory {extraction_dir} already exists. Skipping extraction.")

        self.logger.info("All .tar.gz files have been processed.")

    def verify_mtx_files(self):
        """
        Verify that .mtx files are present in suite_sparse_dir after extraction.
        """
        self.logger.info(f"Verifying .mtx files in {self.suite_sparse_dir}")
        mtx_files = list(self.suite_sparse_dir.rglob("*.mtx"))
        self.logger.info(f"Found {len(mtx_files)} .mtx files.")
        if not mtx_files:
            self.logger.warning("No .mtx files found. Please check the extraction process.")
        else:
            self.logger.info("All .mtx files are present.")

    def remove_mtx_header(self, mtx_file_path: Path) -> str:
        """
        Remove headers from the .mtx file, retain necessary metadata, 
        and return the cleaned content with a Matrix Market header.

        :param mtx_file_path: Path to the .mtx file.
        :return: Cleaned content as a string.
        """
        with open(mtx_file_path, 'r') as file:
            lines = file.readlines()

        # Identify and retain the first line starting with '%%MatrixMarket'
        matrix_market_header = None
        for line in lines:
            if line.startswith('%%MatrixMarket'):
                matrix_market_header = line.strip()
                break

        # If no Matrix Market header is found, set a default
        if not matrix_market_header:
            matrix_market_header = "%%MatrixMarket matrix coordinate real general"

        # Skip all comment lines (starting with '%') and retain data lines
        data_lines = [line for line in lines if not line.startswith('%')]

        # Combine the retained Matrix Market header with the data lines
        cleaned_content = f"{matrix_market_header}\n" + ''.join(data_lines)

        return cleaned_content

    def organize_mtx_files(self):
        """
        Organize .mtx files by removing headers and copying them to organized_dir.
        """
        self.logger.info(f"Organizing .mtx files into {self.organized_dir}")
        
        # Find all extracted directories under suite_sparse_dir
        extracted_dirs = [p for p in self.suite_sparse_dir.rglob("*") if p.is_dir()]
        self.logger.info(f"Found {len(extracted_dirs)} directories to search for .mtx files.")

        # Iterate through directories to find the main .mtx files
        for directory in extracted_dirs:
            # Search for .mtx files in the directory
            mtx_files = list(directory.glob("*.mtx"))
            self.logger.debug(f"Found {len(mtx_files)} .mtx files in {directory}")

            for mtx_file in mtx_files:
                # Skip files with '_x.mtx' or '_b.mtx' suffix
                if mtx_file.name.endswith(('_x.mtx', '_b.mtx')):
                    self.logger.debug(f"Skipping auxiliary file: {mtx_file}")
                    continue

                # Prepare the destination path
                destination = self.organized_dir / mtx_file.name
                if not destination.exists():
                    try:
                        # Clean the .mtx file by removing its header
                        self.logger.debug(f"Cleaning headers for {mtx_file}")
                        cleaned_data = self.remove_mtx_header(mtx_file)

                        # Write the cleaned data to the destination
                        with open(destination, 'w') as cleaned_file:
                            cleaned_file.write(cleaned_data)
                        
                        self.logger.info(f"Cleaned and copied {mtx_file} to {destination}")
                    except IOError as e:
                        self.logger.error(f"Failed to process {mtx_file}: {e}")
                else:
                    self.logger.debug(f"File {destination} already exists. Skipping copy.")

        self.logger.info(f"Organized .mtx files are available in {self.organized_dir}")
