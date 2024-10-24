# Compiler settings
CC = g++
CXX = g++

# Paths
SUPERLU_DIR = ./src/superlu_kernel
BUILD_DIR = $(SUPERLU_DIR)/build
INCLUDE_DIRS = -I$(BUILD_DIR)/include -I$(SUPERLU_DIR)/SRC -I./include

# Library paths and linking settings
LIB_DIRS = -L$(BUILD_DIR)/lib
LIBS = -lsuperlu -lblas -lm

# Compiler flags
CXXFLAGS = -std=c++17 -g -O0

# Source files
SRC = superlu_kernel.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Default target: build SuperLU libraries and the program
all: library superlu_kernel

# Build SuperLU libraries
library:
	@echo "Building SuperLU libraries..."
	$(MAKE) -C $(BUILD_DIR)

# Rule for building the superlu_kernel program
superlu_kernel: $(OBJ)
	@echo "Compiling superlu_kernel..."
	$(CXX) $(INCLUDE_DIRS) $(CXXFLAGS) $(OBJ) -o $@ $(LIB_DIRS) $(LIBS)

# Rule for building object files
%.o: %.cpp
	$(CXX) $(INCLUDE_DIRS) $(CXXFLAGS) -c $< -o $@

# Clean target
clean:
	@echo "Cleaning up..."
	rm -f $(OBJ) superlu_kernel
	$(MAKE) -C $(BUILD_DIR) clean

# Phony targets to avoid conflicts with files named "clean" or "all"
.PHONY: all clean library
