# Compiler settings
CC = g++
CXX = g++

# Library paths and linking settings
CLIB = -L./src/build/lib -lsuperlu -lblas -lm
INCL = -I./src/build/include -I./src/SRC -I./include

# Compiler flags
# -g includes debugging information.
# -O0 disables optimization for easier debugging.
STD = -std=c++17 -march=native -mtune=native -g -O0

# Clean command for removing build artifacts
CLEAN = rm -rf *.o *.a superlu_kernel

# Build targets
.PHONY: library clean all

# Build SuperLU libraries and dependencies
library:
	@echo "Building SuperLU libraries..."
	(cd src/build && $(MAKE))

# Source files
SRC = superlu_kernel.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Default target: build SuperLU libraries and the program
all: superlu_kernel

# Rule for building the superlu_kernel program
superlu_kernel: $(OBJ) library
	@echo "Compiling superlu_kernel..."
	$(CXX) $(INCL) $(STD) $(OBJ) -o superlu_kernel $(CLIB)

# Clean target
clean:
	@echo "Cleaning up..."
	rm -f $(OBJ) superlu_kernel
	$(CLEAN)

# Phony targets to avoid conflicts with files named "clean" or "all"
.PHONY: all clean
