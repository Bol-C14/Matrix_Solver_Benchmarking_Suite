# Define MKLROOT if not already defined
MKLROOT ?= /opt/intel/oneapi/mkl/latest

# Set include and library paths
INCLUDE = $(MKLROOT)/include
LIB = $(MKLROOT)/lib/intel64

# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -m64 -I$(INCLUDE) -std=c++11

# MKL libraries
MKL_LIBS = -Wl,--start-group $(LIB)/libmkl_intel_lp64.a $(LIB)/libmkl_gnu_thread.a $(LIB)/libmkl_core.a $(LIB)/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lgomp -lpthread -lm -ldl

# Target executable
TARGET = pardiso_kernel.o

# Build rule
all: $(TARGET)

$(TARGET): pardiso_kernel.cpp
	$(CXX) $(CXXFLAGS) pardiso_kernel.cpp -o $(TARGET) $(MKL_LIBS)

# Clean rule
clean:
	rm -f $(TARGET)
