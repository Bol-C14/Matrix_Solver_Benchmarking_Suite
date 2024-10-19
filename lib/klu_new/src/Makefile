# Compiler settings
CC = g++
CXX = g++

# Library paths and linking settings
CLIB = -L./SuiteSparse-stable/KLU -lklu -lamd -lbtf -lsuitesparseconfig -lm
INCL = -I./include -I./SuiteSparse-stable/KLU/include -I./SuiteSparse-stable/AMD/Include -I./SuiteSparse-stable/BTF/Include -I./SuiteSparse-stable/COLAMD/Include -I./SuiteSparse-stable/SuiteSparse_config

# Compiler flags
# -g includes debugging information.
# -O0 disables optimization for easier debugging.
STD = -std=c++17 -march=native -mtune=native -g -O0
STD_ARM = -std=c++17 -mcpu=native -g -O0

# Clean command for removing build artifacts
CLEAN = rm -rf *.o *.a

# Build targets
.PHONY: library clean all

# Build the SuiteSparse libraries and dependencies
library:
	@echo "Building SuiteSparse libraries..."
	(cd SuiteSparse-stable/SuiteSparse_config && $(MAKE))
	(cd SuiteSparse-stable/AMD && $(MAKE))
	(cd SuiteSparse-stable/BTF && $(MAKE))
	(cd SuiteSparse-stable/COLAMD && $(MAKE))
	(cd SuiteSparse-stable/KLU && $(MAKE))

# Rule for building klu_kernel with debugging symbols
klu_kernel: klu_kernel.cpp library
	@echo "Compiling klu_kernel..."
	$(CXX) $(INCL) $(STD) klu_kernel.cpp -o klu_kernel.o $(CLIB)

# Rule for ARM build with debugging symbols
klu_kernel_arm: klu_kernel.cpp library
	@echo "Compiling klu_kernel for ARM..."
	$(CXX) $(INCL) $(STD_ARM) klu_kernel.cpp -o klu_kernel_arm.o $(CLIB)

# Rule for building klu_threaded with debugging symbols
klu_threaded: klu_threaded.cpp library
	@echo "Compiling klu_threaded..."
	$(CXX) $(INCL) $(STD) klu_threaded.cpp -o klu_threaded.o $(CLIB) -pthread

# Rule for ARM build with debugging symbols
klu_threaded_arm: klu_threaded.cpp library
	@echo "Compiling klu_threaded for ARM..."
	$(CXX) $(INCL) $(STD_ARM) klu_threaded.cpp -o klu_threaded_arm.o $(CLIB) -pthread

# Build both versions for x86
both: klu_kernel klu_threaded

# Build both versions for ARM
both_arm: klu_kernel_arm klu_threaded_arm

# Clean all generated object files and libraries
clean:
	@echo "Cleaning up..."
	rm -rf SuiteSparse-stable/SuiteSparse_config/*.o
	rm -rf SuiteSparse-stable/AMD/Lib/*.o
	rm -rf SuiteSparse-stable/BTF/Lib/*.o
	rm -rf SuiteSparse-stable/KLU/Lib/*.o
	rm -rf ./lib/*.a
	rm -rf klu_kernel klu_kernel_arm klu_threaded klu_threaded_arm *.o
