CC=g++
CXX=g++

CLIB=-L lib -lklu -lamd -lbtf -lsuitesparseconfig -lm
INCL=-I include

STD=-std=c++17 -march=native -mtune=native
STD_ARM=-std=c++17 -mcpu=native

CLEAN=rm -rf *.o *.a

all: klu_kernel

library:
	(cd SuiteSparse_config ; $(MAKE))
	(cd AMD/Lib ; $(MAKE))
	(cd BTF/Lib ; $(MAKE))
	(cd KLU/Lib ; $(MAKE))

klu_kernel: klu_kernel.cpp library
	$(CXX) $(INCL) $(STD) klu_kernel.cpp -o klu_kernel.o $(CLIB) -O3
	@echo "Compiled klu_kernel.o successfully."

klu_kernel_arm: klu_kernel.cpp library
	$(CXX) $(INCL) $(STD_ARM) klu_kernel.cpp -o klu_kernel.o $(CLIB) -O3
	@echo "Compiled klu_kernel.o for ARM successfully."

klu_threaded: klu_threaded.cpp library
	$(CXX) $(INCL) $(STD) klu_threaded.cpp -o klu_threaded.o $(CLIB) -O3 -pthread
	@echo "Compiled klu_threaded.o successfully."

klu_threaded_arm: klu_threaded.cpp library
	$(CXX) $(INCL) $(STD_ARM) klu_threaded.cpp -o klu_threaded.o $(CLIB) -O3 -pthread
	@echo "Compiled klu_threaded.o for ARM successfully."

both: klu_kernel klu_threaded

both_arm: klu_kernel_arm klu_threaded_arm

clean:
	rm -rf ./SuiteSparse_config/*.o
	rm -rf ./AMD/Lib/*.o
	rm -rf ./BTF/Lib/*.o
	rm -rf ./KLU/Lib/*.o
	rm -rf ./lib/*.a
	rm -rf *.o
	@echo "Cleaned up all build artifacts."
