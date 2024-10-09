CC=g++
CXX=g++

CLIB=-L lib -lklu -lamd -lbtf -lsuitesparseconfig -lm
INCL=-I include

# Add -fsanitize=address to enable AddressSanitizer and -g for debugging info
# -O0 disables optimizations for easier debugging
STD=-std=c++17 -march=native -mtune=native -fsanitize=address -g -O0
STD_ARM=-std=c++17 -mcpu=native -fsanitize=address -g -O0

PRE?=TRUE

CLEAN=rm -rf *.o *.a

library:
	(cd SuiteSparse_config ; $(MAKE))
	(cd AMD/Lib ; $(MAKE))
	(cd BTF/Lib ; $(MAKE))
	(cd KLU/Lib ; $(MAKE))

# Rule for building klu_kernel with debugging symbols and AddressSanitizer
klu_kernel: klu_kernel.cpp library
	$(CXX) $(INCL) $(STD) klu_kernel.cpp -o klu_kernel.o $(CLIB)

# Rule for ARM build with debugging symbols and AddressSanitizer
klu_kernel_arm: klu_kernel.cpp library
	$(CXX) $(INCL) $(STD_ARM) klu_kernel.cpp -o klu_kernel.o $(CLIB)

# Rule for building klu_threaded with debugging symbols and AddressSanitizer
klu_threaded: klu_threaded.cpp library
	$(CXX) $(INCL) $(STD) klu_threaded.cpp -o klu_threaded.o $(CLIB) -pthread

klu_threaded_arm: klu_threaded.cpp library
	$(CXX) $(INCL) $(STD_ARM) klu_threaded.cpp -o klu_threaded.o $(CLIB) -pthread

both: klu_kernel klu_kernel_2 klu_threaded library

both_arm: klu_kernel_arm klu_threaded_arm library

clean:
	rm -rf ./SuiteSparse_config/*.o
	rm -rf ./AMD/Lib/*.o
	rm -rf ./BTF/Lib/*.o
	rm -rf ./KLU/Lib/*.o
	rm -rf ./lib/*.a
	rm -rf *.o
