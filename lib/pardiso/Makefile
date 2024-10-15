# Set MKLROOT if not already set
MKLROOT ?= /opt/intel/oneapi/mkl/2024.2

all:
	g++ -O3 pardiso_kernel.cpp -o pardiso_kernel.o  -m64  -I"${MKLROOT}/include"  \
	-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
	${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
	${MKLROOT}/lib/intel64/libmkl_core.a \
	${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a \
	-Wl,--end-group -lgomp -lpthread -lm -ldl

clean:
	rm -rf *.o
