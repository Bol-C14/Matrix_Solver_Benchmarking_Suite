all:
	g++ -O3 nicslu_kernel.cpp -o nicslu_kernel.o -I include/ -L linux/lib_centos6_x64_gcc482_fma/int32/ -lnicslu -Wl,-rpath=linux/lib_centos6_x64_gcc482_fma/int32/

test:
	./demo

clean:
	rm -rf *.o