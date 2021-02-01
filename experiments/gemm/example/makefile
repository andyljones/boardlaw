all: gemm

gemm: gemm.cpp
	nvcc -o $@ $^ -Xptxas -v -O3 -lcublas

clean:
	rm -f gemm *~
