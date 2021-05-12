all:
	nvcc -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75 direct_access_bench.cu -o bench
clean:
	rm -rf bench
