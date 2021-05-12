all:
	nvcc -gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 direct_access_bench.cu -o bench
clean:
	rm -rf bench
