#include <cuda.h>
#include <cstdint>
#include <iostream>
#include <chrono>
#include "kernels.hu"

#define BLK_SIZE (128)
#define GRID_SIZE (1024ULL)
#define N_BLKS  (GRID_SIZE/BLK_SIZE)

typedef ulong4 d_t;

enum bench_type { READ = 0, WRITE = 1, MIXED = 2};

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Please specify GPU id, number of elems, and bench type\n";
        exit(1);

    }


    const unsigned int gpu_id = std::stoul(std::string(argv[1]));
    const unsigned int num_elems = std::stoul(std::string(argv[2]));
    const unsigned int type = std::stoul(std::string(argv[3]));

    cuda_err_chk(cudaSetDevice(gpu_id));

    cudaSetDeviceFlags(cudaDeviceMapHost);

    d_t * h_arr = nullptr;
    d_t * d_flag = nullptr;


    cuda_err_chk(cudaHostAlloc((void **)&h_arr,  num_elems * sizeof(d_t),  cudaHostAllocMapped));
    cuda_err_chk(cudaMalloc((void **) &d_flag, sizeof(d_t)));

    d_t * d_arr = nullptr;

    cuda_err_chk(cudaHostGetDevicePointer((void **)&d_arr,  (void *) h_arr , 0));

    auto start = std::chrono::high_resolution_clock::now();
    switch (type) {
    case READ:
        gpu_read<BLK_SIZE, d_t><<<GRID_SIZE,BLK_SIZE>>>(d_arr, d_flag, num_elems * sizeof(d_t));
        break;
    case WRITE:
        gpu_write<BLK_SIZE, d_t><<<GRID_SIZE,BLK_SIZE>>>(d_arr, num_elems * sizeof(d_t));
        break;
    case MIXED:
        gpu_mix<BLK_SIZE, d_t><<<GRID_SIZE,BLK_SIZE>>>(d_arr, d_flag, num_elems * sizeof(d_t));
        break;
    default:
        std::cerr << "Please specify valid bench type\n";
        exit(1);
        break;
    }

    cuda_err_chk(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Data: " << (num_elems * sizeof(d_t))/1024/1024/1024 << " gigabytes\n" <<
        "Time: " << ((double)duration.count())/1000000 << " seconds\n" <<
        "Bandwidth: " << ((double)(num_elems * sizeof(d_t)))/1024/1024/1024/((double)duration.count()/1000000) << " GB/s\n";

}
