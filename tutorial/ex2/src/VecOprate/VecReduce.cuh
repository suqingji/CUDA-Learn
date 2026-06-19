//
// Created by suqin on 2026/06/18.
//

#ifndef VECREDUCE_CUH
#define VECREDUCE_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include "../Tools.cuh"

__global__ static void vectorReduce(const float* input, float* output, int n) {
	extern __shared__ float smem[];
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	smem[tid] = gid < n ? input[gid] : 0.0f;
	__syncthreads();

	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		if (tid < i) {
			smem[tid] = smem[tid] + smem[tid + i];
		}
		__syncthreads();

	}
	if (tid == 0) {
		output[blockIdx.x] = smem[tid];
	}
}

inline void testReduce(int size) {
	size_t bytes = sizeof(float) * size;
	float *h_data = (float *)malloc(bytes);
	float *h_output = (float *)malloc(bytes);
	float *d_data, *d_output;

	cudaMalloc(&d_data, bytes);
	cudaMalloc(&d_output, bytes);

	for (int i = 0; i < size; i++) {
		h_data[i] = 1.0f;
	}

	// CPU GPU同步计算
	cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

	int blockSize = 256;
	dim3 grid((size + blockSize - 1) / blockSize, 1, 1);
	dim3 block(blockSize, 1, 1);
	int sem_bytes = blockSize * sizeof(float);
	// exit(0);
	vectorReduce<<<grid, block, sem_bytes>>>(d_data, d_output, size);

	CUDA_CHECK(cudaGetLastError());
	cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

	// 异步流
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_output, h_output, bytes, cudaMemcpyHostToDevice, stream);

	vectorReduce<<<grid, block, sem_bytes, stream>>>(d_data, d_output, size);

	cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);


	float final_sum = 0.0f;
	for(int i = 0; i < grid.x; i++) {
		final_sum += h_output[i];
	}
	printf("Reduce sum: %.4f\n", final_sum);

	free(h_data); free(h_output);
	cudaFree(d_data); cudaFree(d_output);

}


#endif //VECREDUCE_CUH
