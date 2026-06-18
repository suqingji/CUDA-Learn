//
// Created by suqin on 2026/06/18.
//

#ifndef VECDOT_CUH
#define VECDOT_CUH
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include "src/Tools.cuh"

__global__ static void VecDot(const float* A, const float* B, float* C, int N) {
	extern __shared__ float semm[];
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	semm[tid] = gid < N ? A[gid] * B[gid] : 0.0f;
	__syncthreads();
	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		if (tid < i) {
			semm[tid] = semm[tid] + semm[tid + i];
		}
		__syncthreads();
	}
	if (tid == 0) {
		C[blockIdx.x] = semm[tid];
	}

}

inline void VectorDot_V2(int size) {
	size_t bytes = sizeof(float) * size;
	float *h_a = (float *)malloc(bytes);
	float *h_b = (float *)malloc(bytes);
	float *h_output = (float *)malloc(bytes);
	for (int i = 0; i < size; i++) {
		h_a[i] = 1.0f;
		h_b[i] = 2.0f;
	}

	float * d_a, *d_b, * d_output;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_output, bytes);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	int blockSize = 256;
	dim3 grid((size + blockSize - 1) / blockSize, 1, 1);
	dim3 block(blockSize, 1, 1);

	int sem_bytes = blockSize * sizeof(float);
	VecDot<<<grid, block, sem_bytes>>>(d_a, d_b, d_output, size);
	cudaGetLastError();

	cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

	float final_sum = 0.0f;
	for(int i = 0; i < grid.x; i++) {
		final_sum += h_output[i];
	}
	printf("Reduce sum: %.4f\n", final_sum);

	free(h_a); free(h_b); free(h_output);
	cudaFree(d_a); cudaFree(d_b);
}

#endif //VECDOT_CUH
