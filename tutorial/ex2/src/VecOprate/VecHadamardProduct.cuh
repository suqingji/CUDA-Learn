//
// Created by suqin on 2026/06/18.
//

#ifndef VECHADAMARDPRODUCT_CUH
#define VECHADAMARDPRODUCT_CUH
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include "src/Tools.cuh"


__global__ static void VecMulti(const float* A, const float* B, float* C, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		C[idx] = A[idx] * B[idx];
	}

}


inline void VectorDot(int size) {
	size_t bytes = sizeof(float) * size;
	float *h_a = (float *)malloc(bytes);
	float *h_b = (float *)malloc(bytes);
	float *h_c = (float *)malloc(bytes);
	float *h_output = (float *)malloc(bytes);
	for (int i = 0; i < size; i++) {
		h_a[i] = 1.0f;
		h_b[i] = 2.0f;
	}

	float * d_a, *d_b, *d_c, * d_output;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);
	cudaMalloc(&d_output, bytes);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	int blockSize = 256;
	dim3 grid((size + blockSize - 1) / blockSize, 1, 1);
	dim3 block(blockSize, 1, 1);

	int sem_bytes = blockSize * sizeof(float);
	VecMulti<<<grid, block>>>(d_a, d_b, d_c, size);
	cudaGetLastError();
	vectorReduce<<<grid, block, sem_bytes>>>(d_c, d_output, size);
	cudaGetLastError();

	cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

	float final_sum = 0.0f;
	for(int i = 0; i < grid.x; i++) {
		final_sum += h_output[i];
	}
	printf("Reduce sum: %.4f\n", final_sum);

	free(h_a); free(h_b); free(h_c); free(h_output);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
#endif //VECHADAMARDPRODUCT_CUH
