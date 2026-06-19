//
// Created by suqin on 2026/06/18.
//

#ifndef VECSAXPY_CUH
#define VECSAXPY_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include "../Tools.cuh"

__global__ static void VecSAXPY(const float alpha, const float* A, const float* B, float* C, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		C[idx] = alpha * A[idx] + B[idx];
	}


}


inline void testVecSAXPY(int size) {
	size_t bytes = sizeof(float) * size;
	float *h_a = (float *)malloc(bytes);
	float *h_b = (float *)malloc(bytes);
	float *h_c = (float *)malloc(bytes);
	for (int i = 0; i < size; i++) {
		h_a[i] = 1.0f;
		h_b[i] = 2.0f;
	}

	cudaEvent_t start, stop, kernel_start, kernel_stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_stop);
	cudaEventRecord(start);

	float *d_a, *d_b, *d_c;
	CUDA_CHECK(cudaMalloc(&d_a, bytes));
	CUDA_CHECK(cudaMalloc(&d_b, bytes));
	CUDA_CHECK(cudaMalloc(&d_c, bytes));

	CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));


	int blockSize = 256;
	dim3 grid((size + blockSize - 1) / blockSize, 1, 1);
	dim3 block(blockSize, 1, 1);


	cudaEventRecord(kernel_start);
	VecSAXPY<<<grid, block>>>(2.0, d_a, d_b, d_c, size);
	cudaEventRecord(kernel_stop);


	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float gpu_total_ms = 0, kernel_ms = 0;
	cudaEventElapsedTime(&gpu_total_ms, start, stop);
	cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);

	printf("GPU kernel time: %.4f ms\n", kernel_ms);
	printf("GPU total time (with memcpy): %.4f ms\n", gpu_total_ms);


	// 释放
	free(h_a); free(h_b); free(h_c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}


#endif //VECSAXPY_CUH
