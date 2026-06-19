//
// Created by suqin on 2026/06/18.
//

#ifndef VECADD_CUH
#define VECADD_CUH
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include "../Tools.cuh"

__global__ static void vectorAdd(const float* a, const float* b, float* c, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}

__global__ static void vectorAddStride(const float* a, const float* b, float* c, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (idx < n) {
		c[idx] = a[idx] + b[idx];
		idx += stride;
	}
}

// CPU 基线
inline void vectorAddCPU(const float* A, const float* B, float* C, int N) {
	for (int i = 0; i < N; i++) {
		C[i] = A[i] + B[i];
	}
}

inline void testVecadd(int size) {
	size_t bytes = sizeof(float) * size;
	float *h_a = (float *)malloc(bytes);
	float *h_b = (float *)malloc(bytes);
	float *h_c = (float *)malloc(bytes);
	float* h_C_cpu = (float*)malloc(bytes);
	for (int i = 0; i < size; i++) {
		h_a[i] = 1.0f;
		h_b[i] = 2.0f;
	}

	// ========== CPU 计算 ==========
	auto cpu_start = std::chrono::high_resolution_clock::now();
	vectorAddCPU(h_a, h_b, h_C_cpu, size);
	auto cpu_end = std::chrono::high_resolution_clock::now();
	float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
	printf("CPU time: %.4f ms\n", cpu_ms);


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
	// ==========================================
	// 1. 预热 GPU (Warm-up)
	// 忽略这次执行的时间，让 GPU 状态初始化完毕并切换到最高性能模式
	// ==========================================
	vectorAdd<<<grid, block>>>(d_a, d_b, d_c, size);
	cudaDeviceSynchronize(); // 确保预热完成

	// ==========================================
	// 2. 循环多次测量求平均 (Multiple runs)
	// ==========================================
	int num_runs = 100; // 运行 100 次以摊薄开销
	cudaEventRecord(kernel_start);

	for (int i = 0; i < num_runs; i++) {
		vectorAdd<<<grid, block>>>(d_a, d_b, d_c, size);
	}

	cudaEventRecord(kernel_stop);
	cudaEventSynchronize(kernel_stop);


	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float gpu_total_ms = 0, kernel_ms = 0;
	cudaEventElapsedTime(&gpu_total_ms, start, stop);
	cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);



	float kernel_ms_avg = kernel_ms / num_runs;

	printf("GPU kernel time: %.4f ms\n", kernel_ms_avg);
	printf("GPU total time (with memcpy): %.4f ms\n", gpu_total_ms);
	printf("Speedup (kernel only): %.1fx\n", cpu_ms / kernel_ms_avg);
	printf("Speedup (end-to-end): %.1fx\n", cpu_ms / gpu_total_ms);

	for (int i = 0; i < size; i++) {
		if (h_c[i] != 3.0f) {
			printf("Error at index %d: %f\n", i, h_c[i]);
			break;
		}
	}
	printf("Vector addition completed successfully!\n");

	// ========== 计算有效带宽 ==========
	// 向量加法读 2 个 float、写 1 个 float = 12 bytes/element

	float bandwidth_gb = (3.0f * bytes) / (kernel_ms_avg * 1e-3f) / 1e9f;
	printf("Effective bandwidth: %.1f GB/s\n", bandwidth_gb);
	// 释放
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	free(h_a); free(h_b); free(h_c); free(h_C_cpu);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}


#endif //VECADD_CUH
