#include <cuda_runtime.h>
#include "src/Tools.cuh"
#include "src/Reduce.cuh"


int main() {
	constexpr int size = 1e8;
	constexpr size_t bytes = sizeof(float) * size;
	auto *h_data = static_cast<float *>(malloc(bytes));
	auto *h_output = static_cast<float *>(malloc(bytes));
	float *d_data, *d_output;

	cudaMalloc(&d_data, bytes);
	cudaMalloc(&d_output, bytes);

	for (int i = 0; i < size; i++) {
		h_data[i] = 1.0f;
	}

	// CPU GPU同步计算
	cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

	constexpr unsigned int blockSize = 256;
	dim3 grid((size + blockSize - 1) / blockSize, 1, 1);
	dim3 block(blockSize, 1, 1);
	int sem_bytes = blockSize * sizeof(float);
	reduce_v7<<<grid, block, sem_bytes>>>(d_data, d_output, size);
	cudaDeviceSynchronize();

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	// RTX 4060
	for (int i = 0; i < 100; i++) {
		// VecReduce<<<grid, block, sem_bytes>>>(d_data, d_output, size);
		// 79 Gb/s
		// reduce_v0<<<grid, block, sem_bytes>>>(d_data, d_output, size);
		// 126 Gb/s
		// reduce_v1<<<grid, block, sem_bytes>>>(d_data, d_output, size);
		// 131 Gb/s
		// reduce_v2<<<grid, block, sem_bytes>>>(d_data, d_output, size);
		// 127 Gb/s
		// reduce_v3<<<grid, block, sem_bytes>>>(d_data, d_output, size);
		// 175 Gb/s
		// reduce_v4<<<grid, block, sem_bytes>>>(d_data, d_output, size);
		// 174 Gb/s
		// reduce_v5<blockSize><<<grid, block, sem_bytes>>>(d_data, d_output, size);
		// 181 Gb/s
		// reduce_v6<<<grid, block, sem_bytes>>>(d_data, d_output, size);
		// 166 Gb/s
		reduce_v7<<<grid, block, sem_bytes>>>(d_data, d_output, size);
	}

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	// elapsed_total 记录的是 100 次运行的总时间 (毫秒)
	float elapsed_total = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_total, start, stop));

	// 1. 计算单次运行的平均时间 (毫秒)
	float elapsed_per_run = elapsed_total / 100.0f;

	// 2. 计算单次运行处理的总字节数 (Bytes)
	// 使用 double 强转，防止 size 过大时整数乘法溢出
	double bytes_per_run = static_cast<double>(size + grid.x) * sizeof(float);

	// 3. 计算有效带宽 (GB/s)
	// 公式: 字节数 / (时间(秒) * 10^9) -> 等价于 字节数 / (时间(毫秒) * 10^6)
	double bandwidth = bytes_per_run / (elapsed_per_run * 1e6);

	printf("Total elapsed time (100 runs) : %f ms\n", elapsed_total);
	printf("Average time per run          : %f ms\n", elapsed_per_run);
	printf("Effective Bandwidth           : %f GB/s\n", bandwidth);

	// 异步流
	// cudaStream_t stream;
	// cudaStreamCreate(&stream);
	// cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream);
	// cudaMemcpyAsync(d_output, h_output, bytes, cudaMemcpyHostToDevice, stream);
	// vectorReduce<<<grid, block, sem_bytes, stream>>>(d_data, d_output, size);
	// cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream);
	// cudaStreamSynchronize(stream);
	// cudaStreamDestroy(stream);


	float final_sum = 0.0f;
	for(int i = 0; i < grid.x; i++) {
		final_sum += h_output[i];
	}
	printf("Reduce sum: %.4f\n", final_sum);

	free(h_data); free(h_output);
	cudaFree(d_data); cudaFree(d_output);

	return 0;
}