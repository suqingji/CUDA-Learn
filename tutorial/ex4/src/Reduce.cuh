//
// Created by suqin on 2026/06/21.
//
#pragma once
// #ifndef REDUCE_CUH
// #define REDUCE_CUH

#include "Tools.cuh"

// V0: 朴素的树形规约（步长从小到大）
__global__ void static reduce_v0(float* input, float* output, int n) {
	extern __shared__ float smem[];

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// 将全局内存数据加载到共享内存
	smem[tid] = (gid < n) ? input[gid] : 0.0f;
	__syncthreads();

	// 树形规约：步长从 1 开始逐步翻倍
	for (int step = 1; step < blockDim.x; step *= 2) {
		if (tid % (2 * step) == 0) {
			smem[tid] += smem[tid + step];
		}
		__syncthreads();
	}

	// 每个 Block 的结果写回全局内存
	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// V1: strided index 方式，减少 Warp Divergence
__global__ void static reduce_v1(float* input, float* output, int n) {
	extern __shared__ float smem[];

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	smem[tid] = (gid < n) ? input[gid] : 0.0f;
	__syncthreads();

	// 步长从 1 开始逐步翻倍，但用 strided index 映射活跃线程
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = threadIdx.x * 2 * s;
		if (index < blockDim.x) {
			smem[index] += smem[index + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// V2: 步长从大到小，同时消除 Warp Divergence 与 Bank Conflict
__global__ void static reduce_v2(float* input, float* output, int n) {
	extern __shared__ float smem[];

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	smem[tid] = (gid < n) ? input[gid] : 0.0f;
	__syncthreads();

	// 步长从 blockDim.x/2 开始，每轮减半
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			smem[tid] += smem[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// V3: 每线程处理 2 个元素，减少空闲线程
__global__ void static reduce_v3(float* input, float* output, int n) {
	extern __shared__ float smem[];

	int tid = threadIdx.x;
	int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// 每个线程加载 2 个相距 blockDim.x 的元素并求和
	float val = 0.0f;
	if (gid < n)              val += input[gid];
	if (gid + blockDim.x < n) val += input[gid + blockDim.x];
	smem[tid] = val;
	__syncthreads();

	// 步长从大到小的规约（同 V2）
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			smem[tid] += smem[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// 辅助函数：展开最后 32 个线程的规约
__device__ void static warpReduce(volatile float* smem, int tid) {
	smem[tid] += smem[tid + 32];
	smem[tid] += smem[tid + 16];
	smem[tid] += smem[tid +  8];
	smem[tid] += smem[tid +  4];
	smem[tid] += smem[tid +  2];
	smem[tid] += smem[tid +  1];
}

// V4: 展开最后一个 Warp（在 V3 基础上）
__global__ void static reduce_v4(float* input, float* output, int n) {
	extern __shared__ float smem[];

	int tid = threadIdx.x;
	int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// 每线程处理 2 个元素（继承自 V3）
	float val = 0.0f;
	if (gid < n)              val += input[gid];
	if (gid + blockDim.x < n) val += input[gid + blockDim.x];
	smem[tid] = val;
	__syncthreads();

	// 规约循环仅执行到 step > 32
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			smem[tid] += smem[tid + s];
		}
		__syncthreads();
	}

	// 最后一个 Warp 内的规约，无需 __syncthreads()
	if (tid < 32) {
		warpReduce(smem, tid);
	}

	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// V5: 完全循环展开（模板参数）
template <int BLOCK_SIZE>
__global__ void static reduce_v5(float* input, float* output, int n) {
	extern __shared__ float smem[];

	int tid = threadIdx.x;
	int gid = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

	// 每线程处理 2 个元素
	float val = 0.0f;
	if (gid < n)              val += input[gid];
	if (gid + BLOCK_SIZE < n) val += input[gid + BLOCK_SIZE];
	smem[tid] = val;
	__syncthreads();

	// 编译期展开：BLOCK_SIZE 已知，不满足的分支会被编译器直接删除
	if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
	if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
	if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

	// 最后 Warp 内展开
	if (tid < 32) {
		volatile float* vsmem = smem;
		if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid +  8];
		vsmem[tid] += vsmem[tid +  4];
		vsmem[tid] += vsmem[tid +  2];
		vsmem[tid] += vsmem[tid +  1];
	}

	if (tid == 0) output[blockIdx.x] = smem[0];
}

// Warp 内规约辅助函数
__device__ float static warpReduceSum(float val) {
	// 每次将右半边的值加到左半边
	for (int offset = 16; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, offset);
	}
	return val;  // lane 0 持有最终结果
}

// V6: Warp Shuffle + 两级规约
__global__ void static reduce_v6(float* input, float* output, int n) {
	int tid  = threadIdx.x;
	int gid  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	int lane = tid % 32;      // 线程在 Warp 内的编号（0~31）
	int wid  = tid / 32;      // 该线程属于哪个 Warp

	// 每线程处理 2 个元素
	float val = 0.0f;
	if (gid < n)              val += input[gid];
	if (gid + blockDim.x < n) val += input[gid + blockDim.x];

	// 第一级：Warp 内规约
	val = warpReduceSum(val);

	// 将每个 Warp 的结果（仅 lane 0 有效）存入 Shared Memory
	__shared__ float warp_results[32];  // 最多 32 个 Warp（1024/32）
	if (lane == 0) {
		warp_results[wid] = val;
	}
	__syncthreads();

	// 第二级：Warp 间规约（用 Warp 0 处理）
	int num_warps = blockDim.x / 32;
	if (wid == 0) {
		val = (lane < num_warps) ? warp_results[lane] : 0.0f;
		val = warpReduceSum(val);
	}

	if (tid == 0) output[blockIdx.x] = val;
}

// V7: float4 向量化加载 + Grid Stride Loop + Warp Shuffle
__global__ void static reduce_v7(float* input, float* output, int n) {
	int tid  = threadIdx.x;
	int lane = tid % 32;
	int wid  = tid / 32;

	// float4 加载：每线程每次处理 4 个 float
	float4* input4 = reinterpret_cast<float4*>(input);
	int n4 = n / 4;  // float4 的元素数量

	float val = 0.0f;

	// Grid Stride Loop：每个线程以 gridDim.x * blockDim.x 为步长迭代
	for (int idx = blockIdx.x * blockDim.x + tid;
		 idx < n4;
		 idx += gridDim.x * blockDim.x)
	{
		float4 data = input4[idx];
		val += data.x + data.y + data.z + data.w;
	}

	// 处理 n 不是 4 的倍数时的尾部元素
	int tail_start = n4 * 4;
	for (int idx = tail_start + blockIdx.x * blockDim.x + tid;
		 idx < n;
		 idx += gridDim.x * blockDim.x)
	{
		val += input[idx];
	}

	// Warp 内规约
	for (int offset = 16; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, offset);
	}

	__shared__ float warp_results[32];
	if (lane == 0) warp_results[wid] = val;
	__syncthreads();

	int num_warps = blockDim.x / 32;
	if (wid == 0) {
		val = (lane < num_warps) ? warp_results[lane] : 0.0f;
		for (int offset = 16; offset > 0; offset >>= 1) {
			val += __shfl_down_sync(0xffffffff, val, offset);
		}
	}

	if (tid == 0) output[blockIdx.x] = val;
}
























__global__ void static VecReduce(const float * input, float * output, int N) {
	extern __shared__ float data[];

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = gid < N ? input[gid] : 0.f;

	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
		if (tid < offset) {
			data[tid] += data[tid + offset];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[blockIdx.x] = data[0];
	}


}


// #endif //REDUCE_CUH
