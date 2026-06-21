#include <cuda_runtime.h>

__device__ float wrap_summary(float value) {
	for (int offset = 16; offset > 0; offset /= 2) {
		value += __shfl_down_sync(0xffffffff, value, offset);
	}
	return value;
}

__global__ void VecReduce(const float *input, float *output, int N) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float var = tid < N ? input[tid] : 0.f;
	var = wrap_summary(var);

	__shared__ float wrap_sum[32];
	unsigned int lane_id = threadIdx.x % 32;
	unsigned int wrap_id = threadIdx.x / 32;
	if (lane_id == 0) {
		wrap_sum[wrap_id] = var;
	}

	__syncthreads();

	unsigned int numWarps = blockDim.x / 32;
	var = (threadIdx.x < numWarps) ? wrap_sum[threadIdx.x] : 0.0f;

	if (wrap_id == 0) {
		var = wrap_summary(var);
	}

	if (threadIdx.x == 0) {
		output[blockIdx.x] = var;
	}
}

// 每个线程处理 TILE_DIM / BLOCK_ROWS 个元素
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared(float* out, float* in, int width, int height) {
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	for (int offset = 0; offset < TILE_DIM; offset += TILE_DIM/BLOCK_ROWS) {
		if (x < width && (y + offset) < height) {
			tile[threadIdx.y+offset][threadIdx.x] = in[(y+offset)*width + x];
		}
	}
	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int offset = 0; offset < TILE_DIM; offset += TILE_DIM/BLOCK_ROWS) {
		if (x < height && (y + offset) < width) {
			out[(y+offset)*height + x] = tile[threadIdx.x][threadIdx.y+offset];
		}
	}
}



int main() {
	const int N = 5e8;  // 1M 元素


	return 0;
}