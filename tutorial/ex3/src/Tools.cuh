//
// Created by suqin on 2026/06/18.
//

#ifndef TOOLS_CUH
#define TOOLS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

inline void checkrun(cudaError_t exit_code, const char *file, int line) {
	if (exit_code != cudaSuccess) {
		printf("CUDA error at %s, %d, %s", file, line, cudaGetErrorString(exit_code));
		exit(exit_code);
	}

}

#define CUDA_CHECK(call) checkrun(call, __FILE__, __LINE__)

// #define CUDA_CHECK(call) do { \
// cudaError_t err = call; \
// if (err != cudaSuccess) { \
// fprintf(stderr, "CUDA error at %s:%d: %s\n", \
// __FILE__, __LINE__, cudaGetErrorString(err)); \
// exit(EXIT_FAILURE); \
// } \
// } while(0)

#endif //TOOLS_CUH
