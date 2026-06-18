#include "src/VecOprate/VecAdd.cuh"
#include "src/VecOprate/VecReduce.cuh"
#include "src/VecOprate/VecDot.cuh"
#include "src/VecOprate/VecHadamardProduct.cuh"
#include "src/VecOprate/VecSAXPY.cuh"


int main() {
	const int N = 1e9;  // 1M 元素

	testVecadd(N);
	testReduce(N);
	// auto cpu_start = std::chrono::high_resolution_clock::now();
	// VectorDot(N);
	// cudaDeviceSynchronize();
	// auto cpu_end = std::chrono::high_resolution_clock::now();
	// float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
	// printf("CPU time: %.4f ms\n", cpu_ms);
	//
	// cpu_start = std::chrono::high_resolution_clock::now();
	// VectorDot_V2(N);
	// cudaDeviceSynchronize();
	// cpu_end = std::chrono::high_resolution_clock::now();
	// cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
	// printf("CPU time: %.4f ms\n", cpu_ms);
	// testVecSAXPY(N);
	return 0;
}