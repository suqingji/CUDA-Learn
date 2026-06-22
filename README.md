这是我学习CUDA的一个记录项目



# 入门

## tutorial

该部分参考https://suqingji.github.io/AI-Infra-Notebook/

### lesson 1: ex1，感受CUDA

https://suqingji.github.io/AI-Infra-Notebook/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-cuda%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/cuda%E7%BC%96%E7%A8%8B%E5%85%A5%E9%97%A8%E6%8C%87%E5%8D%97/

向量相加

### lesson 2: ex2，CUDA 基础入门

https://suqingji.github.io/AI-Infra-Notebook/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-cuda%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/%E7%AC%AC1%E7%AB%A0-cuda%E7%BC%96%E7%A8%8B%E5%85%A5%E9%97%A8/

核函数

共享内存

规约

Grid-stride Loop

异步传输与计算重叠

性能分析：带宽、算数强度，带宽利用率

### lesson: ex3，CUDA进阶

https://suqingji.github.io/AI-Infra-Notebook/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-cuda%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/%E7%AC%AC2%E7%AB%A0-cuda%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%9F%BA%E7%A1%80/

写出能跑的 CUDA 代码只是起点，写出跑得快的代码才是 AI Infra 工程师的核心能力。本章建立 CUDA 性能优化的核心方法论。

1. **Warp 与执行模型**：详解 SIMT 执行模式、Warp Divergence 导致的性能损失，以及 Warp Shuffle 指令实现线程间高效数据交换。



2. **内存访问优化**是性能提升最大的杠杆：Coalesced Access（合并访问）决定全局内存效率，Bank Conflict 影响共享内存性能（Padding 技巧解决），向量化加载（float4/int4）进一步提升带宽利用率。



3. **Occupancy 与资源分配**解释 Occupancy 的定义与意义，分析影响因素（寄存器数、共享内存、Block 大小），强调 Occupancy 不是越高越好——需要在 Latency Hiding 和 Resource Utilization 之间找平衡。



3. **同步与原子操作**涵盖 `__syncthreads()` 块内同步和原子操作的使用与性能影响。



### lesson：ex4，性能优化实操，Reduce算子

https://suqingji.github.io/AI-Infra-Notebook/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-cuda%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/%E7%AC%AC3%E7%AB%A0-%E7%BB%8F%E5%85%B8%E7%AE%97%E5%AD%90%E5%AE%9E%E7%8E%B0-reduce/

从 V0 到 V7，每一步优化都针对一个具体的性能瓶颈：

1. **Warp Divergence（初步）**：用 strided index 替换 `tid % (2*s) == 0` 判断，让完整 Warp 进入/跳过分支（V1）
2. **Bank Conflict + Warp Divergence（彻底）**：反转步长方向，从 `blockDim/2` 逐步减半，一次性解决两类问题（V2）
3. **空闲线程**：每线程处理 2 个元素，减少 Block 数量，提升线程利用率（V3）
4. **多余同步**：Warp 内天然同步，展开最后 5 轮可以省去 `__syncthreads()`（V4）
5. **循环开销**：模板参数让编译器删除无用分支，生成紧凑的直线代码（V5）
6. **访存层次**：Warp Shuffle 直接在寄存器间通信，比 Shared Memory 更快（V6）
7. **带宽效率**：`float4` 减少指令数，Grid Stride Loop 最大化 GPU 占用率（V7）

理解这些优化思路，不仅对 Reduce 有用——**在几乎所有 Memory-Bound Kernel 的设计中，同样的思路都会反复出现**。



在RTX4060这个轻量级显卡上测试，V7版本甚至更慢，不如原教程文档中A100效果，可能是窄位宽、大缓存、少 SM 的显卡上，**简单粗暴的代码往往比花哨的优化跑得更快**



### lesson：ex5，性能优化实操，GEMM算子

https://suqingji.github.io/AI-Infra-Notebook/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-cuda%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/%E7%AC%AC4%E7%AB%A0-%E7%BB%8F%E5%85%B8%E7%AE%97%E5%AD%90%E5%AE%9E%E7%8E%B0-gemm/





























