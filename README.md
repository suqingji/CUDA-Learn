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

















