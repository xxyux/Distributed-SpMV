# Introduction
`Distributed-SpMV`-这篇工作被IEEE/ACM CCGrid'23接收，[paper link](https://ieeexplore.ieee.org/document/10171520).

分布式稀疏矩阵-向量乘（c/MPI+OpenMP）hightlight:
1. 多节点集群
2. 图重排
3. MPI通信
4. 计算通信重叠

# Algorithm process
![](https://github.com/xxyux/Distributed-SpMV/blob/master/algorithm.png)

# Run
DistSpMV.c - MPI+omp分布式SpMV（未使用超图分割）

DistSpMV_Reordered.c - MPI+omp分布式SpMV（使用超图分割）

DistSpMV_Balanced.c - MPI+omp分布式SpMV (Communication Balance)

编译：

mpicc DistSpMV_Balanced.c -fopenmp -lmetis -lm -O3

运行:

mpirun -n 进程数 ./a.out 矩阵名 线程数
