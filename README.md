# Distributed-SpMV
分布式稀疏矩阵-向量乘（c/MPI+OpenMP）

DistSpMV.c - MPI+omp分布式SpMV（未使用超图分割）

DistSpMV_Reordered.c - MPI+omp分布式SpMV（使用超图分割）

DistSpMV_Balanced.c - MPI+omp分布式SpMV (Communication Balance)

编译：

mpicc DistSpMV_Balanced.c -fopenmp -lmetis -lm -O3

运行:

mpirun -n 进程数 ./a.out 矩阵名 线程数
