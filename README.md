# The Chase for cuBLAS-like SGEMM Performance (WIP)

Optimization of General Matrix Multiplication (GEMM) operations is almost the only way to even truly coming close to learn CUDA, with it taking siginficant FLOPs during the training and inference. So how much work is it to write a quick CUDA SGEMM?

SGEMM (Single-precision General Matrix Multiply) is an implementation of GEMM operations. The operation can be represented as:

$$C_{i,j} = \sum_{k=1}^{N} A_{i,k} \cdot B_{k,j}, \quad \forall i, j \in 1, N$$

Where 𝐶𝑖,𝑗​ is the element at the 𝑖-th row and 𝑗-th column of the resulting matrix 𝐶, and 𝐴 and 𝐵 are the input matrices. The summation runs over the common dimension 𝑘, which ranges from 1 to 𝑁, the size of the square matrices.

The matrices are composed of single-precision floating-point numbers (32-bit). With CUDA, it can leverage the massively parallel architecture of GPUs to perform this operation efficiently by dividing the workload among thousands of lightweight threads running in parallel.

Of course this is the natural progression of my borderline autistic obsession over embarassingly parallel architectures since getting my hands on the PS3 devkit. Like the previous article, the implementation here isn't necessarily an attempt to replace cuBLAS but more as an attempt to learn about low-level programming in parallelized architectures and its interesting performance characteristics.

This article would not be possible without :

* NVIDIA's own documentation on the architecture of its GPUs and CUDA Programming (archive link), specifically the following docs
    
    * WIP
        
* The initial idea that started this from an article by Simon Boehm about [the same topic](https://siboehm.com/articles/22/CUDA-MMM) (written better admittedly)
    
* [Explanation about cuBLAS](https://blog.csdn.net/u011197534/article/details/78378536) by Summit\_Yue in CSDN
    
* More advanced ideas posited by nicholaswilde regarding [CUDA SGEMM optimization](https://www.zhihu.com/question/41060378/answer/2645323107) in Zhihu
    
* An article about [resolving bank conflicts in shared memory](https://zh0ngtian.tech/posts/96744e8c.html) by Zhongtian
    
* WIP
    

# Background

Before diving into CUDA kernel code, we will need some background knowledge about GPUs and CUDA. Those with a foundational knowledge in CUDA and GPU intrinsics can skip this section, but for most mortals we can dive into the section.

### GPU Architecture

A GPU is a SIMT (Single Instruction Multiple Threads) processor that processes thousands of threads in parallel with its many-core structure. Now while i'm using an RTX 4070 for this article which uses the consumer-focused Ada Lovelace archtecture, we will be using NVIDIA's recent Hopper architecture as an example, specifically the H100 GPU, we can understand the advancements and structure of modern GPUs.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1715921468544/1af553a3-4358-4e65-8bfc-82b5c8b7a3b8.png align="center")

The H100 GPU is divided into multiple GPU Processing Clusters (GPCs). Each GPC acts as a comprehensive unit, incorporating several Texture Processing Clusters (TPCs). Within the H100, the entire GPU is organized into 8 GPCs. Each GPC contains several TPCs, which are smaller clusters designed to process texture mapping and other graphics-related tasks. In the H100, each GPC includes up to 6 TPCs, and each TPC is composed of 2 Streaming Multiprocessors (SMs).

### Streaming Microprocessor (SM)

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1715921577387/8cddbc7a-0934-4e32-9e51-88b41dfef07a.png align="center")

The SM is the core computational unit within the GPU, and the H100's SMs are designed to maximize performance and efficiency. Each SM houses several types of cores and specialized units that facilitate various computations. Specifically, the H100 features up to 168 SMs, each containing 128 CUDA cores. These CUDA cores handle integer and floating-point operations, essential for general-purpose GPU computing.

Complementing the CUDA cores are the Tensor Cores, specialized units designed for matrix multiplication operations. The H100's fourth-generation Tensor Cores deliver a significant performance boost, particularly for AI and deep learning workloads. These Tensor Cores support new data formats such as FP8, which provides higher throughput and a reduced memory footprint compared to previous formats like FP16.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1715922013515/791c14c9-2de9-4739-bf2f-c1c3583011de.jpeg align="center")

The SMs also feature advanced warp schedulers and dispatch units. Each SM includes four warp schedulers capable of dispatching two instructions per warp per cycle. Warps, which are groups of 32 threads, execute instructions in lockstep, and the warp schedulers manage their execution to ensure efficient utilization of the CUDA cores and other execution units within the SM. This advanced scheduling capability helps maximize throughput and efficiency across the SM's computational resources.

The memory hierarchy within each SM includes a combined L1 data cache and shared memory. This combined memory block significantly enhances performance by providing faster access to frequently used data. The H100's SMs offer a combined capacity of 256 KB per SM, configurable up to 228 KB for shared memory, allowing for flexible usage based on the application's needs. This flexibility helps optimize performance for different types of workloads.

Another critical component of each SM is the register file, which provides fast, on-chip storage for thread-specific data. The H100's SMs feature a large register file with 256 KB of registers per SM. This extensive register space supports the efficient execution of complex kernels that have high register requirements, ensuring that data is readily available for computation without unnecessary delays.

The H100 SMs also incorporate support for asynchronous execution, which includes the Tensor Memory Accelerator (TMA) for efficient data transfers and the Asynchronous Transaction Barrier for synchronizing data movement and computation. These features reduce idle times and maximize the utilization of the GPU's resources, allowing for more efficient execution of complex tasks.

### **Distributed Shared Memory (DSMEM)**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1715922065919/5a7281c1-e7b7-47f6-88d1-6fd0e047af4b.jpeg align="center")

Another significant enhancement is with Distributed Shared Memory (DSMEM), which allows all threads to directly access the shared memory of other SMs within the same cluster. This is achieved through load, store, and atomic operations, making the shared memory's virtual address space logically distributed across all the thread blocks in a cluster.

DSMEM enables more efficient data exchange between SMs, eliminating the need to use global memory for passing data. This is facilitated by a dedicated SM-to-SM network within clusters, ensuring fast, low-latency access to remote DSMEM. Compared to using global memory, DSMEM accelerates data exchange between thread blocks by approximately 7x.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1716456969882/4d413a65-b944-4a82-81e2-7a4e6088208d.jpeg align="center")

One of the critical advancements of the H100 GPU over the A100 GPU is this direct SM-to-SM network in clusters. In the H100, all DSMEM segments from thread blocks within a cluster are mapped into the generic address space of each thread. This allows direct referencing of DSMEM with simple pointers. CUDA developers can utilize the cooperative\_groups API to create generic pointers to any thread block within the cluster. Additionally, DSMEM transfers can be performed as asynchronous copy operations, synchronized using shared memory-based barriers to track completion.

### **Cache Hierarchy**

The cache hierarchy in the H100 GPU is crucial for managing data access efficiently across various levels of memory. The L1 cache and shared memory are combined within each SM, offering a configurable capacity of up to 256 KB per SM. This configuration allows developers to balance the needs of their applications between faster access and larger storage within the SM.

The L2 cache is shared among all SMs on the GPU. It provides a larger storage capacity than the L1 cache but operates at a higher latency. The L2 cache helps reduce the frequency of accesses to global memory by storing data that is shared across multiple SMs, thereby improving overall memory access efficiency.

Global memory, or DRAM, provides the largest storage capacity but has the highest latency. It serves as the main memory for the GPU, where data is initially stored and from which it is transferred to lower levels of cache as needed.

## CUDA Programming

The CUDA programming model enables developers to harness the immense parallel processing capabilities of GPUs. It is hierarchically structured into grids, blocks, and threads, which correspond directly to the hardware architecture of the GPU.

When a CUDA kernel is launched, it is executed by a grid of thread blocks. Each grid can be one-, two-, or three-dimensional, facilitating the mapping of complex data structures like matrices and volumes directly onto the GPU's processing model. The dimensions of the grid and thread blocks are specified using the `dim3` type, which allows easy indexing and management of data. For instance, the statement `kernel<<<numBlocks, threadsPerBlock>>>(args);` launches a kernel with the specified configuration.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1715923342439/f8cea8d0-4fc6-4556-b8ff-75df123d545f.png align="center")

Each block within a grid consists of multiple threads, and these blocks are the fundamental units of execution on the GPU. A block can contain up to 1024 threads, each of which is uniquely identified using the built-in `threadIdx` variable. Similarly, blocks are indexed within a grid using `blockIdx`. This hierarchical structure allows for organized and efficient parallel computation.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1715923322038/dee53b9a-765e-4339-b8a7-c4b20612bacc.png align="center")

The execution model of CUDA is fundamentally parallel. When a kernel is executed, the threads within a block are grouped into warps, which are groups of 32 threads. Warps are the smallest units of execution and are scheduled by the SM's warp scheduler. Each clock cycle, the scheduler selects a warp that is ready to execute and issues the same instruction to all threads within that warp. This SIMT (Single Instruction, Multiple Threads) model ensures that multiple threads execute concurrently, maximizing the utilization of the GPU's computational resources.

At the lowest latency level, each thread has access to its own private registers. Registers provide the fastest memory access but are limited in size. Next, each SM has shared memory, which is on-chip and much faster than global memory. Shared memory is used for data that needs to be quickly accessed by all threads within a block, reducing the need for multiple threads to access the slower global memory.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1715923273259/ff8ba054-e99e-47b0-8486-58a82558e1cb.png align="center")

The L1 cache is another on-chip memory shared by all threads within an SM, providing fast access to frequently used data. The L2 cache, while larger and shared among all SMs on the GPU, operates at a higher latency compared to the L1 cache. Despite this, the L2 cache is still significantly faster than accessing global memory, which resides in the GPU's DRAM. Global memory provides the largest storage capacity but has the highest latency.

# Design Iterations

## V1 : Naive Kernel Implementation

The definition of GEMM is 𝐶←𝛼𝐴𝐵+𝛽𝐶. For the sake of discussion, we are using 𝛼=1,𝛽=0, which means ordinary matrix multiplication 𝐶←𝐴𝐵. The elements of the matrices in the code are all float types, so it is SGEMM.

Suppose the dimensions of matrices A, B, and C are \[M, K\], \[K, N\], and \[M, N\] respectively. The simplest SGEMM code on the CPU is as follows:

```c
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
        }
        C[OFFSET(m, n, N)] = value;
    }
}
```

It's straightforward, just three nested loops. The calculation of each element in C corresponds to the third loop, with K multiplications and K additions in total. Since there are M \* N elements in C, SGEMM requires 2∗𝐾∗𝑀∗𝑁 floating-point operations in total.

The initial starting point for optimization using CUDA programming is to rewrite the serial code on the CPU into parallel code on the GPU.

This accelerates the computation through parallelism, marking the beginning of CUDA programming and the Qi Refining stage. Although sgemm\_gpu\_v1 is already several orders of magnitude faster than sgemm\_cpu, it's unlikely that one would stop here.

Each thread is responsible for computing a single element of the result matrix 𝐶. This involves accessing an entire row of matrix 𝐴 and an entire column of matrix 𝐵. The memory access pattern for this implementation is inefficient due to non-coalesced accesses and redundant memory loads.

Non-coalesced accesses occur because consecutive threads do not access consecutive memory locations. For example, if threads within a warp (a group of 32 threads executed simultaneously) each access a different row of 𝐴, their memory accesses are scattered across the entire row, leading to multiple memory transactions and higher latency.

Additionally, redundant memory loads happen because multiple threads read the same elements from global memory. For instance, when multiple threads compute elements of 𝐶 that share rows of 𝐴 or columns of 𝐵, they repeatedly load the same data from global memory. This results in unnecessary data transfers and further degrades performance.

## V2 : Shared Memory Utilization

The primary factors affecting speed are memory access and computation. In the naive kernel, the computation part is fixed because matrix multiplication requires a specific number of multiplications and additions. Therefore, the focus is on optimizing memory access, starting with using shared memory.

In the naive kernel, each thread reads an entire row of 𝐴 and an entire column of 𝐵, resulting in 2\*K\*\*\* sizeof(float) = 8\*K bytes of data read from global memory per thread, and this requires K multiplications and K additions.

To accelerate SGEMM operations, we need to reduce global memory accesses, but the naive kernel is the redundant global memory accesses. For example, threads computing 𝐶\[𝑚,𝑛\] and 𝐶\[𝑚,𝑛′\] both read the same row of 𝐴, and threads computing 𝐶\[𝑚,𝑛\]and 𝐶\[𝑚′,𝑛\] both read the same column of 𝐵.

The key idea is to divide the result matrix 𝐶 into smaller tiles and compute these tiles block by block. Each block of threads loads a sub-matrix (tile) of 𝐴 and 𝐵 into shared memory. By doing so, the same data is reused multiple times by different threads within a block, reducing the number of global memory accesses. As shared memory is limited, we can split matrix 𝐶 into blocks, each handled by a block of threads. We read BM x BK elements of 𝐴 and BK x BN elements of 𝐵 into shared memory. Each block of threads then computes a BM x BN sub-matrix of 𝐶. The compute-to-memory ratio of a block under this scheme is :

**Compute amount**:

$$2∗𝐾∗𝐵𝑀∗𝐵𝑁$$

**Memory access amount**:

$$(𝐵𝑀∗𝐾+𝐾∗𝐵𝑁)∗4 bytes$$

**Compute-to-memory ratio**:

$$\frac{1}{2 \left( \frac{1}{BM} + \frac{1}{BN} \right)}$$

​On average, each thread is responsible for 4 floats, which can be read as one float4. Reading according to float4 is actually an additional optimization method which can significantly enhance performance by improving memory coalescing and utilizing the GPU's SIMD (Single Instruction, Multiple Data) capabilities.

Memory coalescing reduces the number of memory transactions by combining multiple memory requests from threads in a warp into fewer transactions. When threads access consecutive memory locations, coalescing occurs, which enhances memory bandwidth utilization.

```c
#define FETCH_FLOAT4(var) (reinterpret_cast<float4 *>(&(var))[0])
```

Here, `FETCH_FLOAT4` is defined to fetch four consecutive floats as a single `float4` type.

GPUs are also designed to execute instructions on multiple data points simultaneously using SIMD architecture. Using `float4` leverages this architecture, allowing a single instruction to operate on four floats at once. This results in more efficient execution and better utilization of the GPU's computational resources.

```c
__global__ void sgemm_sharedmem(float *A, float *B, float *C, int M, int N, int K) {
    const int BM = 16, BN = 16, BK = 64;
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float c_value = 0.0f;

    for (int i = 0; i < K; i += BK) {
        // Load A's and B's tiles into shared memory
        if (row < M && (i + threadIdx.x) < K) {
            FETCH_FLOAT4(s_a[threadIdx.y][threadIdx.x]) = FETCH_FLOAT4(A[OFFSET(row, i + threadIdx.x, K)]);
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (i + threadIdx.y) < K) {
            FETCH_FLOAT4(s_b[threadIdx.y][threadIdx.x]) = FETCH_FLOAT4(B[OFFSET(i + threadIdx.y, col, N)]);
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
```

`float4` here is used to read four consecutive floats from global memory into shared memory, ensuring memory coalescing. Fetching multiple data elements in a single transaction also reduces memory latency, as instead of issuing multiple memory requests for individual floats, a single request fetches four floats, minimizing overhead and improving overall performance.

```c
        // Compute the partial product for this tile
        for (int j = 0; j < BK; ++j) {
            c_value += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to the output matrix
    C[row * N + col] = c_value;
}
```

In here, `__syncthreads()` is used to synchronize threads after loading data into shared memory and before performing the computation. This ensures all data is available for the computation phase, reducing latency associated with memory access.

Final implementation would look like this

## **V3 : Using Registers**

In the previous iteration, each iteration uses BM \* BK and BK \* BN shared memory blocks to compute a BM \* BN block of matrix 𝐶. Now, some might consider further optimizing shared memory by addressing bank conflicts. However, before tackling that, let's explore a similar optimization method using registers. Registers provide even faster memory access than shared memory.

We can further optimize SGEMM by moving data from shared memory to registers before computation. The concept is to divide the BM\*\*\* BN block into smaller TM \* TN blocks, where each smaller block is computed by a single thread. This approach leverages registers for faster data access during computation.

In this method, TK is set to 1, and TM and TN are set to 8. Instead of each thread computing one element of 𝐶, each thread now computes a TM \* TN block. This allows for larger BM and BN values (128 in this case), with BK adjusted accordingly (8 in this case).

The kernel defines constants for block sizes (BM, BN, BK) and thread sizes (TM, TN). Shared memory arrays `s_a` and `s_b` are allocated for sub-matrices of 𝐴 and 𝐵. Each thread computes its starting row and column in the output matrix 𝐶.

```c
int row = blockIdx.y * BM + threadIdx.y * TM;
int col = blockIdx.x * BN + threadIdx.x * TN;
```

Arrays `r_a` and `r_b` hold the elements loaded from shared memory to registers, and `r_c` accumulates the results.

```c
float r_a[TM], r_b[TN], r_c[TM][TN] = {0};
```

The kernel iterates over 𝐾 in steps of BK, loading tiles of 𝐴 and 𝐵 into shared memory using the `FETCH_FLOAT4` macro.

```c
for (int i = 0; i < K; i += BK) {
    for (int j = 0; j < BK; j += 4) {
        FETCH_FLOAT4(s_a[threadIdx.y * TM][j]) = FETCH_FLOAT4(A[row * K + i + j]);
        FETCH_FLOAT4(s_b[j][threadIdx.x * TN]) = FETCH_FLOAT4(B[(i + j) * N + col]);
    }
    __syncthreads();
```

Each thread computes the partial products using data loaded into registers from shared memory. This reduces the number of accesses to shared memory, leveraging the faster access speed of registers.

```c
for (int j = 0; j < BK; ++j) {
    FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[threadIdx.y * TM][j]);
    FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[j][threadIdx.x * TN]);
    for (int ti = 0; ti < TM; ++ti) {
        for (int tj = 0; tj < TN; ++tj) {
            r_c[ti][tj] += r_a[ti] * r_b[tj];
        }
    }
}
__syncthreads();
```

After the computation, each thread writes its computed block back to the global memory.

```c
for (int ti = 0; ti < TM; ++ti) {
    for (int tj = 0; tj < TN; ++tj) {
        C[(row + ti) * N + (col + tj)] = r_c[ti][tj];
    }
}
```

By using registers for intermediate storage, this implementation reduces the number of accesses to shared memory, further optimizing the performance of matrix multiplication on the GPU. The final implementation looks abit like this

## **V4 : Shared Memory Bank Conflicts in SGEMM**

In `sgemm_gpu_v3`, loading data from shared memory into registers can cause bank conflicts. For instance, when loading `s_a` into `r_a` in a warp, adjacent threads may access the same memory bank, leading to conflicts. Let's address these issues by changing the memory layout and read pattern.

```c
for (int k = 0; k < BK; k++) {
    // Load from s_a to r_a
    const int row_start = threadIdx.y * TM;
    for (int i = 0; i < TM; i++) {
        r_a[i] = s_a[row_start + i][k];
    }
    // Load from s_b to r_b
    // ...
}
```

For threads in the same warp, the adjacent threads in the y-direction of the thread block, when executing the line `r_a[i] = s_a[row_start + i][k]` simultaneously, read `s_a` addresses spaced by 𝐵𝐾×𝑇𝑀×sizeof(float)=256 bytes. This ensures they fall into the same bank, causing conflicts.

The calculation formula involves multiplying by BK because `s_a` is row-major. If `s_a` is changed to column-major, the address difference would be 𝑇𝑀×sizeof(float)=32 bytes. Given the thread block size of (BM / TM, BN / TN), i.e., 16 \* 16, threads in the same warp differ by at most 1 in the y-direction, eliminating bank conflicts.

To resolve this, we can change `s_a` to column-major order and adjust the read pattern. . After this change, when loading from shared memory to registers, each thread reads consecutive addresses in shared memory, which allows the use of `float4` for reading instead of loop-based reading as in `sgemm_gpu_v3`.

```c
for (int k = 0; k < BK; k++) {
    // Load from s_a to r_a
    const int row_start = threadIdx.y * TM;
    FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[k][row_start]);
    FETCH_FLOAT4(r_a[4]) = FETCH_FLOAT4(s_a[k][row_start + 4]);
    // Load from s_b to r_b
    const int col_start = threadIdx.x * TN;
    FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[k][col_start]);
    FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[k][col_start + 4]);
    // ......
}
```

Since matrix A is row-major, while `s_a` is column-major, the code for loading A into `s_a` needs a slight modification. It is more cumbersome than the code for loading B into `s_b`.

```c
for (int step = 0; step < K / BK; step++) {
    // Load from A to s_a
    const int col_A = step * BK + col_s_a;
    const int index_A = OFFSET(row_A, col_A, K);
    FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(A[index_A]); // Use r_a[0] as a temporary buffer
    s_a[col_s_a + 0][row_s_a] = r_a[0];
    s_a[col_s_a + 1][row_s_a] = r_a[1];
    s_a[col_s_a + 2][row_s_a] = r_a[2];
    s_a[col_s_a + 3][row_s_a] = r_a[3];
    // Load from B to s_b
    const int row_B = step * BK + row_s_b;
    const int index_B = OFFSET(row_B, col_B, N);
    FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
    __syncthreads();
    // ......
}
```

But although we have resolved the bank conflict when reading `s_a` by making it column-major, we have introduced a bank conflict when writing to `s_a` in the above code. This can be partially mitigated by using padding, but a better solution for write conflicts can be achieved using a permuted approach, which involves rearranging the indices used to access shared memory in such a way that consecutive threads write to different memory banks. This permutation helps distribute memory accesses more evenly across the banks, thereby reducing or eliminating conflicts.

```c
__global__ void permutedWrite(float *A, float *C, int N) {
    __shared__ float s_a[32][32]; // Assuming a 32x32 shared memory tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
```

We declare `s_a` as a 32x32 shared memory array, which will be used to hold portions of matrix data. The thread indices `tx` and `ty` correspond to the x and y coordinates within the thread block.

```c
    // Permutation function (XOR with a constant)
    int permuted_tx = tx ^ (ty & 1);
```

To reduce bank conflicts, we apply a permutation function to the thread index `tx`. This function distributes memory accesses more evenly across different banks. The expression `tx ^ (ty & 1)` is a bitwise XOR operation. The `& 1` operation extracts the least significant bit of `ty`, ensuring that it toggles between 0 and 1. This permutation function effectively redistributes the write indices, helping to avoid conflicts where multiple threads would otherwise access the same bank.

Next, we calculate the indices for accessing global and shared memory.

```c
    // Index calculation
    int index_A = ty * N + tx;
    int index_s = permuted_tx * 32 + ty;
```

`index_A` computes the global memory index for matrix A. Here, `ty * N + tx` maps the 2D thread indices into a 1D array index. `index_s` computes the index for writing into the shared memory array `s_a`. By using `permuted_tx` instead of `tx`, we ensure that the writes are permuted across the shared memory banks.

The kernel then loads data from global memory into a register and writes it to shared memory using the permuted index.

```c
    // Load from global memory to register
    float reg = A[index_A];

    // Write from register to shared memory using permuted index
    s_a[permuted_tx][ty] = reg;
```

The value from global memory `A[index_A]` is loaded into the register `reg`. This value is then written to the shared memory `s_a` at the position `[permuted_tx][ty]`, ensuring the permutation helps to spread accesses across different memory banks. This permuted method significantly reduces write-time bank conflicts by ensuring that consecutive threads access different banks.

After solving the bank conflict for `s_a`, we need to optimize `s_b` as well. Similarly, we analyze the bank conflict for `s_b` based on the `sgemm_gpu_v4` code. The bank conflict for `s_b` occurs during reading.

```c
// Bank conflict code when reading s_b in sgemm_gpu_v4
for (int k = 0; k < BK; k++) {
    // Load from s_a to r_a
    // ......
    // Load from s_b to r_b
    const int col_start = threadIdx.x * TN;
    FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[k][col_start]);
    FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[k][col_start + 4]);
    // Compute
    // ......
}
```

For two adjacent threads in the x-direction of the thread block within the same warp, reading `s_b[k][col_start]` differs by `TN * sizeof(float) = 32` bytes, i.e., 8 banks. Since shared memory is organized into 32 banks, every 32/8 = 4 threads will cause a bank conflict. Given that there are 16 threads in the x-direction within a warp, this results in up to 4-way bank conflicts. Since we are using `float4` to read from `s_b`, this actually results in 2-way bank conflicts. The reason it's 2-way instead of 4-way is that the number of conflicts is calculated [based on transactions](https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900).

The code involves two `float4` reads from `s_b`, each thread reading consecutive `float4` elements. Specifically, for a `float4` read like `FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[k][col_start])`, adjacent threads in the x-direction within a warp read `float4` elements that are one `float4` apart. This means some banks are underutilized, but we can aim to have adjacent threads read adjacent data, fully utilizing all banks and minimizing bank conflicts.

```c
// Load from s_b to r_b, v5 compared to v4, read positions have changed
const int col_start = threadIdx.x * (TN / 2);
FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[k][col_start]);
FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[k][col_start + BN / 2]);
```

In this code, the calculation for `col_start` has changed. When executing `FETCH_FLOAT4(s_b[k][col_start])`, adjacent threads read adjacent `float4` elements in `s_b`. Similarly, `FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[k][col_start + BN / 2])` ensures adjacent reads. This is the "rearrangement of read positions for `s_b`" mentioned in the section title. In this case, the data read from `s_b` is exactly the same as before, just in different positions, but it can still be used to compute `r_c`. The computed `r_c` is then stored in matrix C at an adjusted position accordingly.

The final implementation looks like this.

## V5: Using Double Buffers

In a typical CUDA implementation, a single buffer is used for loading data and performing computations sequentially within a loop:

```c
float buffer[N];
for(int i = 0; i < NUM; i++) {
    load_data_to_buffer(buffer);
    compute(buffer);
}
```

This approach results in idle periods where the GPU waits for data to be loaded into the buffer before it can start computation, which can be a bottleneck. Double buffering addresses this issue by using two buffers, allowing data loading and computation to overlap.

```c
float buffers[2][N];
load_data_to_buffer(buffers[0]);  // Initial load
for (int i = 1; i < NUM; i++) {
    int load_index = i % 2;
    load_data_to_buffer(buffers[load_index]);  // Load data for the next iteration
    int compute_index = (i - 1) % 2;
    compute(buffers[compute_index]);  // Compute using data from the previous iteration
}
int compute_index = (NUM - 1) % 2;
compute(buffers[compute_index]);  // Final computation for the last loaded data
```

When implementing double buffering in CUDA, shared memory is often used due to its low latency compared to global memory.

In this example `s_a` and `s_b` are double buffers in shared memory. Data is loaded into `s_a[0]` and `s_b[0]` initially. Then during each iteration, data is loaded into one buffer (`load_index`) while computation is performed on the other buffer (`compute_index`).

The reason for doubling the buffer is to achieve instruction parallelization through prefetching. In simple terms, it means that in one loop, the data being loaded and the data used for computation are placed in two separate buffers, which can hide memory access latency.
