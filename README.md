# GPU-Accelerated ML Kernels with CUDA

> **CPU vs Naive CUDA vs Optimized CUDA (Shared Memory) + Nsight Profiling**

A from-scratch implementation of core ML operations in CUDA with scientific performance analysis and profiling-driven optimization.

## Motivation

Modern ML frameworks like PyTorch and TensorFlow rely heavily on GPU kernels for performance, but their internal behavior is often abstracted away. This project aims to:

- **Understand** ML core operations by implementing them from scratch using CUDA
- **Measure** performance scientifically across different optimization levels
- **Validate** optimization decisions using NVIDIA Nsight Compute profiling

## What This Project Does

### Implements Core ML Operations:
- **Vector Addition**
- **Matrix Multiplication** 
- **2D Convolution (Conv2D)**

### Performance Comparison:
- **Optimized CPU** (C++ baseline)
- **Naive CUDA** kernels
- **Shared-memory CUDA** kernels

### Scientific Analysis:
- Uses **Nsight Compute** to explain why certain optimizations work — or don't

## Implemented Kernels

### CPU (Baseline)
- Vector Add
- Matrix Multiplication  
- Conv2D

### CUDA (Naive)
- Thread-per-element mapping
- Global memory access only

### CUDA (Optimized)
- Matrix multiplication using shared memory tiling
- Tile-size experiments (16×16 vs 32×32)

## Performance Summary

| Operation | CPU | CUDA Naive | CUDA Shared |
|-----------|-----|------------|-------------|
| **Vector Add** (16M) | ~160 ms | ~31 ms | — |
| **Matrix Mul** (1024²) | ~11.7 s | ~124 ms | **~31 ms** |
| **Conv2D** (512×512, 3×3) | — | **~16 ms** | ~26 ms |

## Nsight Profiling Insights

### Matrix Multiplication
- **Problem**: Naive kernel was memory-bound with redundant global loads
- **Solution**: Shared-memory tiling reduced global memory traffic and improved reuse
- **Nsight Results**:
  - Reduced memory stalls
  - Improved arithmetic intensity  
  - **~4× speedup** (11.7s → 124ms → 31ms)
- **Conclusion**: Shared memory proved effective due to high data reuse

### Convolution (3×3)
- **Naive Conv2D** already achieved:
  - 86% memory throughput
  - 80% occupancy
- **Shared-memory Conv2D** was **slower** due to:
  - Synchronization overhead
  - Low arithmetic intensity
  - Effective L1/L2 cache usage in naive version
- **Conclusion**: Shared memory is not universally beneficial, especially for small kernels

## Key Takeaways

> **GPU optimization must be profiling-driven, not assumption-driven**

- Shared memory helps only when **data reuse outweighs synchronization cost**
- **Nsight Compute** is essential for understanding GPU performance bottlenecks
- **Small kernels** may already be well-optimized by hardware caches

## Tools & Tech Stack

- **Languages**: C++, CUDA
- **Profiling**: NVIDIA Nsight Compute
- **Development**: Visual Studio, NVCC
- **Hardware**: NVIDIA GPU (CUDA-capable)

## Repository Structure

```
gpu-ml-kernels/
├── cpu/                    # CPU baseline implementations
├── cuda/                   # CUDA kernel implementations
├── profiling/              # Nsight profiling reports
│   ├── matmul_naive.ncu-rep
│   ├── matmul_shared_32.ncu-rep
│   └── conv2d_analysis.ncu-rep
├── benchmarks/             # Performance measurement scripts
└── README.md
```

## Future Work

- [ ] **MNIST inference** using custom CUDA Conv + ReLU pipeline
- [ ] **End-to-end GPU benchmarking** scripts
- [ ] **Comparison with cuBLAS/cuDNN** highly-optimized libraries
- [ ] **Kernel fusion** experiments
- [ ] **Mixed precision** (FP16/FP32) implementations

## 🏃‍♂️ Getting Started

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit (11.0+)
- NVIDIA Nsight Compute (for profiling)

### Build & Run
```bash
# Clone the repository
git clone https://github.com/princ0301/cuda_gpu_kernel.git
cd cuda_gpu_kernel

# Compile CUDA kernels
nvcc -o matmul cuda/matmul_shared.cu
nvcc -o conv2d cuda/conv2d_shared.cu

# Run benchmarks
./matmul
./conv2d
```

### Profiling
```bash
# Profile with Nsight Compute
ncu --set full -o matmul_profile ./matmul
ncu-ui matmul_profile.ncu-rep
```

---