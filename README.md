# Uniconn: A Uniform High-Level Communication Library for Portable Multi-GPU Programming

Modern HPC and AI systems increasingly rely on multi-GPU clusters, where communication libraries such as MPI, NCCL/RCCL, and NVSHMEM enable data movement across GPUs. While these libraries are widely used in frameworks and solver packages, their distinct APIs, synchronization models, and integration mechanisms introduce programming complexity and limit portability. Performance also varies across workloads and system architectures, making it difficult to achieve consistent efficiency. These issues present a significant obstacle to writing portable, high-performance code for large-scale GPU systems.


We present Uniconn, a unified, portable high-level C++ communication library that supports both point-to-point and collective operations across GPU clusters. Uniconn enables seamless switching between backends and APIs (host or device) with minimal or no changes to application code. We describe its design and core constructs, and evaluate its performance using network benchmarks, a Jacobi solver, and a Conjugate Gradient solver. Across three supercomputers, we compare Uniconn's overhead against CUDA/ROCm-aware MPI, NCCL/RCCL, and NVSHMEM on up to 64 GPUs. In most cases, Uniconn incurs negligible overhead, typically under 1% for the Jacobi solver and under 2% for the Conjugate Gradient solver.



## Dependencies
### Required
* CMake 3.25 or above
NVIDIA GPUS:
* CUDA 11.8 or above 
* CUDA/ROCM-aware MPI
AMD GPUS:
* ROCM 6.0.3 or above
* ROCM-aware MPI
### Optional
NVIDIA GPUS:
* NCCL 2.7.0 or above
* NVSHMEM 2.8.0 or above
AMD GPUS:
* RCCL 2.14.0 or above



## Build
NVIDIA GPUS:
```bash
$ cmake -S . -B build -DUNICONN_ENABLE_CUDA=ON -DUNICONN_ENABLE_GPUCCL={ON/OFF} -DUNICONN_ENABLE_GPUSHMEM={ON/OFF} -DUNICONN_ENABLE_EXAMPLES={ON/OFF} -DCMAKE_INSTALL_PREFIX={install_dir} 
$ cd build
$ make -j install
```
AMD GPUS:

```bash
$ cmake -S . -B build -DUNICONN_ENABLE_ROCM=ON -DUNICONN_ENABLE_GPUCCL={ON/OFF}  -DUNICONN_ENABLE_EXAMPLES={ON/OFF}  -DCMAKE_INSTALL_PREFIX={install_dir}
$ cd build 
$ make install -j
```

## Usage

Before compiling your application with Uniconn, There are available environment variables to use underlying backend library as default selection. 
* For CUDA/ROCM MPI:
```bash
UNICONN_USE_MPI=1
UNICONN_USE_HOST_ONLY=1
```
* For NCCL/RCCL:
```bash
UNICONN_USE_GPUCCL=1
UNICONN_USE_HOST_ONLY=1
```
* For NVSHMEM:
```bash
UNICONN_USE_GPUSHMEM=1
UNICONN_USE_HOST_ONLY=1 or UNICONN_USE_LIMITED_DEVICE=1 or UNICONN_USE_FULL_DEVICE=1
```


# Acknowledgment
This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 949587).