//
// Created by dsagbili on 21/06/2024.
//

#ifndef __UNICONN_INCLUDE_UNICONN_UTILS_HPP_
#define __UNICONN_INCLUDE_UNICONN_UTILS_HPP_
#include <Unc_config.hpp>
#include <cassert>
// #include <concepts>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// #if defined UNC_HAS_ROCM
// #include <hip/hip_fp16.h>
// #include <hip/hip_bf16.h>
// using Unc_bfloat16 = __hip_bfloat16;

// #elif defined UNC_HAS_CUDA
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// using Unc_bfloat16 = __nv_bfloat16;
// #endif

#ifndef GPUTAGS
#define GPUTAGS
#if !defined __CUDACC__ && !defined __HIPCC__
#define GPU_HOST
#define GPU_DEVICE
#define GPU_UNIFIED
#define GPU_KERNEL
#else
#define GPU_HOST __host__
#define GPU_DEVICE __device__
#define GPU_UNIFIED GPU_HOST GPU_DEVICE
#define GPU_KERNEL __global__
#endif
#endif


namespace uniconn {
enum class ReductionOperator { SUM, PROD, MIN, MAX, LOR, LAND, LXOR, BOR, BAND, BXOR, AVG };
enum class ThreadGroup { THREAD, WARP, BLOCK, CLUSTER };
enum class LaunchMode { HostDriven, LimitedDevice, FullDevice };

struct MPIBackend;
#if defined(UNC_HAS_GPUCCL)
struct GpucclBackend;
#endif
#if defined(UNC_HAS_GPUSHMEM)
struct GpushmemBackend;
#endif

namespace internal {

template <typename T>
T *IN_PLACE = (T *)(-1);
}  // namespace internal
#ifndef UNICONN_USE_MPI
#define UNICONN_USE_MPI 1
#endif

#ifndef UNICONN_USE_GPUCCL
#define UNICONN_USE_GPUCCL 0
#endif

#ifndef UNICONN_USE_GPUSHMEM
#define UNICONN_USE_GPUSHMEM 0
#endif

#ifndef UNICONN_USE_HOST_ONLY
#define UNICONN_USE_HOST_ONLY 1
#endif

#ifndef UNICONN_USE_LIMITED_DEVICE
#define UNICONN_USE_LIMITED_DEVICE 0
#endif

#ifndef UNICONN_USE_FULL_DEVICE
#define UNICONN_USE_FULL_DEVICE 0
#endif


#if UNICONN_USE_GPUSHMEM
using DefaultBackend = GpushmemBackend;
#if UNICONN_USE_LIMITED_DEVICE
constexpr LaunchMode DefaultLaunchType = LaunchMode::LimitedDevice;
#elif UNICONN_USE_FULL_DEVICE
constexpr LaunchMode DefaultLaunchType = LaunchMode::FullDevice;
#else
constexpr LaunchMode DefaultLaunchType = LaunchMode::HostDriven;
#endif
#elif UNICONN_USE_GPUCCL 
using DefaultBackend = GpucclBackend;
constexpr LaunchMode DefaultLaunchType = LaunchMode::HostDriven;
#else
using DefaultBackend = MPIBackend;
constexpr LaunchMode DefaultLaunchType = LaunchMode::HostDriven;
#endif

#define GPU_RT_CALL(call)                                                                   \
    {                                                                                       \
        UncGpuError_t gpuStatus = call;                                                     \
        if (UncGpuSuccess != gpuStatus) {                                                   \
            fprintf(stderr,                                                                 \
                    "ERROR: Uniconn GPU RT call \"%s\" in line %d of file %s failed "       \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, UncGpuGetErrorString(gpuStatus), gpuStatus); \
            exit(gpuStatus);                                                                \
        }                                                                                   \
    }

}  // namespace uniconn
#endif  // __UNICONN_INCLUDE_UNICONN_UTILS_HPP_
