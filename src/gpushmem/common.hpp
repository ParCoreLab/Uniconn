#ifndef __UNICONN_CORE_SRC_GPUSHMEM_COMMON_HPP_
#define __UNICONN_CORE_SRC_GPUSHMEM_COMMON_HPP_

#include "../gpumpi/common.hpp"
#include "../utils.hpp"
#include "uniconn/gpushmem/common.hpp"

#if defined(UNC_HAS_ROCM)
#include <roc_shmem.h>
#elif defined(UNC_HAS_CUDA)
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#define UNC_CONCURRENT_P2P_SIGNAL_COUNT 128

#ifdef UNC_HAS_ROCM
#define UNC_SHMEM_PREFIX roc_shmem
#elif 1
#define UNC_SHMEM_PREFIX nvshmem
#endif

// The layers of indirection begin
#ifdef UNC_SHMEM_PREFIX
#define UNC_GPU_RT_SHMEM(symb) UNC_ADD_PREFIX(UNC_SHMEM_PREFIX, symb)

// Functions

// Types

// Enum values

// Special APIs
#if UNC_HAS_ROCM

#elif 1

#endif
#endif  // defined UNC_SHMEM_PREFIX


#endif  // __UNICONN_CORE_SRC_GPUSHMEM_COMMON_HPP_