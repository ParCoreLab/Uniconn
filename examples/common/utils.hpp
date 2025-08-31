
#include <Uniconn.hpp>
//  #include <cuda_runtime.h>
//  #include <mpi.h>
//  #include <nccl.h>
//  #include <nvshmem.h>
//  #include <nvshmemx.h>
#include <stdint.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <sstream>
#include <ctime>
#include <cstring>
#include <cstdbool>
#include <cerrno>

#define DEFAULT_REPT 3
#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                              \
    {                                                      \
        int color_id = cid;                                \
        color_id = color_id % num_colors;                  \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[color_id];              \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

// #define GPU_RT_CALL(call)                                                                         \
//     {                                                                                             \
//         UncGpuError_t UncGpuStatus = call;                                                        \
//         if (UncGpuSuccess != UncGpuStatus) {                                                      \
//             fprintf(stderr,                                                                       \
//                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                    \
//                     "with "                                                                       \
//                     "%s (%d).\n",                                                                 \
//                     #call, __LINE__, __FILE__, UncGpuGetErrorString(UncGpuStatus), UncGpuStatus); \
//             exit(UncGpuStatus);                                                                   \
//         }                                                                                         \
//     }

// #define MPI_CALL(call)                                                                \
//     {                                                                                 \
//         int mpi_status = call;                                                        \
//         if (MPI_SUCCESS != mpi_status) {                                              \
//             char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
//             int mpi_error_string_length = 0;                                          \
//             MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
//             if (NULL != mpi_error_string)                                             \
//                 fprintf(stderr,                                                       \
//                         "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
//                         "with %s "                                                    \
//                         "(%d).\n",                                                    \
//                         #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
//             else                                                                      \
//                 fprintf(stderr,                                                       \
//                         "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
//                         "with %d.\n",                                                 \
//                         #call, __LINE__, __FILE__, mpi_status);                       \
//             exit(mpi_status);                                                         \
//         }                                                                             \
//     }

// #define NCCL_CALL(call)                                                                     \
//     {                                                                                       \
//         ncclResult_t ncclStatus = call;                                                     \
//         if (ncclSuccess != ncclStatus) {                                                    \
//             fprintf(stderr,                                                                 \
//                     "ERROR: NCCL call \"%s\" in line %d of file %s failed "                 \
//                     "with "                                                                 \
//                     "%s (%d).\n",                                                           \
//                     #call, __LINE__, __FILE__, ncclGetErrorString(ncclStatus), ncclStatus); \
//             exit(ncclStatus);                                                               \
//         }                                                                                   \
//     }

template <typename T>
T get_argval(char **begin, char **end, const std::string &arg, const T default_val) {
    T argval = default_val;
    char **itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

// convert NVSHMEM_SYMMETRIC_SIZE string to long long unsigned int
long long unsigned int parse_nvshmem_symmetric_size(char *value) {
    long long unsigned int units, size;

    assert(value != NULL);

    if (strchr(value, 'G') != NULL) {
        units = 1e9;
    } else if (strchr(value, 'M') != NULL) {
        units = 1e6;
    } else if (strchr(value, 'K') != NULL) {
        units = 1e3;
    } else {
        units = 1;
    }

    assert(atof(value) >= 0);
    size = (long long unsigned int)atof(value) * units;

    return size;
}