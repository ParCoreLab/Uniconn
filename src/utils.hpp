#ifndef __UNICONN_SRC_UTILS_HPP_
#define __UNICONN_SRC_UTILS_HPP_

#include "uniconn/utils.hpp"
// #define MAX_SIZE 100

// #define GPU_RT_CALL(call)                                                                   \
//     {                                                                                       \
//         UncGpuError_t gpuStatus = call;                                                     \
//         if (UncGpuSuccess != gpuStatus) {                                                   \
//             fprintf(stderr,                                                                 \
//                     "ERROR: Uniconn GPU RT call \"%s\" in line %d of file %s failed "       \
//                     "with "                                                                 \
//                     "%s (%d).\n",                                                           \
//                     #call, __LINE__, __FILE__, UncGpuGetErrorString(gpuStatus), gpuStatus); \
//             exit(gpuStatus);                                                                \
//         }                                                                                   \
//     }

// template <typename T>
// GPU_UNIFIED inline T accumulate(const T *v, const int size) {
//     T ret = 0;

//     for (size_t i = 0; i < size; ++i) {
//         ret = v[i];
//     }
//     return ret;
// }

// /**
//  * Compute an exclusive prefix sum.
//  *
//  * This is mostly meant to help with vector collectives.
//  */
// template <typename T>
// GPU_UNIFIED inline T *excl_prefix_sum(const T *v, const int size, T *r) {
//     T r[MAX_SIZE];

//     for (size_t i = 0; i < size; i++) {
//         r[i] = 0;
//     }

//     for (size_t i = 1; i < size; ++i) {
//         r[i] = v[i - 1] + r[i - 1];
//     }
//     return r;
// }

#endif  // __UNICONN_SRC_UTILS_HPP_