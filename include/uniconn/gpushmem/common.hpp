#ifndef __UNICONN_INCLUDE_UNICONN_GPUSHMEM_COMMON_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUSHMEM_COMMON_HPP_

#include "../utils.hpp"
#include "uniconn/gpumpi/common.hpp"

#if defined(UNC_HAS_ROCM)
#include <roc_shmem.h>
#elif defined(UNC_HAS_CUDA)
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace uniconn {
struct GpushmemBackend {};
}  // namespace uniconn

///////////////////////////TEMPLATE INST MACROS///////////////////////////////

#define UNC_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(FN_TEMPLATE) \
    FN_TEMPLATE(float, float)                              \
    FN_TEMPLATE(double, double)                            \
    FN_TEMPLATE(char, char)                                \
    FN_TEMPLATE(short, short)                              \
    FN_TEMPLATE(schar, signed char)                        \
    FN_TEMPLATE(int, int)                                  \
    FN_TEMPLATE(long, long)                                \
    FN_TEMPLATE(longlong, long long)                       \
    FN_TEMPLATE(uchar, unsigned char)                      \
    FN_TEMPLATE(ushort, unsigned short)                    \
    FN_TEMPLATE(uint, unsigned int)                        \
    FN_TEMPLATE(ulong, unsigned long)                      \
    FN_TEMPLATE(ulonglong, unsigned long long)

#define UNC_SHMEM_REPT_STANDARD_TYPE_HOST(FN_TEMPLATE, MODE) \
    FN_TEMPLATE(float, float, MODE)                          \
    FN_TEMPLATE(double, double, MODE)                        \
    FN_TEMPLATE(char, char, MODE)                            \
    FN_TEMPLATE(short, short, MODE)                          \
    FN_TEMPLATE(schar, signed char, MODE)                    \
    FN_TEMPLATE(int, int, MODE)                              \
    FN_TEMPLATE(long, long, MODE)                            \
    FN_TEMPLATE(longlong, long long, MODE)                   \
    FN_TEMPLATE(uchar, unsigned char, MODE)                  \
    FN_TEMPLATE(ushort, unsigned short, MODE)                \
    FN_TEMPLATE(uint, unsigned int, MODE)                    \
    FN_TEMPLATE(ulong, unsigned long, MODE)                  \
    FN_TEMPLATE(ulonglong, unsigned long long, MODE)

#define UNC_SHMEM_REPT_STANDARD_TYPE_MODE_HOST(FN_TEMPLATE)                   \
    UNC_SHMEM_REPT_STANDARD_TYPE_HOST(FN_TEMPLATE, LaunchMode::HostDriven)    \
    UNC_SHMEM_REPT_STANDARD_TYPE_HOST(FN_TEMPLATE, LaunchMode::LimitedDevice) \
    UNC_SHMEM_REPT_STANDARD_TYPE_HOST(FN_TEMPLATE, LaunchMode::FullDevice)

#define UNC_SHMEM_REPT_STANDARD_TYPE_DEVICE(FN_TEMPLATE, MODE, SCOPE) \
    FN_TEMPLATE(float, float, MODE, SCOPE)                            \
    FN_TEMPLATE(double, double, MODE, SCOPE)                          \
    FN_TEMPLATE(char, char, MODE, SCOPE)                              \
    FN_TEMPLATE(short, short, MODE, SCOPE)                            \
    FN_TEMPLATE(schar, signed char, MODE, SCOPE)                      \
    FN_TEMPLATE(int, int, MODE, SCOPE)                                \
    FN_TEMPLATE(long, long, MODE, SCOPE)                              \
    FN_TEMPLATE(longlong, long long, MODE, SCOPE)                     \
    FN_TEMPLATE(uchar, unsigned char, MODE, SCOPE)                    \
    FN_TEMPLATE(ushort, unsigned short, MODE, SCOPE)                  \
    FN_TEMPLATE(uint, unsigned int, MODE, SCOPE)                      \
    FN_TEMPLATE(ulong, unsigned long, MODE, SCOPE)                    \
    FN_TEMPLATE(ulonglong, unsigned long long, MODE, SCOPE)

#define UNC_SHMEM_REPT_STANDARD_TYPE_GROUP_DEVICE(FN_TEMPLATE, MODE)            \
    UNC_SHMEM_REPT_STANDARD_TYPE_DEVICE(FN_TEMPLATE, MODE, ThreadGroup::THREAD) \
    UNC_SHMEM_REPT_STANDARD_TYPE_DEVICE(FN_TEMPLATE, MODE, ThreadGroup::WARP)   \
    UNC_SHMEM_REPT_STANDARD_TYPE_DEVICE(FN_TEMPLATE, MODE, ThreadGroup::BLOCK)

#define UNC_SHMEM_REPT_STANDARD_TYPE_MODE_GROUP_DEVICE(FN_TEMPLATE)                   \
    UNC_SHMEM_REPT_STANDARD_TYPE_GROUP_DEVICE(FN_TEMPLATE, LaunchMode::HostDriven)    \
    UNC_SHMEM_REPT_STANDARD_TYPE_GROUP_DEVICE(FN_TEMPLATE, LaunchMode::LimitedDevice) \
    UNC_SHMEM_REPT_STANDARD_TYPE_GROUP_DEVICE(FN_TEMPLATE, LaunchMode::FullDevice)


#define UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_HOST(FN_TEMPLATE, MODE, OPNAME, OP) \
    FN_TEMPLATE(uchar, unsigned char, OPNAME, OP, MODE)                               \
    FN_TEMPLATE(ushort, unsigned short, OPNAME, OP, MODE)                             \
    FN_TEMPLATE(uint, unsigned int, OPNAME, OP, MODE)                                 \
    FN_TEMPLATE(ulong, unsigned long, OPNAME, OP, MODE)                               \
    FN_TEMPLATE(ulonglong, unsigned long long, OPNAME, OP, MODE)

#define UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_HOST(FN_TEMPLATE, MODE, OPNAME, OP) \
    FN_TEMPLATE(float, float, OPNAME, OP, MODE)                                     \
    FN_TEMPLATE(double, double, OPNAME, OP, MODE)                                   \
    FN_TEMPLATE(char, char, OPNAME, OP, MODE)                                       \
    FN_TEMPLATE(short, short, OPNAME, OP, MODE)                                     \
    FN_TEMPLATE(schar, signed char, OPNAME, OP, MODE)                               \
    FN_TEMPLATE(int, int, OPNAME, OP, MODE)                                         \
    FN_TEMPLATE(long, long, OPNAME, OP, MODE)                                       \
    FN_TEMPLATE(longlong, long long, OPNAME, OP, MODE)

#define UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_MODE_HOST(FN_TEMPLATE, OPNAME, OP)                   \
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_HOST(FN_TEMPLATE, LaunchMode::HostDriven, OPNAME, OP)    \
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_HOST(FN_TEMPLATE, LaunchMode::LimitedDevice, OPNAME, OP) \
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_HOST(FN_TEMPLATE, LaunchMode::FullDevice, OPNAME, OP)

#define UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_MODE_HOST(FN_TEMPLATE, OPNAME, OP)                   \
    UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_HOST(FN_TEMPLATE, LaunchMode::HostDriven, OPNAME, OP)    \
    UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_HOST(FN_TEMPLATE, LaunchMode::LimitedDevice, OPNAME, OP) \
    UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_HOST(FN_TEMPLATE, LaunchMode::FullDevice, OPNAME, OP)

#define UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_DEVICE(FN_TEMPLATE, MODE, SCOPE, OPNAME, OP) \
    FN_TEMPLATE(uchar, unsigned char, OPNAME, OP, MODE, SCOPE)                                 \
    FN_TEMPLATE(ushort, unsigned short, OPNAME, OP, MODE, SCOPE)                               \
    FN_TEMPLATE(uint, unsigned int, OPNAME, OP, MODE, SCOPE)                                   \
    FN_TEMPLATE(ulong, unsigned long, OPNAME, OP, MODE, SCOPE)                                 \
    FN_TEMPLATE(ulonglong, unsigned long long, OPNAME, OP, MODE, SCOPE)

#define UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_DEVICE(FN_TEMPLATE, MODE, SCOPE, OPNAME, OP) \
    FN_TEMPLATE(char, char, OPNAME, OP, MODE, SCOPE)                                         \
    FN_TEMPLATE(schar, signed char, OPNAME, OP, MODE, SCOPE)                                 \
    FN_TEMPLATE(short, short, OPNAME, OP, MODE, SCOPE)                                       \
    FN_TEMPLATE(int, int, OPNAME, OP, MODE, SCOPE)                                           \
    FN_TEMPLATE(long, long, OPNAME, OP, MODE, SCOPE)                                         \
    FN_TEMPLATE(longlong, long long, OPNAME, OP, MODE, SCOPE)                                \
    FN_TEMPLATE(float, float, OPNAME, OP, MODE, SCOPE)                                       \
    FN_TEMPLATE(double, double, OPNAME, OP, MODE, SCOPE)

#define UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_GROUP_DEVICE(FN_TEMPLATE, MODE, OPNAME, OP)            \
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_DEVICE(FN_TEMPLATE, MODE, ThreadGroup::THREAD, OPNAME, OP) \
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_DEVICE(FN_TEMPLATE, MODE, ThreadGroup::WARP, OPNAME, OP)   \
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_DEVICE(FN_TEMPLATE, MODE, ThreadGroup::BLOCK, OPNAME, OP)

#define UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_GROUP_DEVICE(FN_TEMPLATE, MODE, OPNAME, OP)            \
    UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_DEVICE(FN_TEMPLATE, MODE, ThreadGroup::THREAD, OPNAME, OP) \
    UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_DEVICE(FN_TEMPLATE, MODE, ThreadGroup::WARP, OPNAME, OP)   \
    UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_DEVICE(FN_TEMPLATE, MODE, ThreadGroup::BLOCK, OPNAME, OP)

#define UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_MODE_GROUP_DEVICE(FN_TEMPLATE, OPNAME, OP)                   \
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_GROUP_DEVICE(FN_TEMPLATE, LaunchMode::HostDriven, OPNAME, OP)    \
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_GROUP_DEVICE(FN_TEMPLATE, LaunchMode::LimitedDevice, OPNAME, OP) \
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_GROUP_DEVICE(FN_TEMPLATE, LaunchMode::FullDevice, OPNAME, OP)

#define UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_MODE_GROUP_DEVICE(FN_TEMPLATE, OPNAME, OP)                   \
    UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_GROUP_DEVICE(FN_TEMPLATE, LaunchMode::HostDriven, OPNAME, OP)    \
    UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_GROUP_DEVICE(FN_TEMPLATE, LaunchMode::LimitedDevice, OPNAME, OP) \
    UNC_SHMEM_REPT_FOR_ARITH_REDUCE_TYPE_OP_GROUP_DEVICE(FN_TEMPLATE, LaunchMode::FullDevice, OPNAME, OP)


#endif  // __UNICONN_INCLUDE_UNICONN_GPUSHMEM_COMMON_HPP_
