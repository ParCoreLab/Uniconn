#ifndef __UNICONN_INCLUDE_UNICONN_GPUCCL_COMMON_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUCCL_COMMON_HPP_

#include "uniconn/gpumpi/common.hpp"
#include "uniconn/utils.hpp"
#if defined(UNC_HAS_ROCM)
#include <rccl/rccl.h>
#elif defined(UNC_HAS_CUDA)
#include <nccl.h>
#endif

#define NCCL_CALL(call)                                                                                              \
    {                                                                                                                \
        ncclResult_t nccl_result = (call);                                                                           \
        if (nccl_result != ncclSuccess) {                                                                            \
            fprintf(stderr, "ERROR: NCCL call \"%s\" in line %d of file %s failed with %s (%d).\n", #call, __LINE__, \
                    __FILE__, ncclGetErrorString(nccl_result), nccl_result);                                         \
            exit(nccl_result);                                                                                       \
        }                                                                                                            \
    }
namespace uniconn {
struct GpucclBackend {};

namespace internal {
namespace gpuccl {

template <ReductionOperator OP>
inline ncclRedOp_t ReductOp2ncclRedOp();

template <>
inline ncclRedOp_t ReductOp2ncclRedOp<ReductionOperator::SUM>() {
    return ncclSum;
}
template <>
inline ncclRedOp_t ReductOp2ncclRedOp<ReductionOperator::PROD>() {
    return ncclProd;
}
template <>
inline ncclRedOp_t ReductOp2ncclRedOp<ReductionOperator::MIN>() {
    return ncclMin;
}
template <>
inline ncclRedOp_t ReductOp2ncclRedOp<ReductionOperator::MAX>() {
    return ncclMax;
}
template <>
inline ncclRedOp_t ReductOp2ncclRedOp<ReductionOperator::AVG>() {
    return ncclAvg;
}

template <typename T>
inline ncclDataType_t TypeMap();

template <>
inline ncclDataType_t TypeMap<char>() {
    return ncclChar;
}
template <>
inline ncclDataType_t TypeMap<signed char>() {
    return ncclChar;
}
template <>
inline ncclDataType_t TypeMap<unsigned char>() {
    return ncclUint8;
}
template <>
inline ncclDataType_t TypeMap<int>() {
    return ncclInt;
}
template <>
inline ncclDataType_t TypeMap<unsigned int>() {
    return ncclUint32;
}
template <>
inline ncclDataType_t TypeMap<long>() {
    return ncclInt64;
}
template <>
inline ncclDataType_t TypeMap<unsigned long>() {
    return ncclUint64;
}
template <>
inline ncclDataType_t TypeMap<long long>() {
    return ncclInt64;
}
template <>
inline ncclDataType_t TypeMap<unsigned long long>() {
    return ncclUint64;
}
template <>
inline ncclDataType_t TypeMap<__half>() {
    return ncclHalf;
}
// template <>
// inline ncclDataType_t TypeMap<UncGpubfloat16>() {
//     return ncclBfloat16;
// }
template <>
inline ncclDataType_t TypeMap<float>() {
    return ncclFloat;
}
template <>
inline ncclDataType_t TypeMap<double>() {
    return ncclDouble;
}
}  // namespace gpuccl
}  // namespace internal

}  // namespace uniconn


#endif  // __UNICONN_INCLUDE_UNICONN_GPUCCL_COMMON_HPP_
