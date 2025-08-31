#include "common.hpp"
#include "uniconn/gpushmem/common.hpp"
#include "uniconn/interfaces/memory.hpp"
namespace uniconn {

template <>
template <typename TYPE>
GPU_HOST TYPE* Memory<GpushmemBackend>::Alloc(size_t count) {
    return static_cast<TYPE*>(nvshmem_malloc(count * sizeof(TYPE)));
}

template <>
template <typename TYPE>
GPU_HOST void Memory<GpushmemBackend>::Free(TYPE* ptr) {
    nvshmem_free(ptr);
}

#define UNC_SHMEM_REPT_TYPE(FN_TEMPLATE) \
    FN_TEMPLATE(float)                   \
    FN_TEMPLATE(double)                  \
    FN_TEMPLATE(char)                    \
    FN_TEMPLATE(short)                   \
    FN_TEMPLATE(signed char)             \
    FN_TEMPLATE(int)                     \
    FN_TEMPLATE(long)                    \
    FN_TEMPLATE(long long)               \
    FN_TEMPLATE(unsigned char)           \
    FN_TEMPLATE(unsigned short)          \
    FN_TEMPLATE(unsigned int)            \
    FN_TEMPLATE(unsigned long)           \
    FN_TEMPLATE(unsigned long long)

#define DECL_FUNC(TYPE)                                                         \
    template GPU_HOST TYPE* Memory<GpushmemBackend>::Alloc<TYPE>(size_t count); \
    template GPU_HOST void Memory<GpushmemBackend>::Free<TYPE>(TYPE * ptr);

UNC_SHMEM_REPT_TYPE(DECL_FUNC)
#undef DECL_FUNC
#undef UNC_SHMEM_REPT_TYPE
}  // namespace uniconn