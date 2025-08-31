
#include "uniconn/gpuccl/common.hpp"
#include "uniconn/interfaces/memory.hpp"
#include "common.hpp"
namespace uniconn {

template <>
template <typename T>
GPU_HOST T* Memory<GpucclBackend>::Alloc(size_t count) {
    void* ret = NULL;
    GPU_RT_CALL(UncGpuMalloc(&ret, count * sizeof(T)));
    return static_cast<T*>(ret);
}

template <>
template <typename T>
GPU_HOST void Memory<GpucclBackend>::Free(T* ptr) {
    GPU_RT_CALL(UncGpuFree(ptr));
}

#define DECL_FUNC(TYPE)                                                       \
    template GPU_HOST TYPE* Memory<GpucclBackend>::Alloc<TYPE>(size_t count); \
    template GPU_HOST void Memory<GpucclBackend>::Free<TYPE>(TYPE * ptr);

UNC_GPUCCL_REPT(DECL_FUNC)
}  // namespace uniconn
