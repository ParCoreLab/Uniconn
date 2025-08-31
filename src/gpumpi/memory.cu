
#include "uniconn/gpumpi/common.hpp"
#include "uniconn/interfaces/memory.hpp"
#include "common.hpp"
namespace uniconn {

template <>
template <typename T>
GPU_HOST T* Memory<MPIBackend>::Alloc(size_t count) {
    void* ret = NULL;
    GPU_RT_CALL(UncGpuMalloc(&ret, count * sizeof(T)));
    return static_cast<T*>(ret);
}

template <>
template <typename T>
GPU_HOST void Memory<MPIBackend>::Free(T* ptr) {
    GPU_RT_CALL(UncGpuFree(ptr));
}

#define DECL_FUNC(TYPE)                                                    \
    template GPU_HOST TYPE* Memory<MPIBackend>::Alloc<TYPE>(size_t count); \
    template GPU_HOST void Memory<MPIBackend>::Free<TYPE>(TYPE * ptr);

UNC_GPUMPI_REPT(DECL_FUNC)
}  // namespace uniconn
