#ifndef __UNICONN_INCLUDE_UNICONN_GPUCCL_MEMORY_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUCCL_MEMORY_HPP_

#include "common.hpp"
#include "uniconn/interfaces/memory.hpp"
namespace uniconn {

template <>
struct Memory<GpucclBackend> {
    template <typename T>
    GPU_HOST static T* Alloc(size_t count);
    template <typename T>
    GPU_HOST static void Free(T* ptr);
};
}  // namespace uniconn
#endif  // __UNICONN_INCLUDE_UNICONN_GPUCCL_MEMORY_HPP_
