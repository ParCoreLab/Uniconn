#ifndef __UNICONN_INCLUDE_UNICONN_INTERFACES_MEMORY_HPP_
#define __UNICONN_INCLUDE_UNICONN_INTERFACES_MEMORY_HPP_
#include "uniconn/utils.hpp"
namespace uniconn {

template <typename B = DefaultBackend>
struct Memory {
    template <typename T>
    GPU_HOST static T* Alloc(size_t count);
    template <typename T>
    GPU_HOST static void Free(T* ptr);
};

}  // namespace uniconn
#endif  // __UNICONN_INCLUDE_UNICONN_INTERFACES_MEMORY_HPP_
