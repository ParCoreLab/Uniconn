#ifndef __UNICONN_INCLUDE_UNICONN_GPUCCL_ENVIRONMENT_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUCCL_ENVIRONMENT_HPP_

#include "common.hpp"
#include "uniconn/interfaces/environment.hpp"
namespace uniconn {
template <>
class Environment<GpucclBackend> {
   private:
    static inline MPI_Comm node_local_world_comm;
    bool _finalize = false;

   public:
    GPU_HOST Environment();
    GPU_HOST Environment(int argc, char** argv);
    GPU_HOST bool IsInit();
    GPU_HOST void SetDevice(int requested);
    GPU_HOST int WorldSize();
    GPU_HOST int WorldRank();
    GPU_HOST int NodeSize();
    GPU_HOST int NodeRank();
    GPU_HOST ~Environment();
};
}  // namespace uniconn
#endif  // __UNICONN_INCLUDE_UNICONN_GPUCCL_ENVIRONMENT_HPP_