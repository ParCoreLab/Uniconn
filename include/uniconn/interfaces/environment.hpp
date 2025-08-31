#ifndef __UNICONN_INCLUDE_UNICONN_INTERFACES_ENVIRONMENT_HPP_
#define __UNICONN_INCLUDE_UNICONN_INTERFACES_ENVIRONMENT_HPP_
#include "uniconn/utils.hpp"
namespace uniconn {
template <typename Backend=DefaultBackend>
class Environment {
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

#endif  // __UNICONN_INCLUDE_UNICONN_INTERFACES_ENVIRONMENT_HPP_