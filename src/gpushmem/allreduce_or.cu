#include "allreduce_base.hpp"
namespace uniconn {

UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_MODE_HOST(DECL_FUNC_HOST, or, ReductionOperator::BOR)
UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_MODE_GROUP_DEVICE(DECL_FUNC_DEVICE, or, ReductionOperator::BOR)
}  // namespace uniconn