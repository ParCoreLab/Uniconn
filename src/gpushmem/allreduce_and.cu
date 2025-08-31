#include "allreduce_base.hpp"
namespace uniconn {

UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_MODE_HOST(DECL_FUNC_HOST, and, ReductionOperator::BAND)
UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_MODE_GROUP_DEVICE(DECL_FUNC_DEVICE, and, ReductionOperator::BAND)
}  // namespace uniconn
