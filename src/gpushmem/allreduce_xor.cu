#include "allreduce_base.hpp"
namespace uniconn {

    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_MODE_HOST(DECL_FUNC_HOST, xor, ReductionOperator::BXOR)
    UNC_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPE_OP_MODE_GROUP_DEVICE(DECL_FUNC_DEVICE, xor, ReductionOperator::BXOR)
}  // namespace uniconn