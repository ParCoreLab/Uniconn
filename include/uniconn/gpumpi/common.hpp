#ifndef __UNICONN_INCLUDE_UNICONN_GPUMPI_COMMON_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUMPI_COMMON_HPP_

#include <mpi.h>
#include "uniconn/utils.hpp"


#define MPI_CALL(call)                                                                                             \
    {                                                                                                              \
        int mpi_status = call;                                                                                     \
        if (MPI_SUCCESS != mpi_status) {                                                                           \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                                                           \
            int mpi_error_string_length = 0;                                                                       \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length);                              \
            if (NULL != mpi_error_string)                                                                          \
                fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %s (%d).\n", #call,      \
                        __LINE__, __FILE__, mpi_error_string, mpi_status);                                         \
            else                                                                                                   \
                fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %d.\n", #call, __LINE__, \
                        __FILE__, mpi_status);                                                                     \
            exit(mpi_status);                                                                                      \
        }                                                                                                          \
    }

namespace uniconn {
struct MPIBackend {};
namespace internal {
namespace mpi {
#ifdef UNC_HAS_LARGE_COUNT_MPI
using Unc_mpi_count_t = MPI_Count;
using Unc_mpi_displ_t = MPI_Aint;
#else
using Unc_mpi_count_t = int;
using Unc_mpi_displ_t = int;
#endif

/** Used to map types to the associated MPI datatype. */
template <typename T>
inline constexpr MPI_Datatype TypeMap();
template <>
inline constexpr MPI_Datatype TypeMap<char>() {
    return MPI_CHAR;
}
template <>
inline constexpr MPI_Datatype TypeMap<signed char>() {
    return MPI_SIGNED_CHAR;
}
template <>
inline constexpr MPI_Datatype TypeMap<unsigned char>() {
    return MPI_UNSIGNED_CHAR;
}
template <>
inline constexpr MPI_Datatype TypeMap<short>() {
    return MPI_SHORT;
}
template <>
inline constexpr MPI_Datatype TypeMap<unsigned short>() {
    return MPI_UNSIGNED_SHORT;
}
template <>
inline constexpr MPI_Datatype TypeMap<int>() {
    return MPI_INT;
}
template <>
inline constexpr MPI_Datatype TypeMap<unsigned int>() {
    return MPI_UNSIGNED;
}
template <>
inline constexpr MPI_Datatype TypeMap<long>() {
    return MPI_LONG;
}
template <>
inline constexpr MPI_Datatype TypeMap<unsigned long>() {
    return MPI_UNSIGNED_LONG;
}
template <>
inline constexpr MPI_Datatype TypeMap<long long>() {
    return MPI_LONG_LONG_INT;
}
template <>
inline constexpr MPI_Datatype TypeMap<unsigned long long>() {
    return MPI_UNSIGNED_LONG_LONG;
}
template <>
inline constexpr MPI_Datatype TypeMap<float>() {
    return MPI_FLOAT;
}
template <>
inline constexpr MPI_Datatype TypeMap<double>() {
    return MPI_DOUBLE;
}
template <>
inline constexpr MPI_Datatype TypeMap<long double>() {
    return MPI_LONG_DOUBLE;
}

template <ReductionOperator OP>
inline constexpr MPI_Op ReductOp2MPI_Op();

template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::SUM>() {
    return MPI_SUM;
}
template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::PROD>() {
    return MPI_PROD;
}
template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::MIN>() {
    return MPI_MIN;
}
template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::MAX>() {
    return MPI_MAX;
}
template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::LOR>() {
    return MPI_LOR;
}
template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::LAND>() {
    return MPI_LAND;
}
template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::LXOR>() {
    return MPI_LXOR;
}
template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::BOR>() {
    return MPI_BOR;
}
template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::BAND>() {
    return MPI_BAND;
}
template <>
inline constexpr MPI_Op ReductOp2MPI_Op<ReductionOperator::BXOR>() {
    return MPI_BXOR;
}

/** Return either sendbuf or MPI_IN_PLACE. */
template <typename T>
void* buf_or_inplace(T* buf) {
    return buf == IN_PLACE<T> ? MPI_IN_PLACE : buf;
}
/** Return either sendbuf or MPI_IN_PLACE. */
template <typename T>
const void* buf_or_inplace(const T* buf) {
    return buf == IN_PLACE<T> ? MPI_IN_PLACE : buf;
}

/** True if count elements can be sent by MPI. */
inline bool check_count_fits_mpi(size_t count) {
    return count <= static_cast<size_t>(std::numeric_limits<Unc_mpi_displ_t>::max());
}

/** Throw an exception if count elements cannot be sent by MPI. */
inline void assert_count_fits_mpi(size_t count) { assert(check_count_fits_mpi(count)); }

/** True if displ is a valid MPI displacement. */
inline bool check_displ_fits_mpi(size_t displ) {
    return displ <= static_cast<size_t>(std::numeric_limits<Unc_mpi_displ_t>::max());
}

/** Throw an exception if displ is not a valid MPI displacement. */
inline void assert_displ_fits_mpi(size_t displ) { assert(check_displ_fits_mpi(displ)); }

#ifdef UNC_HAS_LARGE_COUNT_MPI
#define UNC_MPI_LARGE_COUNT_CALL(mpi_func) mpi_func##_c
#else
#define UNC_MPI_LARGE_COUNT_CALL(mpi_func) mpi_func
#endif

}  // namespace mpi
}  // namespace internal
}  // namespace uniconn
#endif  // __UNICONN_INCLUDE_UNICONN_GPUMPI_COMMON_HPP_
