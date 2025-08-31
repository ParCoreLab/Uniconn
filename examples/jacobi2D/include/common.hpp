#ifndef __UNICONN_EXAMPLES_JACOBI2D_INCLUDE_COMMON_HPP_
#define __UNICONN_EXAMPLES_JACOBI2D_INCLUDE_COMMON_HPP_

#include "../../common/utils.hpp"

#define DEFAULT_ITER_NUM 100000
#define DEFAULT_SKIP_NUM 10000
#define DEFAULT_NUM_ROW 1<<14
#define DEFAULT_NUM_COL 1<<14

typedef float real;
const real PI = 2.0 * std::asin(1.0);
constexpr real tol = 1.0e-7;
#define MPI_REAL_TYPE MPI_FLOAT
#define NCCL_REAL_TYPE ncclFloat

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a, const real pi,
                                      const int offset, const int nx, const int my_ny, int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

__global__ void jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a, const int iy_start,
                              const int iy_end, const int nx, real *__restrict__ const a_comm) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (iy > iy_start && iy < iy_end - 1 && ix < (nx - 1)) {
        const real new_val =
            0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] + a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
    } else if (iy == iy_start && ix < (nx - 1)) {
        const real new_val =
            0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] + a[(iy + 1) * nx + ix] + a_comm[ ix]);
        a_new[iy * nx + ix] = new_val;
        a_comm[ ix]= new_val;
    } else if (iy == iy_end - 1 && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] + a_comm[nx +ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
        a_comm[ix]= new_val;
        a_comm[nx + ix]= new_val;
    }
}

#endif  // __UNICONN_EXAMPLES_JACOBI2D_INCLUDE_COMMON_HPP_