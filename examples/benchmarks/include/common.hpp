#ifndef __UNICONN_EXAMPLES_BENCHMARKS_INCLUDE_COMMON_HPP_
#define __UNICONN_EXAMPLES_BENCHMARKS_INCLUDE_COMMON_HPP_

#include "../../common/utils.hpp"

#define BW_LOOP_SMALL 1000
#define BW_SKIP_SMALL 100
#define BW_LOOP_LARGE 200
#define BW_SKIP_LARGE 20
#define LAT_LOOP_SMALL 100000
#define LAT_SKIP_SMALL 1000
#define LAT_LOOP_LARGE 10000
#define LAT_SKIP_LARGE 100
#define COLL_LOOP_SMALL 10000
#define COLL_SKIP_SMALL 1000
#define COLL_LOOP_LARGE 1000
#define COLL_SKIP_LARGE 100
#define MAX_MESSAGE_SIZE (1 << 24)
#define MAX_MSG_SIZE_PT2PT (1 << 20)
#define MAX_MSG_SIZE_COLL (1 << 20)
#define LARGE_MESSAGE_SIZE (1 << 17)  // 128KiB
#define WINDOW_SIZE 64

__global__ void init_kernel(char *__restrict__ const a_new, char *__restrict__ const a, const int nx) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < nx) {
        a_new[ix] = a[ix] = 1;
    }
}

#endif  // __UNICONN_EXAMPLES_BENCHMARKS_INCLUDE_COMMON_HPP_