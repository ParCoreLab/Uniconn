
#include "../include/common.hpp"

template <int GRID_DIM_X>
__global__ void jacobi_kernel_p2p_full(real *__restrict__ const a_new, const real *__restrict__ const a,
                                       const int iy_start, const int iy_end, const int nx,
                                       real *__restrict__ const a_comm, real *__restrict__ const a_new_comm,
                                       uint64_t *__restrict__ const sync_arr, const int top_pe, const int bottom_pe,
                                       const uint64_t iter, uniconn::Communicator<> *comm) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    if (blockIdx.x == gridDim.x - 1) {
        for (int ix = threadIdx.y * blockDim.x + threadIdx.x + 1; ix < (nx - 1); ix += blockDim.y * blockDim.x) {
            const real first_row_val = 0.25 * (a[iy_start * nx + ix + 1] + a[iy_start * nx + ix - 1] +
                                               a[(iy_start + 1) * nx + ix] + a_comm[ix]);
            a_new[iy_start * nx + ix] = first_row_val;
            a_comm[ix] = first_row_val;
        }
        uniconn::Coordinator<>::Post<uniconn::ThreadGroup::BLOCK>(a_comm, a_new_comm, nx, sync_arr + 1, iter + 1,
                                                                  top_pe, comm);
        uniconn::Coordinator<>::Acknowledge<uniconn::ThreadGroup::BLOCK>(a_new_comm + nx, nx, sync_arr, iter + 1,
                                                                         top_pe, comm);

    } else if (blockIdx.x == gridDim.x - 2) {
        for (int ix = threadIdx.y * blockDim.x + threadIdx.x + 1; ix < (nx - 1); ix += blockDim.y * blockDim.x) {
            const real last_row_val = 0.25 * (a[(iy_end - 1) * nx + ix + 1] + a[(iy_end - 1) * nx + ix - 1] +
                                              a_comm[nx + ix] + a[(iy_end - 2) * nx + ix]);
            a_new[(iy_end - 1) * nx + ix] = last_row_val;
            a_comm[nx + ix] = last_row_val;
        }
        uniconn::Coordinator<>::Post<uniconn::ThreadGroup::BLOCK>(a_comm + nx, a_new_comm + nx, nx, sync_arr, iter + 1,
                                                                  bottom_pe, comm);
        uniconn::Coordinator<>::Acknowledge<uniconn::ThreadGroup::BLOCK>(a_new_comm, nx, sync_arr + 1, iter + 1,
                                                                         bottom_pe, comm);
    }
    for (int iy = ((blockIdx.x / GRID_DIM_X) * blockDim.y + threadIdx.y + iy_start + 1); iy < (iy_end - 1);
         iy += (gridDim.x / GRID_DIM_X) * blockDim.y) {
        for (int ix = ((blockIdx.x % GRID_DIM_X) * blockDim.x + threadIdx.x + 1); ix < (nx - 1);
             ix += GRID_DIM_X * blockDim.x) {
            a_new[iy * nx + ix] =
                0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] + a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        }
    }
}

__global__ void jacobi_block_p2p_limited(real *__restrict__ const a_new, const real *__restrict__ const a,
                                         const int iy_start, const int iy_end, const int nx,
                                         real *__restrict__ const a_comm, real *__restrict__ const a_new_comm,
                                         const int top_pe, const int bottom_pe, uniconn::Communicator<> *comm) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (iy > iy_start && iy < iy_end - 1 && ix < (nx - 1)) {
        const real new_val =
            0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] + a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
    } else if (iy == iy_start && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] + a[(iy + 1) * nx + ix] + a_comm[ix]);
        a_new[iy * nx + ix] = new_val;
    } else if (iy == iy_end - 1 && ix < (nx - 1)) {
        const real new_val =
            0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] + a_comm[nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
    }

    /* starting (x, y) coordinate of the block */
    int block_iy = iy - threadIdx.y; /* Alternatively, block_iy = blockIdx.y * blockDim.y + iy_start */
    int block_ix = ix - threadIdx.x; /* Alternatively, block_ix = blockIdx.x * blockDim.x + 1 */
    /* Communicate the boundaries */
    if ((nx - 1) > block_ix) {
        if ((block_iy <= iy_start) && (iy_start < block_iy + blockDim.y)) {
            uniconn::Coordinator<>::Post<uniconn::ThreadGroup::BLOCK>(
                a_new + iy_start * nx + block_ix, a_new_comm + block_ix, min(blockDim.x, nx - 1 - block_ix), nullptr, 0,
                top_pe, comm);
        }
        if (((iy_end - 1) < block_iy + blockDim.y) && (block_iy <= (iy_end - 1))) {
            uniconn::Coordinator<>::Post<uniconn::ThreadGroup::BLOCK>(
                a_new + (iy_end - 1) * nx + block_ix, a_new_comm + nx + block_ix, min(blockDim.x, nx - 1 - block_ix),
                nullptr, 0, bottom_pe, comm);
        }
    }
}

int main(int argc, char *argv[]) {
    const uint64_t iter_max = get_argval<uint64_t>(argv, argv + argc, "-niter", DEFAULT_ITER_NUM);
    const uint64_t warmup = get_argval<uint64_t>(argv, argv + argc, "-nwarmup", DEFAULT_SKIP_NUM);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", DEFAULT_NUM_ROW);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", DEFAULT_NUM_COL);

    real *a;
    real *a_new;
    real *a_comm;
    real *a_new_comm;
    uint64_t *sync_arr = nullptr;

    long long unsigned int mesh_size_per_rank = nx * 2;
    long long unsigned int required_symmetric_heap_size = 2 * mesh_size_per_rank * sizeof(real) * 1.1;

    char symmetric_heap_size_str[100];
    sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
    setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    uniconn::Environment<> env(argc, argv);

    int rank = env.WorldRank();
    int size = env.WorldSize();
    int local_pe = env.NodeRank();
    // Add Local size and Rank
    env.SetDevice(local_pe);
    GPU_RT_CALL(UncGpuDeviceSynchronize());

    uniconn::Communicator<> comm;
    uniconn::Communicator<> *comm_d = comm.toDevice();
    int npes = comm.GlobalSize();
    int mype = comm.GlobalRank();
    UncGpuStream_t compute_stream;
    GPU_RT_CALL(UncGpuStreamCreate(&compute_stream));

    UncGpuEvent_t startEvent, stopEvent;
    GPU_RT_CALL(UncGpuEventCreate(&startEvent));
    GPU_RT_CALL(UncGpuEventCreate(&stopEvent));

    int chunk_size;
    int chunk_size_low = (ny - 2) / npes;
    int chunk_size_high = chunk_size_low + 1;

    int num_ranks_low = npes * chunk_size_low + npes - (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
    if (mype < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    a_comm = uniconn::Memory<>::Alloc<real>(2 * nx);
    a_new_comm = uniconn::Memory<>::Alloc<real>(2 * nx);
    sync_arr = uniconn::Memory<>::Alloc<uint64_t>(4);

    GPU_RT_CALL(UncGpuMalloc(&a, nx * chunk_size_high * sizeof(real)));
    GPU_RT_CALL(UncGpuMalloc(&a_new, nx * chunk_size_high * sizeof(real)));

    GPU_RT_CALL(UncGpuMemset(a_comm, 0, nx * 2 * sizeof(real)));
    GPU_RT_CALL(UncGpuMemset(a_new_comm, 0, nx * 2 * sizeof(real)));
    GPU_RT_CALL(UncGpuMemset(sync_arr, 0, 4 * sizeof(uint64_t)));
    GPU_RT_CALL(UncGpuMemset(a, 0, nx * chunk_size * sizeof(real)));
    GPU_RT_CALL(UncGpuMemset(a_new, 0, nx * chunk_size * sizeof(real)));

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (mype < num_ranks_low) {
        iy_start_global = mype * chunk_size_low;
    } else {
        iy_start_global = num_ranks_low * chunk_size_low + (mype - num_ranks_low) * chunk_size_high;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array
    // do not process boundaries
    iy_end_global = std::min(iy_end_global, ny - 1);

    int iy_start = 0;
    int iy_end = (iy_end_global - iy_start_global + 1) + iy_start;

    int device;
    GPU_RT_CALL(UncGpuGetDevice(&device));
    UncGpuDeviceProp_t deviceProp{};
    GPU_RT_CALL(UncGpuGetDeviceProperties(&deviceProp, device));
    int numSms = deviceProp.multiProcessorCount;

    constexpr int grid_dim_x = 11;
    // const int grid_dim_y = (numSms - 2) / grid_dim_x;

    // calculate boundary indices for top and bottom boundaries
    int top_pe = mype > 0 ? mype - 1 : (npes - 1);
    int bottom_pe = (mype + 1) % npes;

    // int iy_end_top = (top_pe < num_ranks_low) ? chunk_size_low + 1 : chunk_size_high + 1;
    // int iy_start_bottom = 0;

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid(nx / dim_block_x + 1, chunk_size / dim_block_y + 1, 1);

    constexpr int dim_block_block_comm_x = 1024;
    constexpr int dim_block_block_comm_y = 1;
    dim3 dim_grid_block_comm((nx + dim_block_block_comm_x - 1) / dim_block_block_comm_x,
                             (chunk_size + dim_block_block_comm_y - 1) / dim_block_block_comm_y, 1);

    uniconn::Coordinator<> jacobi_step(compute_stream);

    uint64_t iter = 0;

    void *kernelArgsHost[] = {(void *)&a_new,  (void *)&a,  (void *)&iy_start,
                              (void *)&iy_end, (void *)&nx, (void *)&a_comm};
    void *kernelArgsLimitedDevice[] = {(void *)&a_new,     (void *)&a,      (void *)&iy_start,   (void *)&iy_end,
                                       (void *)&nx,        (void *)&a_comm, (void *)&a_new_comm, (void *)&top_pe,
                                       (void *)&bottom_pe, (void *)&comm_d};
    void *kernelArgsFullDevice[] = {(void *)&a_new,  (void *)&a,         (void *)&iy_start,   (void *)&iy_end,
                                    (void *)&nx,     (void *)&a_comm,    (void *)&a_new_comm, (void *)&sync_arr,
                                    (void *)&top_pe, (void *)&bottom_pe, (void *)&iter,       (void *)&comm_d};

    jacobi_step.bindKernel<uniconn::LaunchMode::HostDriven>((void *)jacobi_kernel, dim_grid,
                                                            dim3(dim_block_x, dim_block_y, 1), 0, kernelArgsHost);

    jacobi_step.bindKernel<uniconn::LaunchMode::LimitedDevice>((void *)jacobi_block_p2p_limited, dim_grid_block_comm,
                                                               dim3(dim_block_block_comm_x, dim_block_block_comm_y, 1),
                                                               0, kernelArgsLimitedDevice);

    jacobi_step.bindKernel<uniconn::LaunchMode::FullDevice>((void *)jacobi_kernel_p2p_full<grid_dim_x>, numSms,
                                                            dim3(dim_block_x, dim_block_y, 1), 0, kernelArgsFullDevice);

    // Set diriclet boundary conditions on left and right boundary
    initialize_boundaries<<<(ny / npes) / 128 + 1, 128, 0, compute_stream>>>(a, a_new, PI, iy_start_global, nx, iy_end,
                                                                             ny);
    GPU_RT_CALL(UncGpuMemsetAsync(sync_arr, 0, 4 * sizeof(uint64_t), compute_stream));
    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

    PUSH_RANGE("Jacobi solve", 0)
    for (iter = 0; iter < iter_max + warmup; ++iter) {
        if (mype == 0 && iter == warmup) {
            GPU_RT_CALL(UncGpuEventRecord(startEvent, compute_stream));
        }
        jacobi_step.LaunchKernel();
        jacobi_step.CommStart();
        jacobi_step.Post(a_comm, a_new_comm, nx, sync_arr + 1, iter + 1, top_pe, &comm);
        jacobi_step.Post(a_comm + nx, a_new_comm + nx, nx, sync_arr, iter + 1, bottom_pe, &comm);
        jacobi_step.Acknowledge(a_new_comm + nx, nx, sync_arr, iter + 1, top_pe, &comm);
        jacobi_step.Acknowledge(a_new_comm, nx, sync_arr + 1, iter + 1, bottom_pe, &comm);
        jacobi_step.CommEnd();
        std::swap(a_new, a);
        std::swap(a_new_comm, a_comm);
    }
    comm.Barrier(compute_stream);
    if (mype == 0) {
        GPU_RT_CALL(UncGpuEventRecord(stopEvent, compute_stream));
        GPU_RT_CALL(UncGpuEventSynchronize(stopEvent));
        POP_RANGE
        float multi_gpu_time = 0.0;
        GPU_RT_CALL(UncGpuEventElapsedTime(&multi_gpu_time, startEvent, stopEvent));
        printf("jacobi, unc, %d, %f\n", npes, multi_gpu_time);
    }

    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    GPU_RT_CALL(UncGpuFree(a));
    GPU_RT_CALL(UncGpuFree(a_new));
    uniconn::Memory<>::Free(a_comm);
    uniconn::Memory<>::Free(a_new_comm);
    uniconn::Memory<>::Free(sync_arr);
    GPU_RT_CALL(UncGpuEventDestroy(startEvent));
    GPU_RT_CALL(UncGpuEventDestroy(stopEvent));
    GPU_RT_CALL(UncGpuStreamDestroy(compute_stream));
    return 0;
}