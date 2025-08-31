

#include "../include/common.hpp"

typedef float real;

template <int GRID_DIM_X>
__global__ void jacobi_kernel_p2p_full(real *__restrict__ const a_new, const real *__restrict__ const a,
                                       const int iy_start, const int iy_end, const int nx,
                                       real *__restrict__ const a_comm, real *__restrict__ const a_new_comm,
                                       uint64_t *__restrict__ const sync_arr, const int top_pe, const int bottom_pe,
                                       const uint64_t iter) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    if (blockIdx.x == gridDim.x - 1) {
        for (int ix = threadIdx.y * blockDim.x + threadIdx.x + 1; ix < (nx - 1); ix += blockDim.y * blockDim.x) {
            const real first_row_val = 0.25 * (a[iy_start * nx + ix + 1] + a[iy_start * nx + ix - 1] +
                                               a[(iy_start + 1) * nx + ix] + a_comm[ix]);
            a_new[iy_start * nx + ix] = first_row_val;
            a_comm[ix] = first_row_val;
        }
        nvshmemx_float_put_signal_nbi_block(a_new_comm, a_comm, nx, sync_arr + 1, 1, NVSHMEM_SIGNAL_ADD, top_pe);
        if (threadIdx.y * blockDim.x + threadIdx.x == 0) {
            nvshmem_signal_wait_until(sync_arr, NVSHMEM_CMP_GE, iter + 1);
        }
        cta.sync();
    } else if (blockIdx.x == gridDim.x - 2) {
        for (int ix = threadIdx.y * blockDim.x + threadIdx.x + 1; ix < (nx - 1); ix += blockDim.y * blockDim.x) {
            const real last_row_val = 0.25 * (a[(iy_end - 1) * nx + ix + 1] + a[(iy_end - 1) * nx + ix - 1] +
                                              a_comm[nx + ix] + a[(iy_end - 2) * nx + ix]);
            a_new[(iy_end - 1) * nx + ix] = last_row_val;
            a_comm[nx + ix] = last_row_val;
        }
        nvshmemx_float_put_signal_nbi_block(a_new_comm + nx, a_comm + nx, nx, sync_arr, 1, NVSHMEM_SIGNAL_ADD,
                                            bottom_pe);
        if (threadIdx.y * blockDim.x + threadIdx.x == 0) {
            nvshmem_signal_wait_until(sync_arr + 1, NVSHMEM_CMP_GE, iter + 1);
        }
        cta.sync();
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

    MPI_CALL(MPI_Init(&argc, &argv));
    int rank;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    int num_devices = 0;
    GPU_RT_CALL(UncGpuGetDeviceCount(&num_devices));

    int local_rank = -1;
    int local_size = 0;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_size(local_comm, &local_size));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    if (1 < num_devices && num_devices < local_size) {
        fprintf(stderr, "ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n",
                num_devices, local_size);
        MPI_CALL(MPI_Finalize());
        return EXIT_FAILURE;
    }

    GPU_RT_CALL(UncGpuSetDevice(local_rank % num_devices));

    nvshmemx_init_attr_t nvshmemInitAttr;
    MPI_Comm mpiInitComm = MPI_COMM_WORLD;
    nvshmemInitAttr.mpi_comm = &mpiInitComm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &nvshmemInitAttr);

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();

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

    a_comm = static_cast<real *>(nvshmem_malloc(2 * nx * sizeof(real)));
    a_new_comm = static_cast<real *>(nvshmem_malloc(2 * nx * sizeof(real)));
    sync_arr = static_cast<uint64_t *>(nvshmem_malloc(4 * sizeof(uint64_t)));

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

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    uint64_t iter = 0;
    void *kernelArgs[] = {(void *)&a_new,  (void *)&a,         (void *)&iy_start,   (void *)&iy_end,
                          (void *)&nx,     (void *)&a_comm,    (void *)&a_new_comm, (void *)&sync_arr,
                          (void *)&top_pe, (void *)&bottom_pe, (void *)&iter};

    // Set diriclet boundary conditions on left and right boundary
    initialize_boundaries<<<(ny / npes) / 128 + 1, 128, 0, compute_stream>>>(a, a_new, PI, iy_start_global, nx, iy_end,
                                                                             ny);
    GPU_RT_CALL(UncGpuMemsetAsync(sync_arr, 0, 4 * sizeof(uint64_t), compute_stream));
    // comm.Barrier(compute_stream);
    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

    PUSH_RANGE("Jacobi solve", 0)
    for (iter = 0; iter < iter_max + warmup; ++iter) {
        if (mype == 0 && iter == warmup) {
            GPU_RT_CALL(UncGpuEventRecord(startEvent, compute_stream));
        }
        // jacobi_step.LaunchKernel();
        nvshmemx_collective_launch((void *)jacobi_kernel_p2p_full<grid_dim_x>, numSms,
                                   dim3(dim_block_x, dim_block_y, 1), kernelArgs, 0, compute_stream);

        std::swap(a_new, a);
        std::swap(a_new_comm, a_comm);
    }
    // comm.Barrier(compute_stream);
    nvshmemx_barrier_all_on_stream(compute_stream);
    if (mype == 0) {
        GPU_RT_CALL(UncGpuEventRecord(stopEvent, compute_stream));
        GPU_RT_CALL(UncGpuEventSynchronize(stopEvent));
        POP_RANGE
        float multi_gpu_time = 0.0;
        GPU_RT_CALL(UncGpuEventElapsedTime(&multi_gpu_time, startEvent, stopEvent));
        printf("jacobi, nvshmem_d, %d, %f\n", npes, multi_gpu_time);
    }

    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    GPU_RT_CALL(UncGpuFree(a));
    GPU_RT_CALL(UncGpuFree(a_new));
    nvshmem_free(a_comm);
    nvshmem_free(a_new_comm);
    nvshmem_free(sync_arr);
    GPU_RT_CALL(UncGpuEventDestroy(startEvent));
    GPU_RT_CALL(UncGpuEventDestroy(stopEvent));
    GPU_RT_CALL(UncGpuStreamDestroy(compute_stream));

    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());
    return 0;
}
