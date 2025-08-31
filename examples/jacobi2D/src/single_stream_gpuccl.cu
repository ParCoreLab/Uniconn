

#include "../include/common.hpp"

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

    MPI_CALL(MPI_Init(&argc, &argv));
    int rank;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    int num_devices = 0;
    GPU_RT_CALL(UncGpuGetDeviceCount(&num_devices));

    ncclUniqueId nccl_uid;
    if (rank == 0) NCCL_CALL(ncclGetUniqueId(&nccl_uid));
    MPI_CALL(MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

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

    ncclComm_t nccl_comm;
    NCCL_CALL(ncclCommInitRank(&nccl_comm, size, nccl_uid, rank));
    int nccl_version = 0;
    NCCL_CALL(ncclGetVersion(&nccl_version));
    if (nccl_version < 2800) {
        fprintf(stderr, "ERROR NCCL 2.8 or newer is required.\n");
        NCCL_CALL(ncclCommDestroy(nccl_comm));
        MPI_CALL(MPI_Finalize());
        return 1;
    }

    GPU_RT_CALL(UncGpuDeviceSynchronize());

    UncGpuStream_t compute_stream;
    GPU_RT_CALL(UncGpuStreamCreate(&compute_stream));

    UncGpuEvent_t startEvent, stopEvent;
    GPU_RT_CALL(UncGpuEventCreate(&startEvent));
    GPU_RT_CALL(UncGpuEventCreate(&stopEvent));

    int chunk_size;
    int chunk_size_low = (ny - 2) / size;
    int chunk_size_high = chunk_size_low + 1;

    int num_ranks_low = size * chunk_size_low + size - (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
    if (rank < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    
    GPU_RT_CALL(UncGpuMalloc(&a_comm, 2 * nx * sizeof(real)));
    GPU_RT_CALL(UncGpuMalloc(&a_new_comm, 2 * nx * sizeof(real)));
    GPU_RT_CALL(UncGpuMalloc(&sync_arr, 4 * sizeof(uint64_t)));

    unsigned char *barrier_buf;
    GPU_RT_CALL(UncGpuMalloc(&barrier_buf, 1 * sizeof(unsigned char)));

    GPU_RT_CALL(UncGpuMalloc(&a, nx * chunk_size_high * sizeof(real)));
    GPU_RT_CALL(UncGpuMalloc(&a_new, nx * chunk_size_high * sizeof(real)));

    GPU_RT_CALL(UncGpuMemset(a, 0, nx * chunk_size * sizeof(real)));
    GPU_RT_CALL(UncGpuMemset(a_new, 0, nx * chunk_size * sizeof(real)));

    GPU_RT_CALL(UncGpuMemset(a_comm, 0, nx * 2 * sizeof(real)));
    GPU_RT_CALL(UncGpuMemset(a_new_comm, 0, nx * 2 * sizeof(real)));
    GPU_RT_CALL(UncGpuMemset(sync_arr, 0, 4 * sizeof(uint64_t)));
    GPU_RT_CALL(UncGpuMemset(barrier_buf, 0, 1 * sizeof(unsigned char)));

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (rank < num_ranks_low) {
        iy_start_global = rank * chunk_size_low;
    } else {
        iy_start_global = num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array
    // do not process boundaries
    iy_end_global = std::min(iy_end_global, ny - 1);

    int iy_start = 0;
    int iy_end = (iy_end_global - iy_start_global + 1) + iy_start;

    // calculate boundary indices for top and bottom boundaries
    int top_pe = rank > 0 ? rank - 1 : (size - 1);
    int bottom_pe = (rank + 1) % size;

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid(nx / dim_block_x + 1, chunk_size / dim_block_y + 1, 1);

    // Set diriclet boundary conditions on left and right boundary
    initialize_boundaries<<<(ny / size) / 128 + 1, 128, 0, compute_stream>>>(a, a_new, PI, iy_start_global, nx,
                                                                             chunk_size, ny);
    // comm.Barrier(compute_stream);
    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

    PUSH_RANGE("Jacobi solve", 0)
    for (uint64_t iter = 0; iter < iter_max + warmup; ++iter) {
        if (rank == 0 && iter == warmup) {
            GPU_RT_CALL(UncGpuEventRecord(startEvent, compute_stream));
        }
        
        jacobi_kernel<<<dim_grid, dim3(dim_block_x, dim_block_y, 1), 0, compute_stream>>>(a_new, a, iy_start, iy_end,
                                                                                          nx, a_comm);
        
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(a_new + iy_start * nx, nx, NCCL_REAL_TYPE, top_pe, nccl_comm, compute_stream));
        NCCL_CALL(ncclSend(a_new + (iy_end - 1) * nx, nx, NCCL_REAL_TYPE, bottom_pe, nccl_comm, compute_stream));
        NCCL_CALL(ncclRecv(a_new_comm, nx, NCCL_REAL_TYPE, bottom_pe, nccl_comm, compute_stream));
        NCCL_CALL(ncclRecv(a_new_comm + nx, nx, NCCL_REAL_TYPE, top_pe, nccl_comm, compute_stream));
        NCCL_CALL(ncclGroupEnd());
        std::swap(a_new, a);
        std::swap(a_new_comm, a_comm);
    }
    // comm.Barrier(compute_stream);
    NCCL_CALL(ncclAllReduce((const void *)barrier_buf, (void *)barrier_buf, 1, ncclUint8, ncclSum, nccl_comm,
                            compute_stream));
    if (rank == 0) {
        GPU_RT_CALL(UncGpuEventRecord(stopEvent, compute_stream));
        GPU_RT_CALL(UncGpuEventSynchronize(stopEvent));
        POP_RANGE

        float multi_gpu_time = 0.0;
        GPU_RT_CALL(UncGpuEventElapsedTime(&multi_gpu_time, startEvent, stopEvent));
        printf("jacobi, gpuccl, %d, %f\n", size, multi_gpu_time);
    }

    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    GPU_RT_CALL(UncGpuFree(a));
    GPU_RT_CALL(UncGpuFree(a_new));
    GPU_RT_CALL(UncGpuFree(a_comm));
    GPU_RT_CALL(UncGpuFree(a_new_comm));
    GPU_RT_CALL(UncGpuFree(sync_arr));
    GPU_RT_CALL(UncGpuFree(barrier_buf));

    GPU_RT_CALL(UncGpuEventDestroy(startEvent));
    GPU_RT_CALL(UncGpuEventDestroy(stopEvent));
    GPU_RT_CALL(UncGpuStreamDestroy(compute_stream));

    NCCL_CALL(ncclCommDestroy(nccl_comm));
    MPI_CALL(MPI_Finalize());
    return 0;
}
