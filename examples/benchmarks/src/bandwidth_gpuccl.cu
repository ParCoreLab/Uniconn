
#include "../include/common.hpp"

typedef char real;
#define NCCL_REAL_TYPE ncclChar

int main(int argc, char *argv[]) {
    // Getting arguments from user.
    const int window_size = get_argval<int>(argv, argv + argc, "-nwindow", WINDOW_SIZE);
    const int max_msg_size = get_argval<int>(argv, argv + argc, "-maxmsg", MAX_MESSAGE_SIZE);  // coll 1<<20

    // MPI initialization.
    MPI_CALL(MPI_Init(&argc, &argv));

    int rank;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    int num_devices = 0;
    GPU_RT_CALL(UncGpuGetDeviceCount(&num_devices));
    // Divide the processors to their nodes.
    int local_rank = -1;
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    if (1 > num_devices || num_devices < local_size) {
        fprintf(stderr, "ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n",
                num_devices, local_size);
        MPI_CALL(MPI_Finalize());
        return 1;
    }

    GPU_RT_CALL(UncGpuSetDevice(local_rank % num_devices));

    ncclUniqueId nccl_uid;
    if (rank == 0) NCCL_CALL(ncclGetUniqueId(&nccl_uid));
    MPI_CALL(MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

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

    UncGpuStream_t nccl_stream;
    GPU_RT_CALL(UncGpuStreamCreate(&nccl_stream));

    UncGpuEvent_t startEvent, stopEvent;
    GPU_RT_CALL(UncGpuEventCreate(&startEvent));
    GPU_RT_CALL(UncGpuEventCreate(&stopEvent));

    // Memory allocation.
    real *send_buff;
    GPU_RT_CALL(UncGpuMalloc((void **)&send_buff, max_msg_size * sizeof(real)));
    real *recv_buff;
    GPU_RT_CALL(UncGpuMalloc((void **)&recv_buff, max_msg_size * sizeof(real)));

    const int neighbour = (rank + 1) % size;

    int iter_max = BW_LOOP_SMALL;
    uint64_t warmup = BW_SKIP_SMALL;
    for (int message_size = 1; message_size <= max_msg_size; message_size *= 2) {
        if (message_size > LARGE_MESSAGE_SIZE) {
            iter_max = BW_LOOP_LARGE;
            warmup = BW_SKIP_LARGE;
        }
        GPU_RT_CALL(UncGpuDeviceSynchronize());
        if (rank == 0) {
            double start = 0.0, end = 0.0;
            for (size_t i = 0; i < warmup + iter_max; i++) {
                if (i == warmup) {
                    start = MPI_Wtime();
                    // GPU_RT_CALL(UncGpuEventRecord(startEvent, nccl_stream));
                }
                init_kernel<<<1, 1024, 0, nccl_stream>>>(recv_buff, send_buff, message_size);
                NCCL_CALL(ncclGroupStart());
                for (int j = 0; j < window_size; ++j) {
                    NCCL_CALL(ncclSend(send_buff, message_size, NCCL_REAL_TYPE, neighbour, nccl_comm, nccl_stream));
                }
                NCCL_CALL(ncclGroupEnd());
                NCCL_CALL(ncclRecv(recv_buff, 1, NCCL_REAL_TYPE, neighbour, nccl_comm, nccl_stream));
            }
            GPU_RT_CALL(UncGpuStreamSynchronize(nccl_stream));
            end = MPI_Wtime();
            // GPU_RT_CALL(UncGpuEventRecord(stopEvent, nccl_stream));
            // GPU_RT_CALL(UncGpuEventSynchronize(stopEvent));
            // float multi_gpu_time = 0.0;
            // GPU_RT_CALL(UncGpuEventElapsedTime(&multi_gpu_time, startEvent, stopEvent));

            printf("bandwidth, gpuccl, %d, %f\n", message_size,
                   ((message_size / 1e6 * window_size * iter_max) / (end - start)));

        } else if (rank == 1)  // process 1
        {
            for (size_t i = 0; i < iter_max + warmup; i++) {
                NCCL_CALL(ncclGroupStart());
                for (int j = 0; j < window_size; ++j) {
                    NCCL_CALL(ncclRecv(recv_buff, message_size, NCCL_REAL_TYPE, neighbour, nccl_comm, nccl_stream));
                }
                NCCL_CALL(ncclGroupEnd());
                NCCL_CALL(ncclSend(send_buff, 1, NCCL_REAL_TYPE, neighbour, nccl_comm, nccl_stream));
            }
        }
    }

    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    GPU_RT_CALL(UncGpuFree(recv_buff));
    GPU_RT_CALL(UncGpuFree(send_buff));
    GPU_RT_CALL(UncGpuEventDestroy(startEvent));
    GPU_RT_CALL(UncGpuEventDestroy(stopEvent));
    GPU_RT_CALL(UncGpuStreamDestroy(nccl_stream));
    NCCL_CALL(ncclCommDestroy(nccl_comm));
    MPI_CALL(MPI_Finalize());
    return 0;
}
