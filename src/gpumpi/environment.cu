
#include "common.hpp"
#include "uniconn/gpumpi/environment.hpp"

namespace uniconn {

GPU_HOST Environment<MPIBackend>::Environment() {
    if (!IsInit()) {
        
        MPI_CALL(MPI_Init(NULL, NULL));
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, WorldRank(), MPI_INFO_NULL,
                                     &node_local_world_comm));
        _finalize = true;
    } else {
        _finalize = false;
    }
}
GPU_HOST Environment<MPIBackend>::Environment(int argc, char** argv) {
    if (!IsInit()) {
       

        MPI_CALL(MPI_Init(&argc, &argv));
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, WorldRank(), MPI_INFO_NULL,
                                     &node_local_world_comm));
        _finalize = true;
    } else {
        _finalize = false;
    }
}
GPU_HOST bool Environment<MPIBackend>::IsInit() {
    int result;
    MPI_CALL(MPI_Initialized(&result));
    return result == true;
}

GPU_HOST void Environment<MPIBackend>::SetDevice(int requested) {
    if (IsInit()) {
        int local_size = NodeSize();
        int rank = WorldRank();
        int num_devices;
        GPU_RT_CALL(UncGpuGetDeviceCount(&num_devices));

        if (1 < num_devices && num_devices < NodeSize()) {
            fprintf(stderr, "ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n",
                    num_devices, NodeSize());
            MPI_CALL(MPI_Comm_free(&node_local_world_comm));
            MPI_CALL(MPI_Finalize());
            exit(1);
        }
        if (requested >= NodeSize()) {
            fprintf(stderr, "ERROR requested device id (%d) is larger than number of ranks on the node (%d)!\n",
                    requested, NodeSize());
            MPI_CALL(MPI_Comm_free(&node_local_world_comm));
            MPI_CALL(MPI_Finalize());
            exit(1);
        }
        GPU_RT_CALL(UncGpuSetDevice(requested % num_devices));
    }
}

GPU_HOST int Environment<MPIBackend>::WorldSize() {
    int ret = 0;
    if (IsInit()) {
        MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &ret));
    }

    return ret;
}
GPU_HOST int Environment<MPIBackend>::WorldRank() {
    int ret = -1;
    if (IsInit()) {
        MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &ret));
    }
    return ret;
}
GPU_HOST int Environment<MPIBackend>::NodeSize() {
    int ret = 0;
    if (IsInit()) {
        MPI_CALL(MPI_Comm_size(node_local_world_comm, &ret));
    }
    return ret;
}
GPU_HOST int Environment<MPIBackend>::NodeRank() {
    int ret = -1;
    if (IsInit()) {
        MPI_CALL(MPI_Comm_rank(node_local_world_comm, &ret));
    }
    return ret;
}
GPU_HOST Environment<MPIBackend>::~Environment() {
    if (_finalize) {
        MPI_CALL(MPI_Comm_free(&node_local_world_comm));
        MPI_CALL(MPI_Finalize());
    }
}

}  // namespace uniconn