
#include "common.hpp"
#include "uniconn/gpuccl/environment.hpp"

namespace uniconn {

GPU_HOST Environment<GpucclBackend>::Environment() {
    if (!IsInit()) {
        MPI_CALL(MPI_Init(NULL, NULL));
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, WorldRank(), MPI_INFO_NULL,
                                     &node_local_world_comm));
        this->_finalize = true;
    } else {
        this->_finalize = false;
    }
}
GPU_HOST Environment<GpucclBackend>::Environment(int argc, char** argv) {
    if (!IsInit()) {
        MPI_CALL(MPI_Init(&argc, &argv));
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, WorldRank(), MPI_INFO_NULL,
                                     &node_local_world_comm));
        this->_finalize = true;
    } else {
        this->_finalize = false;
    }
}
GPU_HOST bool Environment<GpucclBackend>::IsInit() {
    int result;
    MPI_CALL(MPI_Initialized(&result));
    return result == true;
}

GPU_HOST void Environment<GpucclBackend>::SetDevice(int requested) {
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
        // If only 1 device visible, assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
        GPU_RT_CALL(UncGpuSetDevice(requested % num_devices));
    }
}

GPU_HOST int Environment<GpucclBackend>::WorldSize() {
    int ret = 0;
    if (IsInit()) {
        MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &ret));
    }

    return ret;
}
GPU_HOST int Environment<GpucclBackend>::WorldRank() {
    int ret = -1;
    if (IsInit()) {
        MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &ret));
    }
    return ret;
}
GPU_HOST int Environment<GpucclBackend>::NodeSize() {
    int ret = 0;
    if (IsInit()) {
        MPI_CALL(MPI_Comm_size(node_local_world_comm, &ret));
    }
    return ret;
}
GPU_HOST int Environment<GpucclBackend>::NodeRank() {
    int ret = -1;
    if (IsInit()) {
        MPI_CALL(MPI_Comm_rank(node_local_world_comm, &ret));
    }
    return ret;
}
GPU_HOST Environment<GpucclBackend>::~Environment() {
    if (_finalize) {
        MPI_CALL(MPI_Comm_free(&node_local_world_comm));
        MPI_CALL(MPI_Finalize());
    }
}

}  // namespace uniconn