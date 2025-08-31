
#include "../include/common.hpp"
#define ROOT 0

typedef struct {
    uniconn::Communicator<>* comm;
    int comm_rank;
    int comm_size;
    int local_rows;
    int local_nnz;
    int N_global;
    hipblasHandle_t hipblas_handle;
    hipsparseHandle_t hipsparse_handle;
    hipStream_t stream;
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
    // --- Device pointers are double ---
    double* d_r;
    double* d_p;
    double* d_t;
    double* d_alpha;
    double* d_beta;
    double* d_rnrm2sqr;
    double* d_rnrm2sqr_prev;
    double* d_pdott;
    double* d_one;
    double* d_minus_one;
    double* d_zero;
    int* d_A_rowptr;
    int* d_A_colidx;   // Still global indices
    double* d_A_vals;  // Values are double
    hipsparseSpMatDescr_t matA_descr;
    hipsparseDnVecDescr_t vecr_descr;
    hipsparseDnVecDescr_t vecp_descr;
    hipsparseDnVecDescr_t vect_descr;
    void* spmv_buffer;
    int niterations;
    double final_residual_norm;    // double
    double initial_residual_norm;  // double
} CGSolverData;

__global__ void spmv_allgather_fused(int num_rows, int comm_size, size_t local_rows, size_t* recvcount_d,
                                     size_t* displs_d, double* d_source_local, double* recvbuf_d,
                                     double* __restrict__ y, const int* __restrict__ rowptr,
                                     const int* __restrict__ colidx, const double* __restrict__ a, double* alpha,
                                     uniconn::Communicator<>* comm) {
    cg::grid_group grid = cg::this_grid();

    // Perform Allgatherv
    if (comm->GlobalSize() > 1) {
        if (blockIdx.x == 0) {
            uniconn::Coordinator<>::AllGatherv<uniconn::ThreadGroup::BLOCK>(d_source_local, recvbuf_d, recvcount_d,
                                                                            displs_d, comm);
        }
    } else if (local_rows > 0 && comm_size == 1) {
        for (int i = grid.thread_rank(); i < local_rows; i += grid.size()) {
            recvbuf_d[i] = d_source_local[i];
        }
    } else if (num_rows > 0 && comm_size == 1) {
        for (int i = grid.thread_rank(); i < num_rows; i += grid.size()) {
            recvbuf_d[i] = 0;
        }
    }
    // Perform SpMV only if the target rank actually has rows
    cg::sync(grid);
    int row_idx = grid.thread_rank();

    for (int i = row_idx; i < local_rows; i += grid.size()) {
        int row_start = rowptr[i];
        int row_end = rowptr[i + 1];

        double sum = 0.0;

        for (int k = row_start; k < row_end; ++k) {
            int col = colidx[k];
            double val = a[k];

            sum += val * recvbuf_d[col];
        }

        y[i] = (*alpha) * sum;
    }
}

__inline__ __device__ double warpReduceSum(unsigned long long mask, double x) {
    for (int i = warpSize / 2; i > 0; i /= 2) {
        x += __shfl_down(x, i);
    }
    return x;
}

__global__ void ddot_allreduce_fused(int n, const double* __restrict x, const double* __restrict y,
                                     double* __restrict dot, uniconn::Communicator<>* comm) {
    cg::grid_group grid = cg::this_grid();

    int n_pes = comm->GlobalSize();

    /* compute per-thread partial results */
    double z = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        z += x[i] * y[i];
    }

    /* perform reduction across threads in a warp */
    z = warpReduceSum(0xffffffff, z);

    if ((threadIdx.x & (warpSize - 1)) == 0) atomicAdd(dot, z);

    if (n_pes > 1) {
        cg::sync(grid);
        if (blockIdx.x == 0) {
            uniconn::Coordinator<>::AllReduce<uniconn::ThreadGroup::BLOCK, uniconn::ReductionOperator::SUM>(dot, 1,
                                                                                                            comm);
        }
    }
}

void cg_solver_free(CGSolverData* cg) {
    // Free device buffers
    uniconn::Memory<>::Free(cg->d_r);
    cg->d_r = NULL;
    uniconn::Memory<>::Free(cg->d_p);
    cg->d_p = NULL;
    GPU_RT_CALL(hipFree(cg->d_t));
    cg->d_t = NULL;
    uniconn::Memory<>::Free(cg->d_rnrm2sqr);
    cg->d_rnrm2sqr = NULL;
    uniconn::Memory<>::Free(cg->d_pdott);
    cg->d_pdott = NULL;
    GPU_RT_CALL(hipFree(cg->d_rnrm2sqr_prev));
    cg->d_rnrm2sqr_prev = NULL;
    GPU_RT_CALL(hipFree(cg->d_alpha));
    cg->d_alpha = NULL;
    GPU_RT_CALL(hipFree(cg->d_beta));
    cg->d_beta = NULL;
    GPU_RT_CALL(hipFree(cg->d_one));
    cg->d_one = NULL;
    GPU_RT_CALL(hipFree(cg->d_minus_one));
    cg->d_minus_one = NULL;
    GPU_RT_CALL(hipFree(cg->d_zero));
    cg->d_zero = NULL;
    GPU_RT_CALL(hipFree(cg->d_A_rowptr));
    cg->d_A_rowptr = NULL;
    GPU_RT_CALL(hipFree(cg->d_A_colidx));
    cg->d_A_colidx = NULL;
    GPU_RT_CALL(hipFree(cg->d_A_vals));
    cg->d_A_vals = NULL;
    GPU_RT_CALL(hipFree(cg->spmv_buffer));
    cg->spmv_buffer = NULL;

    // Destroy cuSPARSE handles
    if (cg->matA_descr) CHECK_HIPSPARSE(hipsparseDestroySpMat(cg->matA_descr));
    cg->matA_descr = NULL;
    if (cg->vecr_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(cg->vecr_descr));
    cg->vecr_descr = NULL;
    if (cg->vecp_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(cg->vecp_descr));
    cg->vecp_descr = NULL;
    if (cg->vect_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(cg->vect_descr));
    cg->vect_descr = NULL;

    GPU_RT_CALL(hipEventDestroy(cg->startEvent));
    GPU_RT_CALL(hipEventDestroy(cg->stopEvent));
}

int cg_solver_init(CGSolverData* cg, int local_rows, int local_nnz, const int* h_A_rowptr,
                   const int* h_A_colidx,        // GLOBAL indices
                   const double* h_A_vals,       // CHANGED: double values
                   hipblasHandle_t hipblas,      // Passed in
                   hipsparseHandle_t hipsparse,  // Passed in
                   MPI_Comm comm,                // Passed in
                   hipsparseSpMVAlg_t alg) {
    if (!cg) return 1;
    if (cg->N_global <= 0) {
        if (cg->comm_rank == 0) fprintf(stderr, "Error: cg_solver_init called with N_global=%d\n", cg->N_global);
        return 1;
    }

    cg->local_rows = local_rows;
    cg->local_nnz = local_nnz;

    if (local_rows > 0) {
        // --- Allocate device vectors (local r, p, t) ---
        cg->d_r = uniconn::Memory<>::Alloc<double>(local_rows);
        cg->d_p = uniconn::Memory<>::Alloc<double>(local_rows);
        GPU_RT_CALL(hipMalloc((void**)&cg->d_t, local_rows * sizeof(double)));  // CHANGED: double
        GPU_RT_CALL(hipMemset(cg->d_r, 0, local_rows * sizeof(double)));        // CHANGED: double
        GPU_RT_CALL(hipMemset(cg->d_p, 0, local_rows * sizeof(double)));        // CHANGED: double
        GPU_RT_CALL(hipMemset(cg->d_t, 0, local_rows * sizeof(double)));        // CHANGED: double

        // --- Allocate and copy sparse matrix (local part) to device ---
        GPU_RT_CALL(hipMalloc((void**)&cg->d_A_rowptr, (local_rows + 1) * sizeof(int)));
        GPU_RT_CALL(hipMemcpy(cg->d_A_rowptr, h_A_rowptr, (local_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
        GPU_RT_CALL(hipMalloc((void**)&cg->d_A_colidx,
                              local_nnz * sizeof(int)));  // Global indices
        if (local_nnz > 0)
            GPU_RT_CALL(hipMemcpy(cg->d_A_colidx, h_A_colidx, local_nnz * sizeof(int), hipMemcpyHostToDevice));
        GPU_RT_CALL(hipMalloc((void**)&cg->d_A_vals,
                              local_nnz * sizeof(double)));  // CHANGED: double
        if (local_nnz > 0)
            GPU_RT_CALL(hipMemcpy(cg->d_A_vals, h_A_vals, local_nnz * sizeof(double),  // CHANGED: double
                                  hipMemcpyHostToDevice));

        // --- Create cuSPARSE descriptors for matrix and LOCAL vectors ---
        CHECK_HIPSPARSE(hipsparseCreateCsr(&cg->matA_descr, local_rows, cg->N_global, local_nnz, cg->d_A_rowptr,
                                           cg->d_A_colidx, cg->d_A_vals, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                           HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));  // CHANGED: HIP_R_64F
        CHECK_HIPSPARSE(hipsparseCreateDnVec(&cg->vecr_descr, local_rows, cg->d_r,
                                             HIP_R_64F));  // CHANGED: HIP_R_64F
        CHECK_HIPSPARSE(hipsparseCreateDnVec(&cg->vecp_descr, local_rows, cg->d_p,
                                             HIP_R_64F));  // CHANGED: HIP_R_64F
        CHECK_HIPSPARSE(hipsparseCreateDnVec(&cg->vect_descr, local_rows, cg->d_t,
                                             HIP_R_64F));  // CHANGED: HIP_R_64F
    } else {
        cg->d_r = cg->d_p = cg->d_t = NULL;
        cg->d_A_rowptr = cg->d_A_colidx = NULL;
        cg->d_A_vals = NULL;
        cg->matA_descr = NULL;
        cg->vecr_descr = NULL;
        cg->vecp_descr = NULL;
        cg->vect_descr = NULL;
    }

    // --- Allocate device scalars ---
    double one = 1.0, minus_one = -1.0, zero = 0.0;                                  // CHANGED: double, literals
    GPU_RT_CALL(hipMalloc((void**)&cg->d_one, sizeof(double)));                      // CHANGED: double
    GPU_RT_CALL(hipMemcpy(cg->d_one, &one, sizeof(double), hipMemcpyHostToDevice));  // CHANGED: double
    GPU_RT_CALL(hipMalloc((void**)&cg->d_minus_one, sizeof(double)));                // CHANGED: double
    GPU_RT_CALL(hipMemcpy(cg->d_minus_one, &minus_one, sizeof(double),               // CHANGED: double
                          hipMemcpyHostToDevice));
    GPU_RT_CALL(hipMalloc((void**)&cg->d_zero, sizeof(double)));                       // CHANGED: double
    GPU_RT_CALL(hipMemcpy(cg->d_zero, &zero, sizeof(double), hipMemcpyHostToDevice));  // CHANGED: double
    cg->d_rnrm2sqr = uniconn::Memory<>::Alloc<double>(1);
    cg->d_pdott = uniconn::Memory<>::Alloc<double>(1);
    GPU_RT_CALL(hipMalloc((void**)&cg->d_rnrm2sqr_prev, sizeof(double)));  // CHANGED: double
    GPU_RT_CALL(hipMalloc((void**)&cg->d_alpha, sizeof(double)));          // CHANGED: double
    GPU_RT_CALL(hipMalloc((void**)&cg->d_beta, sizeof(double)));           // CHANGED: double

    // --- Allocate SpMV buffer (sized for GLOBAL SpMV) ---
    hipsparseDnVecDescr_t tmp_global_vec_descr;
    double* tmp_d_global_vec;  // CHANGED: double
    hipsparseDnVecDescr_t tmp_local_target_descr = cg->vect_descr;
    double* tmp_d_local_target = cg->d_t;  // CHANGED: double

    GPU_RT_CALL(hipMalloc((void**)&tmp_d_global_vec, cg->N_global * sizeof(double)));  // CHANGED: double
    CHECK_HIPSPARSE(hipsparseCreateDnVec(&tmp_global_vec_descr, cg->N_global, tmp_d_global_vec,
                                         HIP_R_64F));  // CHANGED: HIP_R_64F

    hipsparseSpMatDescr_t tmp_mat_descr = cg->matA_descr;
    if (local_rows == 0) {
        CHECK_HIPSPARSE(hipsparseCreateCsr(&tmp_mat_descr, 0, cg->N_global, 0, NULL, NULL, NULL, HIPSPARSE_INDEX_32I,
                                           HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO,
                                           HIP_R_64F));  // CHANGED: HIP_R_64F
        GPU_RT_CALL(hipMalloc((void**)&tmp_d_local_target,
                              1 * sizeof(double)));  // CHANGED: double (Min allocation)
        CHECK_HIPSPARSE(hipsparseCreateDnVec(&tmp_local_target_descr, 0, tmp_d_local_target,
                                             HIP_R_64F));  // CHANGED: HIP_R_64F
    }

    size_t bufferSize = 0;
    CHECK_HIPSPARSE(hipsparseSpMV_bufferSize(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, cg->d_one, tmp_mat_descr,
                                             tmp_global_vec_descr, cg->d_zero, tmp_local_target_descr, HIP_R_64F, alg,
                                             &bufferSize));  // CHANGED: HIP_R_64F
    GPU_RT_CALL(hipMalloc(&cg->spmv_buffer, bufferSize));

    // Clean up temporary sizing resources
    CHECK_HIPSPARSE(hipsparseDestroyDnVec(tmp_global_vec_descr));
    GPU_RT_CALL(hipFree(tmp_d_global_vec));
    if (local_rows == 0) {
        CHECK_HIPSPARSE(hipsparseDestroySpMat(tmp_mat_descr));
        CHECK_HIPSPARSE(hipsparseDestroyDnVec(tmp_local_target_descr));
        GPU_RT_CALL(hipFree(tmp_d_local_target));
    }

    cg->niterations = 0;
    cg->final_residual_norm = INFINITY;
    cg->initial_residual_norm = INFINITY;
    return 0;  // Success
}

// --- allgather_spmv (double version) ---
static int allgather_spmv(CGSolverData* cg,
                          double* d_source_local,        // CHANGED: double
                          double* d_target_local,        // CHANGED: double
                          double* d_comm_buffer_global,  // CHANGED: double
                          size_t* recvcounts, size_t* displs, hipsparseDnVecDescr_t vec_global_descr,
                          hipsparseDnVecDescr_t vec_target_local_descr,
                          double* d_spmv_alpha,  // CHANGED: double
                          double* d_spmv_beta,   // CHANGED: double
                          hipsparseSpMVAlg_t alg, uniconn::Coordinator<>* coordinator) {
    if (cg->local_rows > 0 && d_source_local != NULL) {
        GPU_RT_CALL(hipStreamSynchronize(cg->stream));
    }
    // Perform Allgatherv
    if (cg->comm_size > 1) {
        coordinator->AllGatherv(d_source_local, d_comm_buffer_global, recvcounts, displs, cg->comm);
    } else if (cg->local_rows > 0) {
        GPU_RT_CALL(hipMemcpyAsync(d_comm_buffer_global, d_source_local,
                                   cg->local_rows * sizeof(double),  // CHANGED: double
                                   hipMemcpyDeviceToDevice, cg->stream));
        GPU_RT_CALL(hipStreamSynchronize(cg->stream));
    } else if (cg->N_global > 0) {
        GPU_RT_CALL(hipMemset(d_comm_buffer_global, 0, cg->N_global * sizeof(double)));  // CHANGED: double
    }

    // Perform SpMV only if the target rank actually has rows
    if (cg->local_rows > 0 && cg->matA_descr != NULL && vec_global_descr != NULL && vec_target_local_descr != NULL) {
        CHECK_HIPSPARSE(hipsparseSpMV(cg->hipsparse_handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_spmv_alpha,
                                      cg->matA_descr, vec_global_descr, d_spmv_beta, vec_target_local_descr, HIP_R_64F,
                                      alg,  // CHANGED: HIP_R_64F
                                      cg->spmv_buffer));
    }

    return 0;  // Success
}

// --- cg_solver_solve (double version) ---
int cg_solver_solve(CGSolverData* cg,
                    const double* h_b,  // Host pointer to local b - double
                    double* h_x,        // Host pointer to local x (in/out) - double
                    int maxits,
                    double reltol,  // Relative residual tolerance - double
                    hipsparseSpMVAlg_t alg) {
    if (!cg) return 1;

    int k = 0;
    double alpha = 0.0, beta = 0.0, pdott = 0.0, rnrm2sqr = 0.0,
           rnrm2sqr_prev = 0.0;  // double scalars
    // double local_pdott = 0.0, local_rnrm2sqr = 0.0; // No longer needed with MPI_IN_PLACE
    bool converged = false;

    // Get handles, sizes, device pointers from struct
    hipblasHandle_t hipblas = cg->hipblas_handle;
    hipsparseHandle_t hipsparse = cg->hipsparse_handle;
    hipStream_t stream = cg->stream;
    size_t n = cg->local_rows;
    int N_global = cg->N_global;

    double* d_r = cg->d_r;
    double* d_p = cg->d_p;
    double* d_t = cg->d_t;  // Local vectors (double)

    double* d_alpha_dev = cg->d_alpha;
    double* d_beta_dev = cg->d_beta;  // Scalars (double)
    double* d_rnrm2sqr_dev = cg->d_rnrm2sqr;
    double* d_pdott_dev = cg->d_pdott;  // Scalars (double)
    double* d_one = cg->d_one;
    double* d_minus_one = cg->d_minus_one;  // Constants (double)
    double* d_zero = cg->d_zero;

    int* d_rowptr = cg->d_A_rowptr;
    int* d_colidx = cg->d_A_colidx;
    double* d_vals = cg->d_A_vals;

    // --- Allocate temporary resources for this solve ---
    // Device memory for x and b (local) - double
    double *d_x = NULL, *d_b = NULL;
    hipsparseDnVecDescr_t vecx_descr = NULL, vecb_descr = NULL;
    if (n > 0) {
        d_x = uniconn::Memory<>::Alloc<double>(n);

        GPU_RT_CALL(hipMalloc((void**)&d_b, n * sizeof(double)));                        // CHANGED: double
        GPU_RT_CALL(hipMemcpyAsync(d_x, h_x, n * sizeof(double), hipMemcpyHostToDevice,  // CHANGED: double
                                   stream));
        GPU_RT_CALL(hipMemcpyAsync(d_b, h_b, n * sizeof(double), hipMemcpyHostToDevice,  // CHANGED: double
                                   stream));
        CHECK_HIPSPARSE(hipsparseCreateDnVec(&vecx_descr, n, d_x, HIP_R_64F));  // CHANGED: HIP_R_64F
        CHECK_HIPSPARSE(hipsparseCreateDnVec(&vecb_descr, n, d_b, HIP_R_64F));  // CHANGED: HIP_R_64F
    }

    // Buffers for Allgather communication - double
    size_t* recvcounts = NULL;
    size_t* displs = NULL;  // MPI counts/displs
    size_t* recvcounts_d;
    size_t* displs_d;

    double* d_p_global = NULL;  // Device buffer for global vector (double)
    hipsparseDnVecDescr_t vecpglobal_descr = NULL;

    d_p_global = uniconn::Memory<>::Alloc<double>(N_global);
    if (!d_p_global && N_global > 0)  // Check allocation only if N_global > 0
    {
        fprintf(stderr, "Rank %d Error: Failed malloc for global communication buffer.\n", cg->comm_rank);
        // Cleanup partially allocated buffers
        if (n > 0) {
            uniconn::Memory<>::Free(d_x);
            GPU_RT_CALL(hipFree(d_b));
        }
        if (vecx_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecx_descr));
        if (vecb_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecb_descr));
        return 1;
    }
    CHECK_HIPSPARSE(hipsparseCreateDnVec(&vecpglobal_descr, N_global, d_p_global, HIP_R_64F));  // CHANGED: HIP_R_64F

    if (cg->comm_size > 1) {
        recvcounts = (size_t*)malloc(cg->comm_size * sizeof(size_t));
        displs = (size_t*)malloc(cg->comm_size * sizeof(size_t));
        recvcounts_d = uniconn::Memory<>::Alloc<size_t>(cg->comm_size);
        displs_d = uniconn::Memory<>::Alloc<size_t>(cg->comm_size);
        if (!recvcounts || !displs) {
            fprintf(stderr, "Rank %d: Failed malloc for MPI recvcounts/displs\n", cg->comm_rank);
            // Basic cleanup before abort
            uniconn::Memory<>::Free(d_p_global);
            if (vecpglobal_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecpglobal_descr));
            if (n > 0) {
                uniconn::Memory<>::Free(d_x);
                GPU_RT_CALL(hipFree(d_b));
            }
            if (vecx_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecx_descr));
            if (vecb_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecb_descr));
            MPI_Abort(MPI_COMM_WORLD, 1);  // Abort is safer here
        }

        MPI_CALL(MPI_Allgather(&n, 1, MPI_UNSIGNED_LONG, recvcounts, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD));

        displs[0] = 0;
        for (int i = 1; i < cg->comm_size; ++i) displs[i] = displs[i - 1] + recvcounts[i - 1];
        GPU_RT_CALL(
            hipMemcpyAsync(recvcounts_d, recvcounts, cg->comm_size * sizeof(size_t), hipMemcpyHostToDevice, stream));
        GPU_RT_CALL(hipMemcpyAsync(displs_d, displs, cg->comm_size * sizeof(size_t), hipMemcpyHostToDevice, stream));
    }

    // Configure pointer modes
    hipblasPointerMode_t hipblas_mode_orig;
    hipsparsePointerMode_t hipsparse_mode_orig;
    CHECK_HIPBLAS(hipblasGetPointerMode(hipblas, &hipblas_mode_orig));
    CHECK_HIPBLAS(hipblasSetPointerMode(hipblas, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE(hipsparseGetPointerMode(hipsparse, &hipsparse_mode_orig));
    CHECK_HIPSPARSE(hipsparseSetPointerMode(hipsparse, HIPSPARSE_POINTER_MODE_DEVICE));
    int mingridsize = 0, blocksize = 0;
    uniconn::Communicator<>* comm_d = cg->comm->toDevice();
    uniconn::Coordinator<> coordinator_allgatherv_d_comm1(cg->stream);
    uniconn::Coordinator<> coordinator_allgatherv_d_comm2(cg->stream);
    uniconn::Coordinator<> coordinator_reduce_rnrm2sqr(cg->stream);
    uniconn::Coordinator<> coordinator_reduce_pdott(cg->stream);

    void* kernelargsallgatherv1[] = {(void*)&N_global, (void*)&cg->comm_size, (void*)&n,          (void*)&recvcounts_d,
                                     (void*)&displs_d, (void*)&d_x,           (void*)&d_p_global, (void*)&d_r,
                                     (void*)&d_rowptr, (void*)&d_colidx,      (void*)&d_vals,     (void*)&d_minus_one,
                                     (void*)&comm_d};
    void* kernelargsallgatherv2[] = {(void*)&N_global, (void*)&cg->comm_size, (void*)&n,          (void*)&recvcounts_d,
                                     (void*)&displs_d, (void*)&d_p,           (void*)&d_p_global, (void*)&d_t,
                                     (void*)&d_rowptr, (void*)&d_colidx,      (void*)&d_vals,     (void*)&d_one,
                                     (void*)&comm_d};

    void* kernelargsrnrm2sqr[] = {(void*)&n, (void*)&d_r, (void*)&d_r, (void*)&d_rnrm2sqr_dev, (void*)&comm_d};
    void* kernelargspdott[] = {(void*)&n, (void*)&d_p, (void*)&d_t, (void*)&d_pdott_dev, (void*)&comm_d};

    GPU_RT_CALL(hipOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize, spmv_allgather_fused, 0, 0));
    coordinator_allgatherv_d_comm1.bindKernel<uniconn::LaunchMode::FullDevice>(
        (void*)spmv_allgather_fused, dim3(mingridsize, 1, 1), dim3(blocksize, 1, 1), 0, kernelargsallgatherv1);
    coordinator_allgatherv_d_comm2.bindKernel<uniconn::LaunchMode::FullDevice>(
        (void*)spmv_allgather_fused, dim3(mingridsize, 1, 1), dim3(blocksize, 1, 1), 0, kernelargsallgatherv2);

    GPU_RT_CALL(hipOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize, ddot_allreduce_fused, 0, 0));
    coordinator_reduce_rnrm2sqr.bindKernel<uniconn::LaunchMode::FullDevice>(
        (void*)ddot_allreduce_fused, dim3(mingridsize, 1, 1), dim3(blocksize, 1, 1), 0, kernelargsrnrm2sqr);
    coordinator_reduce_pdott.bindKernel<uniconn::LaunchMode::FullDevice>(
        (void*)ddot_allreduce_fused, dim3(mingridsize, 1, 1), dim3(blocksize, 1, 1), 0, kernelargspdott);

    // --- Initial Setup ---
    cg->niterations = 0;
    if (n > 0) GPU_RT_CALL(hipStreamSynchronize(stream));  // Ensure x and b copied if n>0
    GPU_RT_CALL(hipEventRecord(cg->startEvent, cg->stream));
    // Calculate initial residual: r = b - A*x
    if (n > 0) {
        if constexpr (uniconn::DefaultLaunchType == uniconn::LaunchMode::FullDevice) {
            coordinator_allgatherv_d_comm1.LaunchKernel();
        } else {
            CHECK_HIPBLAS(hipblasDcopy(hipblas, n, d_b, 1, d_r, 1));  // CHANGED: Dcopy
            // SpMV part: r = -1*A*x_global + 1*r
            int spmv_status = allgather_spmv(cg, d_x, d_r,  // Source d_x, target d_r (double)
                                             d_p_global, recvcounts, displs, vecpglobal_descr,
                                             cg->vecr_descr,      // Target is vecr_descr
                                             d_minus_one, d_one,  // alpha=-1, beta=1 (double ptrs)
                                             alg, &coordinator_allgatherv_d_comm1);
        }
        GPU_RT_CALL(hipStreamSynchronize(stream));
    }

    // Calculate initial residual norm (all ranks participate)
    if (n == 0) {
        GPU_RT_CALL(hipMemsetAsync(d_rnrm2sqr_dev, 0, sizeof(double), stream));  // CHANGED: double
    } else {
        if constexpr (uniconn::DefaultLaunchType == uniconn::LaunchMode::FullDevice) {
            coordinator_reduce_rnrm2sqr.LaunchKernel();
        } else {
            CHECK_HIPBLAS(hipblasDdot(  // CHANGED: Ddot
                hipblas, n, d_r, 1, d_r, 1, d_rnrm2sqr_dev));

            coordinator_reduce_rnrm2sqr.AllReduce<uniconn::ReductionOperator::SUM>(d_rnrm2sqr_dev, 1, cg->comm);
        }
    }
    GPU_RT_CALL(hipStreamSynchronize(stream));

    GPU_RT_CALL(hipMemcpy(&rnrm2sqr, d_rnrm2sqr_dev, sizeof(double), hipMemcpyDeviceToHost));  // CHANGED: double

    cg->initial_residual_norm = sqrt(rnrm2sqr);  // CHANGED: sqrt (from math.h)
    cg->final_residual_norm = cg->initial_residual_norm;

    // Optional checks for initial residual (use double precision tolerance)
    /*
    if (cg->initial_residual_norm < 1e-15) { ... }
    if (isnan(cg->initial_residual_norm) || isinf(cg->initial_residual_norm)) { ... }
    */

    // Initial search direction: p = r
    if (n > 0) CHECK_HIPBLAS(hipblasDcopy(hipblas, n, d_r, 1, d_p, 1));  // CHANGED: Dcopy

    // --- Main CG Loop ---
    for (k = 0; k < maxits; k++) {
        cg->niterations = k + 1;
        // 1. Calculate t = A*p (using Allgather)
        if constexpr (uniconn::DefaultLaunchType == uniconn::LaunchMode::FullDevice) {
            coordinator_allgatherv_d_comm2.LaunchKernel();
        } else {
            int spmv_status = allgather_spmv(cg, d_p, d_t,  // Source d_p, target d_t (double)
                                             d_p_global, recvcounts, displs, vecpglobal_descr,
                                             cg->vect_descr,  // Target is vect_descr
                                             d_one, d_zero,   // alpha=1, beta=0 (double ptrs)
                                             alg, &coordinator_allgatherv_d_comm2);
        }
        GPU_RT_CALL(hipStreamSynchronize(stream));

        // 2. Calculate alpha = (r^T * r) / (p^T * t)
        if (n == 0) {
            GPU_RT_CALL(hipMemsetAsync(d_pdott_dev, 0, sizeof(double), stream));  // CHANGED: double
        } else {
            if constexpr (uniconn::DefaultLaunchType == uniconn::LaunchMode::FullDevice) {
                coordinator_reduce_pdott.LaunchKernel();
            } else {
                CHECK_HIPBLAS(hipblasDdot(  // CHANGED: Ddot
                    hipblas, n, d_p, 1, d_t, 1, d_pdott_dev));

                coordinator_reduce_pdott.AllReduce<uniconn::ReductionOperator::SUM>(d_pdott_dev, 1, cg->comm);
            }
            GPU_RT_CALL(hipStreamSynchronize(stream));
        }
        GPU_RT_CALL(hipMemcpy(&pdott, d_pdott_dev, sizeof(double), hipMemcpyDeviceToHost));  // CHANGED: double

        // Check for breakdown (use double precision tolerance)
        double initial_rnorm_sq = cg->initial_residual_norm * cg->initial_residual_norm;

        alpha = rnrm2sqr / pdott;                                        // double division
        GPU_RT_CALL(hipMemcpyAsync(d_alpha_dev, &alpha, sizeof(double),  // CHANGED: double
                                   hipMemcpyHostToDevice, stream));

        // 3. Update solution: x = x + alpha * p (local)
        if (n > 0)
            CHECK_HIPBLAS(hipblasDaxpy(  // CHANGED: Daxpy
                hipblas, n, d_alpha_dev, d_p, 1, d_x, 1));

        // 4. Update residual: r = r - alpha * t (local)
        double minus_alpha = -alpha;
        double* d_minus_alpha_tmp = NULL;                                            // Temp scalar (double)
        GPU_RT_CALL(hipMalloc((void**)&d_minus_alpha_tmp, sizeof(double)));          // CHANGED: double
        GPU_RT_CALL(hipMemcpyAsync(d_minus_alpha_tmp, &minus_alpha, sizeof(double),  // CHANGED: double
                                   hipMemcpyHostToDevice, stream));
        if (n > 0) {
            GPU_RT_CALL(hipStreamSynchronize(stream));  // Ensure minus_alpha ready before Daxpy
            CHECK_HIPBLAS(hipblasDaxpy(                 // CHANGED: Daxpy
                hipblas, n, d_minus_alpha_tmp, d_t, 1, d_r, 1));
        }
        GPU_RT_CALL(hipFree(d_minus_alpha_tmp));  // Free immediately

        // 5. Calculate new residual norm squared ||r_{k+1}||^2
        rnrm2sqr_prev = rnrm2sqr;  // Store previous norm squared (double)
        if (n == 0) {
            GPU_RT_CALL(hipMemsetAsync(d_rnrm2sqr_dev, 0, sizeof(double), stream));  // CHANGED: double
        } else {
            if constexpr (uniconn::DefaultLaunchType == uniconn::LaunchMode::FullDevice) {
                coordinator_reduce_rnrm2sqr.LaunchKernel();
            } else {
                CHECK_HIPBLAS(hipblasDdot(  // CHANGED: Ddot
                    hipblas, n, d_r, 1, d_r, 1, d_rnrm2sqr_dev));

                coordinator_reduce_rnrm2sqr.AllReduce<uniconn::ReductionOperator::SUM>(d_rnrm2sqr_dev, 1, cg->comm);
            }
        }
        GPU_RT_CALL(hipMemcpyAsync(&rnrm2sqr, d_rnrm2sqr_dev, sizeof(double), hipMemcpyDeviceToHost,
                                   stream));  // CHANGED: double

        GPU_RT_CALL(hipStreamSynchronize(stream));  // Ensure rnrm2sqr available on host

        cg->final_residual_norm = sqrt(rnrm2sqr);  // CHANGED: sqrt

        /*
        // 6. Check convergence
        if (cg->initial_residual_norm > 1e-20 && // Avoid division by zero, use double tolerance
            (cg->final_residual_norm / cg->initial_residual_norm < reltol))
        {
            converged = true;
            break; // Exit loop
        }
        // Check for NaN/Inf
        if (isnan(cg->final_residual_norm) || isinf(cg->final_residual_norm))
        {
            if (cg->rank == 0)
                fprintf(
                    stderr,
                    "Warning: Residual is NaN/Inf at iter %d.\n", k);
            break;
        }
        // Optional Stagnation check (use double tolerance)

        if (k > 10 && fabs(rnrm2sqr - rnrm2sqr_prev) < 1e-20 * rnrm2sqr) {
             if(cg->rank == 0) fprintf(stderr, "Warning: Stagnation detected at iter %d.\n", k);
             // Check convergence status before breaking
             if (cg->initial_residual_norm > 1e-20 &&
                (cg->final_residual_norm / cg->initial_residual_norm < reltol)) {
                converged = true;
             }
             break;
        }

        // 7. Calculate beta = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
        if (fabs(rnrm2sqr_prev) < 1e-30) // CHANGED: fabs, very small double tolerance
        {
            if (cg->rank == 0)
            {
                if (rnrm2sqr < 1e-25) // If current residual is also tiny
                {
                    printf(
                        "Info: Previous rnorm^2 is near zero, but "
                        "current rnorm is also small (%e). Assuming "
                        "convergence.\n",
                        cg->final_residual_norm);
                    converged = true;
                }
                else
                {
                    fprintf(
                        stderr,
                        "Warning: Previous rnorm^2 (%e) near zero, "
                        "but current rnorm^2 (%e) is not. Beta "
                        "calculation failed at iter %d.\n",
                        rnrm2sqr_prev, rnrm2sqr, k);
                }
            }
             // Check convergence status before breaking
             if (!converged && cg->initial_residual_norm > 1e-20 &&
                (cg->final_residual_norm / cg->initial_residual_norm < reltol)) {
                converged = true;
             }
            break; // Exit loop if division by zero is likely
        }
        */
        beta = rnrm2sqr / rnrm2sqr_prev;                                                      // double division
        GPU_RT_CALL(hipMemcpyAsync(d_beta_dev, &beta, sizeof(double), hipMemcpyHostToDevice,  // CHANGED: double
                                   stream));

        // 8. Update search direction: p = r + beta * p (local)
        if (n > 0) {
            CHECK_HIPBLAS(hipblasDscal(hipblas, n, d_beta_dev, d_p, 1));  // CHANGED: Dscal
            CHECK_HIPBLAS(hipblasDaxpy(                                   // CHANGED: Daxpy
                hipblas, n, d_one, d_r, 1, d_p, 1));
        }

        GPU_RT_CALL(hipStreamSynchronize(stream));  // Ensure scalar copies complete
    }  // End of CG loop

    GPU_RT_CALL(hipEventRecord(cg->stopEvent, cg->stream));
    GPU_RT_CALL(hipEventSynchronize(cg->stopEvent));

cleanup:
    // Copy final solution x back to host
    if (n > 0) {
        GPU_RT_CALL(hipStreamSynchronize(stream));                                    // Ensure compute is done
        GPU_RT_CALL(hipMemcpy(h_x, d_x, n * sizeof(double), hipMemcpyDeviceToHost));  // CHANGED: double
    }

    // Free resources allocated *in this function*
    if (n > 0) {
        uniconn::Memory<>::Free(d_x);
        d_x = NULL;
        GPU_RT_CALL(hipFree(d_b));
        d_b = NULL;
        if (vecx_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecx_descr));
        vecx_descr = NULL;
        if (vecb_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecb_descr));
        vecb_descr = NULL;
    }
    if (cg->comm_size > 1) {
        free(recvcounts);
        recvcounts = NULL;
        free(displs);
        displs = NULL;
    }
    uniconn::Memory<>::Free(d_p_global);
    d_p_global = NULL;
    if (vecpglobal_descr) CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecpglobal_descr));
    vecpglobal_descr = NULL;

    // Restore original pointer modes
    CHECK_HIPBLAS(hipblasSetPointerMode(hipblas, hipblas_mode_orig));
    CHECK_HIPSPARSE(hipsparseSetPointerMode(hipsparse, hipsparse_mode_orig));

    // Final status message from Rank 0
    if (cg->comm_rank == 0) {
        double final_rel_res = (cg->initial_residual_norm > 1e-20)  // CHANGED: double tolerance
                                   ? (cg->final_residual_norm / cg->initial_residual_norm)
                                   : cg->final_residual_norm;
        if (converged) {
            fprintf(stderr,
                    "CG converged in %d iterations. Final rel resid: "
                    "%.4e\n",  // Use %e or %g for double
                    cg->niterations, final_rel_res);
        } else if (k >= maxits) {
            fprintf(stderr,
                    "CG Max iterations (%d) reached. Final rel resid: "
                    "%.4e\n",  // Use %e or %g for double
                    maxits, final_rel_res);
        } else {
            // k is the index of the *last completed* iteration (0-based)
            // If loop broke early, niterations was set inside loop.
            fprintf(stderr,
                    "CG stopped early at iteration %d without "
                    "converging (reason see warnings). Final rel resid: %.4e\n",
                    cg->niterations,
                    final_rel_res);  // Use niterations stored in cg struct
        }
    }
    return converged ? 0 : 2;
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    uniconn::Environment<> env(argc, argv);
    int size = env.WorldSize();
    int rank = env.WorldRank();
    int local_rank = env.NodeRank();
    env.SetDevice(local_rank);

    int num_devices;
    GPU_RT_CALL(hipGetDeviceCount(&num_devices));
    int device_id;
    GPU_RT_CALL(hipGetDevice(&device_id));
    struct hipDeviceProp_t prop;
    GPU_RT_CALL(hipGetDeviceProperties(&prop, device_id));
    if (rank == 0) fprintf(stderr, "MPI Ranks: %d, HIPDA Devices: %d. Rank 0 using %s\n", size, num_devices, prop.name);
    // Check for double precision support (optional but good practice)
    if (prop.major < 1 || (prop.major == 1 && prop.minor < 3)) {
        if (rank == 0)
            fprintf(stderr, "Warning: GPU %s may lack efficient double precision support (Compute Capability %d.%d).\n",
                    prop.name, prop.major, prop.minor);
        // Consider aborting if DP is strictly required and hardware is too old
        return EXIT_FAILURE;
    }

    uniconn::Communicator<> comm;
    uniconn::Communicator<>* comm_d = comm.toDevice();

    // --- Initialize HIPDA Handles (Unchanged) ---
    hipblasHandle_t hipblas_handle;
    hipsparseHandle_t hipsparse_handle;
    CHECK_HIPBLAS(hipblasCreate(&hipblas_handle));
    CHECK_HIPSPARSE(hipsparseCreate(&hipsparse_handle));

    // --- Generate or Read Matrix ---
    int N_global = 1000000;  // Smaller default for quicker testing maybe?
    int nz_global = 0;       // Determined after generation/read

    int *I_global_um = NULL, *J_global_um = NULL;
    double* val_global_um = NULL;  // CHANGED: double

    if (rank == 0) {
        if (argc > 1) {
            const char* filename = argv[1];
            fprintf(stderr, "Rank 0: Reading matrix %s...\n", filename);
            csr_matrix matrix;
            memset(&matrix, 0, sizeof(csr_matrix));

            // *** Assumes read_mtx_to_csr reads/converts to double ***
            int ret = read_mtx_to_csr(filename, &matrix);
            if (ret != 0) {
                fprintf(stderr, "Rank 0: Failed to read matrix file %s.\n", filename);
                int success = 1;
                MPI_CALL(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));  // Inform others
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            N_global = matrix.M;
            nz_global = matrix.nz;
            fprintf(stderr, "Rank 0: Matrix read: N=%d, NZ=%d\n", N_global, nz_global);

            GPU_RT_CALL(hipMallocManaged((void**)&I_global_um, (N_global + 1) * sizeof(int), hipMemAttachGlobal));
            GPU_RT_CALL(hipMallocManaged((void**)&J_global_um, nz_global * sizeof(int), hipMemAttachGlobal));
            GPU_RT_CALL(hipMallocManaged((void**)&val_global_um, nz_global * sizeof(double),
                                         hipMemAttachGlobal));  // CHANGED: double

            // Copy from host CSR struct (read from file) to UM arrays
            GPU_RT_CALL(hipMemcpy(I_global_um, matrix.row_ptr, (N_global + 1) * sizeof(int),
                                  hipMemcpyHostToHost));  // Actually Host->UM
            GPU_RT_CALL(hipMemcpy(J_global_um, matrix.col_ind, nz_global * sizeof(int),
                                  hipMemcpyHostToHost));  // Actually Host->UM
            GPU_RT_CALL(hipMemcpy(val_global_um, matrix.values, nz_global * sizeof(double),
                                  hipMemcpyHostToHost));  // CHANGED: double, Host->UM

            free_csr_matrix(&matrix);  // Free host memory from reading
        } else {
            // Calculate expected nz for tridiagonal
            nz_global = (N_global <= 0) ? 0 : ((N_global == 1) ? 1 : (3 * N_global - 2));  // Corrected formula
            if (N_global <= 0) {
                fprintf(stderr, "Rank 0 Error: N_global must be positive.\n");
                int success = 1;
                MPI_CALL(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            fprintf(stderr, "Rank 0: Generating %d x %d tridiagonal matrix (expected nz=%d)...\n", N_global, N_global,
                    nz_global);

            GPU_RT_CALL(hipMallocManaged((void**)&I_global_um, (N_global + 1) * sizeof(int), hipMemAttachGlobal));
            GPU_RT_CALL(hipMallocManaged((void**)&J_global_um, nz_global * sizeof(int), hipMemAttachGlobal));
            GPU_RT_CALL(hipMallocManaged((void**)&val_global_um, nz_global * sizeof(double),
                                         hipMemAttachGlobal));  // CHANGED: double

            if (!I_global_um || !J_global_um || !val_global_um) {
                fprintf(stderr, "Rank 0 Error: hipMallocManaged failed.\n");
                int success = 1;
                MPI_CALL(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            genTridiag(I_global_um, J_global_um, val_global_um, N_global, nz_global);

            nz_global = I_global_um[N_global];
            fprintf(stderr, "Rank 0: Generation complete (actual nz=%d).\n", nz_global);
        }
        int success = 0;
        MPI_CALL(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));
    } else {
        int success = 1;  // Assume failure until broadcast
        MPI_CALL(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));
        if (success != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);  // Abort if rank 0 failed
        }
    }

    // Broadcast final N and actual nz
    MPI_CALL(MPI_Bcast(&N_global, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&nz_global, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (N_global <= 0) {
        if (rank == 0) fprintf(stderr, "Error: N_global is zero or negative after setup.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- Distribute ---
    int local_n = 0, local_nnz = 0;
    int *h_rowptr = NULL, *h_colidx = NULL;
    double *h_vals = NULL, *h_b = NULL, *h_x = NULL;  // CHANGED: double

    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    double dist_start_time = MPI_Wtime();

    int dist_status = distribute_tridiag_scatterv(
        N_global, nz_global, I_global_um, J_global_um, val_global_um,                     // Rank 0 passes UM ptrs
        rank, size, MPI_COMM_WORLD, &local_n, &local_nnz, &h_rowptr, &h_colidx, &h_vals,  // h_vals is double** now
        &h_b, &h_x);                                                                      // h_b, h_x are double**
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    double dist_end_time = MPI_Wtime();

    if (rank == 0) {
        fprintf(stderr, "Distribution time (Scatterv): %f seconds\n", dist_end_time - dist_start_time);
        // Free global UM arrays now that distribution is done
        if (I_global_um) GPU_RT_CALL(hipFree(I_global_um));
        I_global_um = NULL;
        if (J_global_um) GPU_RT_CALL(hipFree(J_global_um));
        J_global_um = NULL;
        if (val_global_um) GPU_RT_CALL(hipFree(val_global_um));
        val_global_um = NULL;
        fprintf(stderr, "Rank 0: Freed global Unified Memory arrays.\n");
    }

    if (dist_status != 0) {
        if (rank == 0) fprintf(stderr, "Failed to distribute matrix.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Check for null pointers after distribution
    if (local_n > 0 && (!h_rowptr || !h_b || !h_x || (local_nnz > 0 && (!h_colidx || !h_vals)))) {
        fprintf(stderr,
                "Rank %d Error: Null host pointers after successful "
                "distribution (n=%d, nnz=%d).\n",
                rank, local_n, local_nnz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(stderr, "Rank %d: Received partition n=%d, nnz=%d. Device ID: %d\n", rank, local_n, local_nnz, device_id);

    // --- Initialize Solver ---
    CGSolverData cg_solver;
    memset(&cg_solver, 0, sizeof(CGSolverData));            // Zero out struct
    hipsparseSpMVAlg_t spmv_alg = HIPSPARSE_SPMV_CSR_ALG1;  // Or ALG2
    int init_status = 1;                                    // Default to failure

    // --- Timing Setup ---
    hipEvent_t start_event, stop_event;
    GPU_RT_CALL(hipEventCreate(&start_event));
    GPU_RT_CALL(hipEventCreate(&stop_event));

    // Set common fields before calling init
    cg_solver.N_global = N_global;
    cg_solver.comm = &comm;
    cg_solver.comm_rank = rank;
    cg_solver.comm_size = size;
    cg_solver.hipblas_handle = hipblas_handle;
    cg_solver.hipsparse_handle = hipsparse_handle;
    cg_solver.stream = 0;  // Default stream
    cg_solver.startEvent = start_event;
    cg_solver.stopEvent = stop_event;

    init_status = cg_solver_init(&cg_solver, local_n, local_nnz, h_rowptr, h_colidx, h_vals,  // Pass double h_vals
                                 hipblas_handle, hipsparse_handle, MPI_COMM_WORLD, spmv_alg);

    // Free Host Buffers for matrix now (copied to device in init)
    free(h_rowptr);
    h_rowptr = NULL;
    free(h_colidx);
    h_colidx = NULL;
    free(h_vals);
    h_vals = NULL;

    if (init_status != 0) {
        if (rank == 0) fprintf(stderr, "Solver initialization failed.\n");
        cg_solver_free(&cg_solver);  // Attempt cleanup before abort
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(stderr, "Rank %d: Solver initialized.\n", rank);

    int max_iterations = DEFAULT_ITER_NUM;  // More realistic max iterations
    double tolerance = 1e-10;               // double tolerance
    if (rank == 0)
        fprintf(stderr, "Starting solve: N_global=%d, max_its=%d, rel_tol=%.2e\n", cg_solver.N_global, max_iterations,
                tolerance);

    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    double solve_start_time = MPI_Wtime();

    int solve_status = cg_solver_solve(&cg_solver, h_b, h_x,  // Pass double vectors
                                       max_iterations, tolerance, spmv_alg);

    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    double solve_end_time = MPI_Wtime();

    if (rank == 0) {
        float milliseconds = 0;
        GPU_RT_CALL(hipEventElapsedTime(&milliseconds, cg_solver.startEvent, cg_solver.stopEvent));

        fprintf(stdout, "cg, unc, %s, %d  %.3f ms\n", get_filename_from_path(argv[1]), cg_solver.comm_size,
                milliseconds);
    }

    fprintf(stderr, "Rank %d: Cleaning up...\n", rank);
    cg_solver_free(&cg_solver);

    free(h_b);
    h_b = NULL;
    free(h_x);
    h_x = NULL;

    CHECK_HIPBLAS(hipblasDestroy(hipblas_handle));
    CHECK_HIPSPARSE(hipsparseDestroy(hipsparse_handle));

    fprintf(stderr, "Rank %d: Finalizing Uniconn.\n", rank);
    return 0;
}
