#include "../include/common.hpp"
#define ROOT 0

// --- Solver Data Structure (Using double) ---
typedef struct {
    MPI_Comm comm;
    int rank;
    int comm_size;
    int local_rows;
    int* local_rows_global;
    int local_nnz;
    int N_global;
    ///
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    cudaStream_t stream;
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
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
    cusparseSpMatDescr_t matA_descr;
    cusparseDnVecDescr_t vecr_descr;
    cusparseDnVecDescr_t vecp_descr;
    cusparseDnVecDescr_t vect_descr;
    void* spmv_buffer;
    int niterations;
    double final_residual_norm;    // double
    double initial_residual_norm;  // double
} CGSolverData;

void nvshmem_allgatherv(const void* sendbuf, const int* sendcounts, const int* displs, double* recvbuf, int* recvcount,
                        size_t datatype_size, int root, cudaStream_t stream) {
    int my_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    int n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);

    nvshmemx_barrier_all_on_stream(stream);

    for (int dest_pe = 0; dest_pe < n_pes; ++dest_pe) {
        nvshmemx_double_put_nbi_on_stream(recvbuf + displs[my_pe], (double*)sendbuf, recvcount[my_pe], dest_pe, stream);
    }

    nvshmemx_barrier_all_on_stream(stream);
}

// --- Solver Functions (double versions) ---

void cg_solver_free(CGSolverData* cg) {
    // Free device buffers
    nvshmem_free(cg->d_r);
    cg->d_r = NULL;
    nvshmem_free(cg->d_p);
    cg->d_p = NULL;
    GPU_RT_CALL(cudaFree(cg->d_t));
    cg->d_t = NULL;
    nvshmem_free(cg->d_rnrm2sqr);
    cg->d_rnrm2sqr = NULL;
    nvshmem_free(cg->d_pdott);
    cg->d_pdott = NULL;
    GPU_RT_CALL(cudaFree(cg->d_rnrm2sqr_prev));
    cg->d_rnrm2sqr_prev = NULL;
    GPU_RT_CALL(cudaFree(cg->d_alpha));
    cg->d_alpha = NULL;
    GPU_RT_CALL(cudaFree(cg->d_beta));
    cg->d_beta = NULL;
    GPU_RT_CALL(cudaFree(cg->d_one));
    cg->d_one = NULL;
    GPU_RT_CALL(cudaFree(cg->d_minus_one));
    cg->d_minus_one = NULL;
    GPU_RT_CALL(cudaFree(cg->d_zero));
    cg->d_zero = NULL;
    GPU_RT_CALL(cudaFree(cg->d_A_rowptr));
    cg->d_A_rowptr = NULL;
    GPU_RT_CALL(cudaFree(cg->d_A_colidx));
    cg->d_A_colidx = NULL;
    GPU_RT_CALL(cudaFree(cg->d_A_vals));
    cg->d_A_vals = NULL;
    GPU_RT_CALL(cudaFree(cg->spmv_buffer));
    cg->spmv_buffer = NULL;

    // Destroy cuSPARSE handles
    if (cg->matA_descr) CHECK_CUSPARSE(cusparseDestroySpMat(cg->matA_descr));
    cg->matA_descr = NULL;
    if (cg->vecr_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(cg->vecr_descr));
    cg->vecr_descr = NULL;
    if (cg->vecp_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(cg->vecp_descr));
    cg->vecp_descr = NULL;
    if (cg->vect_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(cg->vect_descr));
    cg->vect_descr = NULL;

    // event
    GPU_RT_CALL(cudaEventDestroy(cg->startEvent));
    GPU_RT_CALL(cudaEventDestroy(cg->stopEvent));
}

int cg_solver_init(CGSolverData* cg, int local_rows, int local_nnz, const int* h_A_rowptr,
                   const int* h_A_colidx,      // GLOBAL indices
                   const double* h_A_vals,     // CHANGED: double values
                   cublasHandle_t cublas,      // Passed in
                   cusparseHandle_t cusparse,  // Passed in
                   MPI_Comm comm,              // Passed in
                   cusparseSpMVAlg_t alg) {
    if (!cg) return 1;
    if (cg->N_global <= 0) {
        if (cg->rank == 0) fprintf(stderr, "Error: cg_solver_init called with N_global=%d\n", cg->N_global);
        return 1;
    }

    cg->local_rows = local_rows;
    cg->local_nnz = local_nnz;

    MPI_Barrier(MPI_COMM_WORLD);

    if (cg->rank == 0) {
        cg->local_rows_global = (int*)malloc(cg->comm_size * sizeof(int));
        MPI_CALL(MPI_Gather(&cg->local_rows, 1, MPI_INT, cg->local_rows_global, 1, MPI_INT, 0, MPI_COMM_WORLD));
    } else {
        MPI_CALL(MPI_Gather(&cg->local_rows, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD));
    }

    if (local_rows > 0) {
        // --- Allocate device vectors (local r, p, t) ---
        // GPU_RT_CALL(cudaMalloc((void**)&cg->d_r, local_rows * sizeof(double)));  // CHANGED: double
        cg->d_r = (double*)nvshmem_malloc(local_rows * sizeof(double));
        /*
        GPU_RT_CALL(cudaMalloc(
            (void**) &cg->d_p, local_rows * sizeof(double))); // CHANGED: double
        */
        cg->d_p = (double*)nvshmem_malloc(local_rows * sizeof(double));
        GPU_RT_CALL(cudaMalloc((void**)&cg->d_t, local_rows * sizeof(double)));  // CHANGED: double
        GPU_RT_CALL(cudaMemset(cg->d_r, 0, local_rows * sizeof(double)));        // CHANGED: double
        GPU_RT_CALL(cudaMemset(cg->d_p, 0, local_rows * sizeof(double)));        // CHANGED: double
        GPU_RT_CALL(cudaMemset(cg->d_t, 0, local_rows * sizeof(double)));        // CHANGED: double

        // --- Allocate and copy sparse matrix (local part) to device ---
        GPU_RT_CALL(cudaMalloc((void**)&cg->d_A_rowptr, (local_rows + 1) * sizeof(int)));
        GPU_RT_CALL(cudaMemcpy(cg->d_A_rowptr, h_A_rowptr, (local_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        GPU_RT_CALL(cudaMalloc((void**)&cg->d_A_colidx,
                               local_nnz * sizeof(int)));  // Global indices

        if (local_nnz > 0)
            GPU_RT_CALL(cudaMemcpy(cg->d_A_colidx, h_A_colidx, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
        GPU_RT_CALL(cudaMalloc((void**)&cg->d_A_vals,
                               local_nnz * sizeof(double)));  // CHANGED: double
        if (local_nnz > 0)
            GPU_RT_CALL(cudaMemcpy(cg->d_A_vals, h_A_vals, local_nnz * sizeof(double),  // CHANGED: double
                                   cudaMemcpyHostToDevice));

        // --- Create cuSPARSE descriptors for matrix and LOCAL vectors ---
        CHECK_CUSPARSE(cusparseCreateCsr(&cg->matA_descr, local_rows, cg->N_global, local_nnz, cg->d_A_rowptr,
                                         cg->d_A_colidx, cg->d_A_vals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));  // CHANGED: CUDA_R_64F
        CHECK_CUSPARSE(cusparseCreateDnVec(&cg->vecr_descr, local_rows, cg->d_r,
                                           CUDA_R_64F));  // CHANGED: CUDA_R_64F
        CHECK_CUSPARSE(cusparseCreateDnVec(&cg->vecp_descr, local_rows, cg->d_p,
                                           CUDA_R_64F));  // CHANGED: CUDA_R_64F
        CHECK_CUSPARSE(cusparseCreateDnVec(&cg->vect_descr, local_rows, cg->d_t,
                                           CUDA_R_64F));  // CHANGED: CUDA_R_64F
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
    double one = 1.0, minus_one = -1.0, zero = 0.0;                                    // CHANGED: double, literals
    GPU_RT_CALL(cudaMalloc((void**)&cg->d_one, sizeof(double)));                       // CHANGED: double
    GPU_RT_CALL(cudaMemcpy(cg->d_one, &one, sizeof(double), cudaMemcpyHostToDevice));  // CHANGED: double
    GPU_RT_CALL(cudaMalloc((void**)&cg->d_minus_one, sizeof(double)));                 // CHANGED: double
    GPU_RT_CALL(cudaMemcpy(cg->d_minus_one, &minus_one, sizeof(double),                // CHANGED: double
                           cudaMemcpyHostToDevice));
    GPU_RT_CALL(cudaMalloc((void**)&cg->d_zero, sizeof(double)));                        // CHANGED: double
    GPU_RT_CALL(cudaMemcpy(cg->d_zero, &zero, sizeof(double), cudaMemcpyHostToDevice));  // CHANGED: double
    // GPU_RT_CALL(cudaMalloc((void**) &cg->d_rnrm2sqr, sizeof(double))); // CHANGED: double
    cg->d_rnrm2sqr = (double*)nvshmem_malloc(sizeof(double));
    // GPU_RT_CALL(cudaMalloc((void**) &cg->d_pdott, sizeof(double))); // CHANGED: double
    cg->d_pdott = (double*)nvshmem_malloc(sizeof(double));
    GPU_RT_CALL(cudaMalloc((void**)&cg->d_rnrm2sqr_prev, sizeof(double)));  // CHANGED: double
    //(void**) &cg->d_rnrm2sqr_prev = nvshmem_malloc(sizeof(double));
    GPU_RT_CALL(cudaMalloc((void**)&cg->d_alpha, sizeof(double)));  // CHANGED: double
    GPU_RT_CALL(cudaMalloc((void**)&cg->d_beta, sizeof(double)));   // CHANGED: double

    // --- Allocate SpMV buffer (sized for GLOBAL SpMV) ---
    cusparseDnVecDescr_t tmp_global_vec_descr;
    double* tmp_d_global_vec;  // CHANGED: double
    cusparseDnVecDescr_t tmp_local_target_descr = cg->vect_descr;
    double* tmp_d_local_target = cg->d_t;  // CHANGED: double

    GPU_RT_CALL(cudaMalloc((void**)&tmp_d_global_vec, cg->N_global * sizeof(double)));  // CHANGED: double
    CHECK_CUSPARSE(cusparseCreateDnVec(&tmp_global_vec_descr, cg->N_global, tmp_d_global_vec,
                                       CUDA_R_64F));  // CHANGED: CUDA_R_64F

    cusparseSpMatDescr_t tmp_mat_descr = cg->matA_descr;
    if (local_rows == 0) {
        CHECK_CUSPARSE(cusparseCreateCsr(&tmp_mat_descr, 0, cg->N_global, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_64F));  // CHANGED: CUDA_R_64F
        GPU_RT_CALL(cudaMalloc((void**)&tmp_d_local_target,
                               1 * sizeof(double)));  // CHANGED: double (Min allocation)
        CHECK_CUSPARSE(cusparseCreateDnVec(&tmp_local_target_descr, 0, tmp_d_local_target,
                                           CUDA_R_64F));  // CHANGED: CUDA_R_64F
    }

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, cg->d_one, tmp_mat_descr,
                                           tmp_global_vec_descr, cg->d_zero, tmp_local_target_descr, CUDA_R_64F, alg,
                                           &bufferSize));  // CHANGED: CUDA_R_64F
    GPU_RT_CALL(cudaMalloc(&cg->spmv_buffer, bufferSize));

    // Clean up temporary sizing resources
    CHECK_CUSPARSE(cusparseDestroyDnVec(tmp_global_vec_descr));
    GPU_RT_CALL(cudaFree(tmp_d_global_vec));
    if (local_rows == 0) {
        CHECK_CUSPARSE(cusparseDestroySpMat(tmp_mat_descr));
        CHECK_CUSPARSE(cusparseDestroyDnVec(tmp_local_target_descr));
        GPU_RT_CALL(cudaFree(tmp_d_local_target));
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
                          int* recvcounts, int* displs, cusparseDnVecDescr_t vec_global_descr,
                          cusparseDnVecDescr_t vec_target_local_descr,
                          double* d_spmv_alpha,  // CHANGED: double
                          double* d_spmv_beta,   // CHANGED: double
                          cusparseSpMVAlg_t alg) {
    if (cg->local_rows > 0 && d_source_local != NULL) {
        GPU_RT_CALL(cudaStreamSynchronize(cg->stream));
    }
    // Perform Allgatherv
    if (cg->comm_size > 1) {
        nvshmem_allgatherv(d_source_local, cg->local_rows_global, displs, d_comm_buffer_global, recvcounts,
                           sizeof(double), 0, cg->stream);
    } else if (cg->local_rows > 0) {
        GPU_RT_CALL(cudaMemcpyAsync(d_comm_buffer_global, d_source_local,
                                    cg->local_rows * sizeof(double),  // CHANGED: double
                                    cudaMemcpyDeviceToDevice, cg->stream));
        GPU_RT_CALL(cudaStreamSynchronize(cg->stream));
    } else if (cg->N_global > 0) {
        GPU_RT_CALL(cudaMemset(d_comm_buffer_global, 0, cg->N_global * sizeof(double)));  // CHANGED: double
    }

    // Perform SpMV only if the target rank actually has rows
    if (cg->local_rows > 0 && cg->matA_descr != NULL && vec_global_descr != NULL && vec_target_local_descr != NULL) {
        CHECK_CUSPARSE(cusparseSpMV(cg->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, d_spmv_alpha, cg->matA_descr,
                                    vec_global_descr, d_spmv_beta, vec_target_local_descr, CUDA_R_64F,
                                    alg,  // CHANGED: CUDA_R_64F
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
                    cusparseSpMVAlg_t alg) {
    if (!cg) return 1;

    int k = 0;
    double alpha = 0.0, beta = 0.0, pdott = 0.0, rnrm2sqr = 0.0,
           rnrm2sqr_prev = 0.0;  // double scalars
    // double local_pdott = 0.0, local_rnrm2sqr = 0.0; // No longer needed with MPI_IN_PLACE
    bool converged = false;

    // Get handles, sizes, device pointers from struct
    cublasHandle_t cublas = cg->cublas_handle;
    cusparseHandle_t cusparse = cg->cusparse_handle;
    cudaStream_t stream = cg->stream;
    int n = cg->local_rows;
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

    // --- Allocate temporary resources for this solve ---
    // Device memory for x and b (local) - double
    double *d_x = NULL, *d_b = NULL;
    cusparseDnVecDescr_t vecx_descr = NULL, vecb_descr = NULL;
    if (n > 0) {
        d_x = (double*)nvshmem_malloc(n * sizeof(double));
        GPU_RT_CALL(cudaMalloc((void**)&d_b, n * sizeof(double)));                         // CHANGED: double
        GPU_RT_CALL(cudaMemcpyAsync(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice,  // CHANGED: double
                                    stream));
        GPU_RT_CALL(cudaMemcpyAsync(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice,  // CHANGED: double
                                    stream));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecx_descr, n, d_x, CUDA_R_64F));  // CHANGED: CUDA_R_64F
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecb_descr, n, d_b, CUDA_R_64F));  // CHANGED: CUDA_R_64F
    }

    // Buffers for Allgather communication - double
    int* recvcounts = NULL;
    int* displs = NULL;         // MPI counts/displs
    double* d_p_global = NULL;  // Device buffer for global vector (double)
    cusparseDnVecDescr_t vecpglobal_descr = NULL;

    // GPU_RT_CALL(cudaMalloc((void**) &d_p_global, N_global * sizeof(double))); // CHANGED: double
    d_p_global = (double*)nvshmem_malloc(N_global * sizeof(double));
    if (!d_p_global && N_global > 0)  // Check allocation only if N_global > 0
    {
        fprintf(stderr, "Rank %d Error: Failed malloc for global communication buffer.\n", cg->rank);
        // Cleanup partially allocated buffers
        if (n > 0) {
            nvshmem_free(d_x);
            GPU_RT_CALL(cudaFree(d_b));
        }
        if (vecx_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vecx_descr));
        if (vecb_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vecb_descr));
        return 1;
    }
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecpglobal_descr, N_global, d_p_global, CUDA_R_64F));  // CHANGED: CUDA_R_64F

    if (cg->comm_size > 1) {
        recvcounts = (int*)malloc(cg->comm_size * sizeof(int));
        displs = (int*)malloc(cg->comm_size * sizeof(int));
        if (!recvcounts || !displs) {
            fprintf(stderr, "Rank %d: Failed malloc for MPI recvcounts/displs\n", cg->rank);
            // Basic cleanup before abort
            nvshmem_free(d_p_global);
            if (vecpglobal_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vecpglobal_descr));
            if (n > 0) {
                nvshmem_free(d_x);
                GPU_RT_CALL(cudaFree(d_b));
            }
            if (vecx_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vecx_descr));
            if (vecb_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vecb_descr));
            MPI_Abort(cg->comm, 1);  // Abort is safer here
        }
        MPI_CALL(MPI_Allgather(&n, 1, MPI_INT, recvcounts, 1, MPI_INT, cg->comm));
        displs[0] = 0;
        for (int i = 1; i < cg->comm_size; ++i) displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    // Configure pointer modes
    cublasPointerMode_t cublas_mode_orig;
    cusparsePointerMode_t cusparse_mode_orig;
    CHECK_CUBLAS(cublasGetPointerMode(cublas, &cublas_mode_orig));
    CHECK_CUBLAS(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUSPARSE(cusparseGetPointerMode(cusparse, &cusparse_mode_orig));
    CHECK_CUSPARSE(cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE));

    // --- Initial Setup ---
    cg->niterations = 0;
    if (n > 0) GPU_RT_CALL(cudaStreamSynchronize(stream));  // Ensure x and b copied if n>0

    GPU_RT_CALL(cudaEventRecord(cg->startEvent, cg->stream));

    // Calculate initial residual: r = b - A*x
    if (n > 0) {
        CHECK_CUBLAS(cublasDcopy(cublas, n, d_b, 1, d_r, 1));  // CHANGED: Dcopy
        // SpMV part: r = -1*A*x_global + 1*r

        int spmv_status =
            allgather_spmv(cg, d_x, d_r,  // Source d_x, target d_r (double)
                           d_p_global, recvcounts, displs, vecpglobal_descr, cg->vecr_descr,  // Target is vecr_descr
                           d_minus_one, d_one,  // alpha=-1, beta=1 (double ptrs)
                           alg);

        if (spmv_status != 0) goto cleanup;
    }

    // Calculate initial residual norm (all ranks participate)
    if (n == 0) {
        GPU_RT_CALL(cudaMemsetAsync(d_rnrm2sqr_dev, 0, sizeof(double), stream));  // CHANGED: double
    } else {
        CHECK_CUBLAS(cublasDdot(  // CHANGED: Ddot
            cublas, n, d_r, 1, d_r, 1, d_rnrm2sqr_dev));
        nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, d_rnrm2sqr_dev, d_rnrm2sqr_dev, 1, cg->stream);
    }
    GPU_RT_CALL(cudaStreamSynchronize(stream));

    GPU_RT_CALL(cudaMemcpy(&rnrm2sqr, d_rnrm2sqr_dev, sizeof(double), cudaMemcpyDeviceToHost));  // CHANGED: double

    cg->initial_residual_norm = sqrt(rnrm2sqr);  // CHANGED: sqrt (from math.h)
    cg->final_residual_norm = cg->initial_residual_norm;

    // Optional checks for initial residual (use double precision tolerance)
    /*
    if (cg->initial_residual_norm < 1e-15) { ... }
    if (isnan(cg->initial_residual_norm) || isinf(cg->initial_residual_norm)) { ... }
    */

    // Initial search direction: p = r
    if (n > 0) CHECK_CUBLAS(cublasDcopy(cublas, n, d_r, 1, d_p, 1));  // CHANGED: Dcopy

    // --- Main CG Loop ---
    for (k = 0; k < maxits; k++) {
        cg->niterations = k + 1;
        // 1. Calculate t = A*p (using Allgather)

        int spmv_status =
            allgather_spmv(cg, d_p, d_t,  // Source d_p, target d_t (double)
                           d_p_global, recvcounts, displs, vecpglobal_descr, cg->vect_descr,  // Target is vect_descr
                           d_one, d_zero,  // alpha=1, beta=0 (double ptrs)
                           alg);

        if (spmv_status != 0) goto cleanup;

        // 2. Calculate alpha = (r^T * r) / (p^T * t)
        if (n == 0) {
            GPU_RT_CALL(cudaMemsetAsync(d_pdott_dev, 0, sizeof(double), stream));  // CHANGED: double
        } else {
            CHECK_CUBLAS(cublasDdot(  // CHANGED: Ddot
                cublas, n, d_p, 1, d_t, 1, d_pdott_dev));

            nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, d_pdott_dev, d_pdott_dev, 1, cg->stream);

            GPU_RT_CALL(cudaStreamSynchronize(stream));
        }
        GPU_RT_CALL(cudaMemcpy(&pdott, d_pdott_dev, sizeof(double), cudaMemcpyDeviceToHost));  // CHANGED: double

        // Check for breakdown (use double precision tolerance)
        double initial_rnorm_sq = cg->initial_residual_norm * cg->initial_residual_norm;
        if (fabs(pdott) < 1e-20 * initial_rnorm_sq)  // CHANGED: fabs, double tolerance
        {
            if (cg->rank == 0)
                fprintf(stderr,
                        "Warning: p^T*A*p = %e near zero rel. to init "
                        "rnorm^2 at iter %d. Stopping.\n",
                        pdott, k);
            // Check convergence status before breaking
            if (cg->initial_residual_norm > 1e-20 &&  // Avoid division by zero
                (cg->final_residual_norm / cg->initial_residual_norm < reltol)) {
                converged = true;
            }
            break;  // Exit loop
        }

        alpha = rnrm2sqr / pdott;                                         // double division
        GPU_RT_CALL(cudaMemcpyAsync(d_alpha_dev, &alpha, sizeof(double),  // CHANGED: double
                                    cudaMemcpyHostToDevice, stream));

        // 3. Update solution: x = x + alpha * p (local)
        if (n > 0)
            CHECK_CUBLAS(cublasDaxpy(  // CHANGED: Daxpy
                cublas, n, d_alpha_dev, d_p, 1, d_x, 1));

        // 4. Update residual: r = r - alpha * t (local)
        double minus_alpha = -alpha;
        double* d_minus_alpha_tmp = NULL;                                             // Temp scalar (double)
        GPU_RT_CALL(cudaMalloc((void**)&d_minus_alpha_tmp, sizeof(double)));          // CHANGED: double
        GPU_RT_CALL(cudaMemcpyAsync(d_minus_alpha_tmp, &minus_alpha, sizeof(double),  // CHANGED: double
                                    cudaMemcpyHostToDevice, stream));
        if (n > 0) {
            GPU_RT_CALL(cudaStreamSynchronize(stream));  // Ensure minus_alpha ready before Daxpy
            CHECK_CUBLAS(cublasDaxpy(                    // CHANGED: Daxpy
                cublas, n, d_minus_alpha_tmp, d_t, 1, d_r, 1));
        }
        GPU_RT_CALL(cudaFree(d_minus_alpha_tmp));  // Free immediately

        // 5. Calculate new residual norm squared ||r_{k+1}||^2
        rnrm2sqr_prev = rnrm2sqr;  // Store previous norm squared (double)
        if (n == 0) {
            GPU_RT_CALL(cudaMemsetAsync(d_rnrm2sqr_dev, 0, sizeof(double), stream));  // CHANGED: double
        } else {
            CHECK_CUBLAS(cublasDdot(  // CHANGED: Ddot
                cublas, n, d_r, 1, d_r, 1, d_rnrm2sqr_dev));

            nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, d_rnrm2sqr_dev, d_rnrm2sqr_dev, 1, cg->stream);
        }

        GPU_RT_CALL(cudaMemcpyAsync(&rnrm2sqr, d_rnrm2sqr_dev, sizeof(double), cudaMemcpyDeviceToHost,
                                    stream));        // CHANGED: double
        GPU_RT_CALL(cudaStreamSynchronize(stream));  // Ensure rnrm2sqr available on host

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
        beta = rnrm2sqr / rnrm2sqr_prev;                                                        // double division
        GPU_RT_CALL(cudaMemcpyAsync(d_beta_dev, &beta, sizeof(double), cudaMemcpyHostToDevice,  // CHANGED: double
                                    stream));

        // 8. Update search direction: p = r + beta * p (local)
        if (n > 0) {
            CHECK_CUBLAS(cublasDscal(cublas, n, d_beta_dev, d_p, 1));  // CHANGED: Dscal
            CHECK_CUBLAS(cublasDaxpy(                                  // CHANGED: Daxpy
                cublas, n, d_one, d_r, 1, d_p, 1));
        }

        GPU_RT_CALL(cudaStreamSynchronize(stream));  // Ensure scalar copies complete
    }

    GPU_RT_CALL(cudaEventRecord(cg->stopEvent, cg->stream));
    GPU_RT_CALL(cudaEventSynchronize(cg->stopEvent));

cleanup:
    // Copy final solution x back to host
    if (n > 0) {
        GPU_RT_CALL(cudaStreamSynchronize(stream));                                     // Ensure compute is done
        GPU_RT_CALL(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));  // CHANGED: double
    }

    // Free resources allocated *in this function*
    if (n > 0) {
        nvshmem_free(d_x);
        d_x = NULL;
        GPU_RT_CALL(cudaFree(d_b));
        d_b = NULL;
        if (vecx_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vecx_descr));
        vecx_descr = NULL;
        if (vecb_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vecb_descr));
        vecb_descr = NULL;
    }
    if (cg->comm_size > 1) {
        free(recvcounts);
        recvcounts = NULL;
        free(displs);
        displs = NULL;
    }
    nvshmem_free(d_p_global);
    d_p_global = NULL;
    if (vecpglobal_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vecpglobal_descr));
    vecpglobal_descr = NULL;

    // Restore original pointer modes
    CHECK_CUBLAS(cublasSetPointerMode(cublas, cublas_mode_orig));
    CHECK_CUSPARSE(cusparseSetPointerMode(cusparse, cusparse_mode_orig));

    // Final status message from Rank 0
    if (cg->rank == 0) {
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
                    cg->niterations, final_rel_res);  // Use niterations stored in cg struct
        }
    }
    return converged ? 0 : 2;
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    MPI_CALL(MPI_Init(&argc, &argv));

    int rank, size;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    // --- Determine Local Rank for GPU Selection (Unchanged) ---
    int local_rank = -1;
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_size(local_comm, &local_size));
        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    // --- Determine GPU (Unchanged) ---
    int num_devices;
    GPU_RT_CALL(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device_id = local_rank % num_devices;
    GPU_RT_CALL(cudaSetDevice(device_id));
    struct cudaDeviceProp prop;
    GPU_RT_CALL(cudaGetDeviceProperties(&prop, device_id));
    if (rank == 0) fprintf(stderr, "MPI Ranks: %d, CUDA Devices: %d. Rank 0 using %s\n", size, num_devices, prop.name);
    // Check for double precision support (optional but good practice)
    if (prop.major < 1 || (prop.major == 1 && prop.minor < 3)) {
        if (rank == 0)
            fprintf(stderr, "Warning: GPU %s may lack efficient double precision support (Compute Capability %d.%d).\n",
                    prop.name, prop.major, prop.minor);
        // Consider aborting if DP is strictly required and hardware is too old
        // MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attributes;

    mpi_comm = MPI_COMM_WORLD;
    attributes.mpi_comm = &mpi_comm;

    int status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attributes);
    if (status != 0) {
        fprintf(stderr, "[MPI Rank %d] NVSHMEM initialization failed with status %d\n", rank, status);
        MPI_Abort(MPI_COMM_WORLD, status);  // Abort MPI if NVSHMEM fails
        return 1;                           // Or exit(1);
    }

    int mype, npes;

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (rank == 0) {
        fprintf(stderr, "MPI Initialized: size = %d\n", size);
        fprintf(stderr, "NVSHMEM Initialized via MPI: npes = %d\n", npes);
    }

    // Check consistency on each rank
    if (rank != mype || size != npes) {
        fprintf(stderr, "[MPI Rank %d / NVSHMEM PE %d] Mismatch detected! MPI Size: %d, NVSHMEM NPES: %d\n", rank, mype,
                size, npes);
        nvshmem_finalize();  // Try to finalize cleanly
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) fprintf(stderr, "MPI Ranks and NVSHMEM PEs consistency verified.\n");
    }

    // --- Initialize CUDA Handles (Unchanged) ---
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));

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

            GPU_RT_CALL(cudaMallocManaged((void**)&I_global_um, (N_global + 1) * sizeof(int), cudaMemAttachGlobal));
            GPU_RT_CALL(cudaMallocManaged((void**)&J_global_um, nz_global * sizeof(int), cudaMemAttachGlobal));
            GPU_RT_CALL(cudaMallocManaged((void**)&val_global_um, nz_global * sizeof(double),
                                          cudaMemAttachGlobal));  // CHANGED: double

            // Copy from host CSR struct (read from file) to UM arrays
            GPU_RT_CALL(cudaMemcpy(I_global_um, matrix.row_ptr, (N_global + 1) * sizeof(int),
                                   cudaMemcpyHostToHost));  // Actually Host->UM
            GPU_RT_CALL(cudaMemcpy(J_global_um, matrix.col_ind, nz_global * sizeof(int),
                                   cudaMemcpyHostToHost));  // Actually Host->UM
            GPU_RT_CALL(cudaMemcpy(val_global_um, matrix.values, nz_global * sizeof(double),
                                   cudaMemcpyHostToHost));  // CHANGED: double, Host->UM

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

            GPU_RT_CALL(cudaMallocManaged((void**)&I_global_um, (N_global + 1) * sizeof(int), cudaMemAttachGlobal));
            GPU_RT_CALL(cudaMallocManaged((void**)&J_global_um, nz_global * sizeof(int), cudaMemAttachGlobal));
            GPU_RT_CALL(cudaMallocManaged((void**)&val_global_um, nz_global * sizeof(double),
                                          cudaMemAttachGlobal));  // CHANGED: double

            if (!I_global_um || !J_global_um || !val_global_um) {
                fprintf(stderr, "Rank 0 Error: cudaMallocManaged failed.\n");
                int success = 1;
                MPI_CALL(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            genTridiag(I_global_um, J_global_um, val_global_um, N_global, nz_global);

            // Check actual nz generated (important!)
            nz_global = I_global_um[N_global];
            fprintf(stderr, "Rank 0: Generation complete (actual nz=%d).\n", nz_global);
        }
        // Broadcast success (0) or failure (1)
        int success = 0;
        MPI_CALL(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));
    } else {
        // Other ranks wait for success signal
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
        if (I_global_um) GPU_RT_CALL(cudaFree(I_global_um));
        I_global_um = NULL;
        if (J_global_um) GPU_RT_CALL(cudaFree(J_global_um));
        J_global_um = NULL;
        if (val_global_um) GPU_RT_CALL(cudaFree(val_global_um));
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
    memset(&cg_solver, 0, sizeof(CGSolverData));          // Zero out struct
    cusparseSpMVAlg_t spmv_alg = CUSPARSE_SPMV_CSR_ALG1;  // Or ALG2
    int init_status = 1;                                  // Default to failure

    cudaEvent_t start_event, stop_event;
    GPU_RT_CALL(cudaEventCreate(&start_event));
    GPU_RT_CALL(cudaEventCreate(&stop_event));

    // Set common fields before calling init
    cg_solver.N_global = N_global;
    cg_solver.comm = MPI_COMM_WORLD;
    cg_solver.rank = rank;
    cg_solver.comm_size = size;
    cg_solver.cublas_handle = cublas_handle;
    cg_solver.cusparse_handle = cusparse_handle;
    cg_solver.stream = 0;  // Default stream
    cg_solver.startEvent = start_event;
    cg_solver.stopEvent = stop_event;

    init_status = cg_solver_init(&cg_solver, local_n, local_nnz, h_rowptr, h_colidx, h_vals,  // Pass double h_vals
                                 cublas_handle, cusparse_handle, MPI_COMM_WORLD, spmv_alg);

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

    // --- Solve ---
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
        GPU_RT_CALL(cudaEventElapsedTime(&milliseconds, cg_solver.startEvent, cg_solver.stopEvent));

        fprintf(stdout, "cg, nvshmem_h, %s, %d  %.3f ms\n", get_filename_from_path(argv[1]), cg_solver.comm_size,
                milliseconds);
    }
    fflush(stdout);

    // --- Cleanup ---
    fprintf(stderr, "Rank %d: Cleaning up...\n", rank);
    cg_solver_free(&cg_solver);

    free(h_b);
    h_b = NULL;
    free(h_x);
    h_x = NULL;

    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));

    fprintf(stderr, "Rank %d: Finalizing MPI.\n", rank);
    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());
    return 0;
}
