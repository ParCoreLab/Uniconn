#ifndef __UNICONN_EXAMPLES_CG_INCLUDE_COMMON_HPP_
#define __UNICONN_EXAMPLES_CG_INCLUDE_COMMON_HPP_

#include "../../common/utils.hpp"
#include "mmio.h"

#define DEFAULT_ITER_NUM 10000
#define DEFAULT_NUM_ROW 10485760 * 2
#define DEFAULT_NNZ (DEFAULT_NUM_ROW - 2) * 3 + 4

#if defined UNC_HAS_ROCM
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>
#define UNC_CG_RUNTIME_PREFIX hip
#elif defined UNC_HAS_CUDA
#include <cublas_v2.h>
#include <cusparse.h>
#define UNC_CG_RUNTIME_PREFIX cu
#else
#error "Define UNC_HAS_ROCM or UNC_HAS_CUDA"
#endif

#define CHECK_HIPBLAS(call)                                                                      \
    do {                                                                                        \
        hipblasStatus_t status = call;                                                           \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                                  \
            fprintf(stderr, "HIPBLAS Error at %s:%d - Status %d\n", __FILE__, __LINE__, status); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

#define CHECK_HIPSPARSE(call)                                                                      \
    do {                                                                                          \
        hipsparseStatus_t status = call;                                                           \
        if (status != HIPSPARSE_STATUS_SUCCESS) {                                                  \
            fprintf(stderr, "HIPSPARSE Error at %s:%d - Status %d\n", __FILE__, __LINE__, status); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                         \
        }                                                                                         \
    } while (0)

typedef struct {
    int M;           // Number of rows
    int N;           // Number of columns
    int nz;          // Number of non-zeros
    int *row_ptr;    // Row pointers (size M+1)
    int *col_ind;    // Column indices (size nz)
    double *values;  // Non-zero values (size nz) - CHANGED TO double
} csr_matrix;


const char *get_filename_with_extension(const char *path) {
    if (path == NULL) {
        return NULL;
    }
    const char *last_slash = strrchr(path, '/');
    if (last_slash == NULL) {
        return path;  // No '/' found
    } else {
        return last_slash + 1;  // Point after the last '/'
    }
}

char *get_filename_from_path(const char *full_path) {
    const char *filename_ext = get_filename_with_extension(full_path);
    if (filename_ext == NULL) {
        return NULL;  // Input path was NULL
    }

    // Find the last dot in the filename part
    const char *last_dot = strrchr(filename_ext, '.');
    size_t base_len;
    char *basename = NULL;

    // Check if a dot was found AND if it's followed by "mtx"
    if (last_dot != NULL && strcmp(last_dot, ".mtx") == 0) {
        // Calculate length *before* the dot
        base_len = last_dot - filename_ext;
    } else {
        // No ".mtx" extension found, use the whole filename_ext
        base_len = strlen(filename_ext);
    }

    // Allocate memory for the new basename string (+1 for null terminator)
    basename = (char *)malloc(base_len + 1);
    if (basename == NULL) {
        perror("Error allocating memory for basename");
        return NULL;  // Allocation failed
    }

    // Copy the base part of the filename
    strncpy(basename, filename_ext, base_len);
    basename[base_len] = '\0';  // Ensure null termination

    return basename;
}

void free_csr_matrix(csr_matrix *mat) {
    if (!mat) return;
    free(mat->row_ptr);
    free(mat->col_ind);
    free(mat->values);
    // Optional: Reset pointers if the struct itself isn't freed immediately
    mat->row_ptr = NULL;
    mat->col_ind = NULL;
    mat->values = NULL;
    mat->M = 0;
    mat->N = 0;
    mat->nz = 0;
}

int read_mtx_to_csr(const char *filename, csr_matrix *csr) {
    int M, N, nz;
    int *coo_I, *coo_J;
    double *coo_val;  // CHANGED TO double*
    int ret_code;

    // 1. Read the MTX file into COO format using mmio.c function
    //    Note: mm_read_unsymmetric_sparse handles banner reading, size reading,
    //          data reading, and converts to 0-based indexing internally.
    //    *** CRITICAL ASSUMPTION: This function must be adapted/chosen to handle double ***
    ret_code = mm_read_unsymmetric_sparse(filename, &M, &N, &nz, &coo_val, &coo_I, &coo_J);

    if (ret_code != 0) {
        fprintf(stderr, "Error reading Matrix Market file %s: ", filename);
        // mm_read_unsymmetric_sparse returns -1 for file/format errors
        if (ret_code == -1) {
            fprintf(stderr, "File or format error.\n");
            // Could refine error reporting based on mmio return codes if needed
        } else {
            fprintf(stderr, "Unknown error code %d.\n", ret_code);
        }
        return -1;  // Indicate file/read error
    }

    fprintf(stderr, "Successfully read COO data: M=%d, N=%d, nz=%d\n", M, N, nz);

    // 2. Allocate memory for CSR arrays
    csr->M = M;
    csr->N = N;
    csr->nz = nz;
    csr->row_ptr = (int *)malloc((M + 1) * sizeof(int));
    csr->col_ind = (int *)malloc(nz * sizeof(int));
    csr->values = (double *)malloc(nz * sizeof(double));  // CHANGED TO double

    if (!csr->row_ptr || !csr->col_ind || !csr->values) {
        fprintf(stderr, "Error allocating memory for CSR matrix.\n");
        free(csr->row_ptr);  // Free potentially allocated parts
        free(csr->col_ind);
        free(csr->values);
        free(coo_I);  // Free COO arrays from mm_read_unsymmetric_sparse
        free(coo_J);
        free(coo_val);
        return -2;  // Indicate memory allocation error
    }

    // 3. Convert COO to CSR
    //    (Using a common two-pass approach)

    // Initialize row_ptr counters to zero
    memset(csr->row_ptr, 0, (M + 1) * sizeof(int));

    // --- Pass 1: Count non-zeros in each row ---
    for (int i = 0; i < nz; i++) {
        csr->row_ptr[coo_I[i] + 1]++;
    }

    // --- Pass 2: Calculate cumulative sum for row_ptr ---
    for (int i = 0; i < M; i++) {
        csr->row_ptr[i + 1] += csr->row_ptr[i];
    }

    // --- Pass 3: Populate col_ind and values ---
    int *current_row_pos = (int *)malloc((M + 1) * sizeof(int));
    if (!current_row_pos) {
        fprintf(stderr, "Error allocating temporary memory for CSR conversion.\n");
        free_csr_matrix(csr);  // Free CSR arrays
        free(coo_I);
        free(coo_J);
        free(coo_val);  // Free COO arrays
        return -2;
    }
    memcpy(current_row_pos, csr->row_ptr, (M + 1) * sizeof(int));

    for (int i = 0; i < nz; i++) {
        int row = coo_I[i];
        int col = coo_J[i];
        double val = coo_val[i];  // CHANGED TO double

        int pos = current_row_pos[row];

        csr->col_ind[pos] = col;
        csr->values[pos] = val;  // Assign double value

        current_row_pos[row]++;
    }

    // 4. Free the temporary COO arrays and the temporary position array
    free(coo_I);
    free(coo_J);
    free(coo_val);
    free(current_row_pos);

    fprintf(stderr, "Conversion to CSR format complete.\n");
    return 0;  // Success
}

// --- Matrix Generation ---

/* genTridiag: generate a tridiagonal symmetric matrix in CSR format
 */
void genTridiag(int *I, int *J, double *val, int N, int nz)  // CHANGED: double* val
{
    // Use fixed seed for reproducibility during testing
    srand(12345);

    if (N <= 0) {
        if (N == 0) I[0] = 0;  // Handle N=0 case
        return;
    }

    I[0] = 0;
    // First row entries
    J[0] = 0;                                   // Col 0
    val[0] = (double)rand() / RAND_MAX + 10.0;  // CHANGED: (double), 10.0
    if (N > 1) {
        J[1] = 1;                            // Col 1
        val[1] = (double)rand() / RAND_MAX;  // CHANGED: (double)
        I[1] = 2;
    } else {
        I[1] = 1;  // Only diagonal if N=1
    }

    for (int i = 1; i < N; i++) {
        int current_row_start_nnz = I[i];  // NNZ count before this row
        int nnz_idx = current_row_start_nnz;

        // Entry from previous row (Col i-1)
        J[nnz_idx] = i - 1;
        // Find the corresponding value generated for row i-1, col i
        int prev_row_offdiag_idx = -1;
        for (int k = I[i - 1]; k < I[i]; ++k) {  // Search non-zeros of row i-1
            if (J[k] == i) {                     // Found the entry (i-1, i)
                prev_row_offdiag_idx = k;
                break;
            }
        }
        if (prev_row_offdiag_idx != -1) {
            val[nnz_idx] = val[prev_row_offdiag_idx];  // Symmetry
        } else {
            fprintf(stderr,
                    "Error in genTridiag: Could not find symmetric "
                    "counterpart for row %d, col %d.\n",
                    i, i - 1);
            val[nnz_idx] = 0.0;  // CHANGED: 0.0
        }
        nnz_idx++;

        // Diagonal entry (Col i)
        J[nnz_idx] = i;
        val[nnz_idx] = (double)rand() / RAND_MAX + 10.0;  // CHANGED: (double), 10.0
        nnz_idx++;

        // Entry for next row (Col i+1), if not the last row
        if (i < N - 1) {
            J[nnz_idx] = i + 1;
            val[nnz_idx] = (double)rand() / RAND_MAX;  // CHANGED: (double)
            nnz_idx++;
        }

        I[i + 1] = nnz_idx;  // Update row pointer for the *next* row
    }

    // Final check of non-zero count
    if (I[N] != nz) {
        fprintf(stderr,
                "Warning: genTridiag calculated nz %d does not match "
                "expected nz %d for N=%d.\n",
                I[N], nz, N);
    }
}

int distribute_tridiag_scatterv(int N_global_in, int nz_global_in, int *I_global_um, int *J_global_um,
                                double *val_global_um,  // CHANGED: double (Global UM arrays)
                                int rank, int size, MPI_Comm comm, int *local_n_out, int *local_nnz_out,
                                int **h_rowptr_out, int **h_colidx_out, double **h_vals_out,  // CHANGED: double*
                                double **h_b_out, double **h_x_out)                           // CHANGED: double*
{
    int success = 0;
    int N_global = N_global_in;
    int nz_global = nz_global_in;  // Use local copy

    // --- Rank 0: Precompute all counts and displacements ---
    int *local_n_all = NULL;
    int *local_nnz_all = NULL;
    int *displs_n = NULL;
    int *displs_nnz = NULL;
    int *displs_rowptr = NULL;
    double *b_global_host = NULL;  // CHANGED: double
    double *x_global_host = NULL;  // CHANGED: double
    // *** Host Buffers for J and val on Rank 0 ***
    int *J_global_host = NULL;
    double *val_global_host = NULL;  // CHANGED: double

    if (rank == 0) {
        fprintf(stderr, "Rank 0: Calculating distribution info for Scatterv...\n");
        local_n_all = (int *)malloc(size * sizeof(int));
        local_nnz_all = (int *)malloc(size * sizeof(int));
        displs_n = (int *)malloc(size * sizeof(int));
        displs_nnz = (int *)malloc(size * sizeof(int));
        displs_rowptr = (int *)malloc(size * sizeof(int));
        b_global_host = (double *)malloc(N_global * sizeof(double));  // CHANGED: double
        x_global_host = (double *)malloc(N_global * sizeof(double));  // CHANGED: double
        // *** Allocate host J and val ***
        J_global_host = (int *)malloc(nz_global * sizeof(int));
        val_global_host = (double *)malloc(nz_global * sizeof(double));  // CHANGED: double

        if (!local_n_all || !local_nnz_all || !displs_n || !displs_nnz || !displs_rowptr || !b_global_host ||
            !x_global_host || !J_global_host || !val_global_host) {  // Check new buffers
            fprintf(stderr, "Rank 0 Error: Failed malloc for Scatterv metadata/host staging buffers.\n");
            success = 1;
        }

        if (success == 0) {
            // --- Copy UM J and val to Host J and val ---
            fprintf(stderr, "Rank 0: Copying J and val from Unified Memory to Host buffers...\n");
            memcpy(J_global_host, J_global_um, nz_global * sizeof(int));
            memcpy(val_global_host, val_global_um, nz_global * sizeof(double));  // CHANGED: double
            fprintf(stderr, "Rank 0: Host copy complete.\n");

            // --- Calculate counts/displacements (same as before) ---
            int current_row_start_global = 0;
            int current_nnz_start_global = 0;
            int current_rowptr_start_global = 0;
            int base_local_n = (N_global > 0) ? N_global / size : 0;
            int remainder = (N_global > 0) ? N_global % size : 0;

            // ... (same as before) ...
            for (int r = 0; r < size; ++r) {
                local_n_all[r] = base_local_n + (r < remainder ? 1 : 0);
                displs_n[r] = current_row_start_global;
                int row_end_global = current_row_start_global + local_n_all[r];
                if (row_end_global > N_global) {
                    success = 1;
                    break;
                }
                int nnz_end_global = I_global_um[row_end_global];
                local_nnz_all[r] = nnz_end_global - current_nnz_start_global;
                displs_nnz[r] = current_nnz_start_global;
                displs_rowptr[r] = current_row_start_global;
                if (local_nnz_all[r] < 0) {
                    success = 1;
                    break;
                }
                current_row_start_global = row_end_global;
                current_nnz_start_global = nnz_end_global;
            }

            // Initialize global b and x on Rank 0 (host buffers)
            if (success == 0) {
                for (int i = 0; i < N_global; ++i) {
                    b_global_host[i] = 1.0;  // CHANGED: 1.0
                    x_global_host[i] = 0.0;  // CHANGED: 0.0
                }
            }
        }
    }  // End of Rank 0 precomputation

    // --- Broadcast success/failure status from Rank 0 ---
    MPI_CALL(MPI_Bcast(&success, 1, MPI_INT, 0, comm));
    if (success != 0) {
        if (rank == 0) {  // Free metadata if allocated
            free(local_n_all);
            free(local_nnz_all);
            free(displs_n);
            free(displs_nnz);
            free(displs_rowptr);
            free(b_global_host);
            free(x_global_host);
            free(J_global_host);
            free(val_global_host);  // Free new buffers
        }
        fprintf(stderr, "Rank %d: Aborting due to failure on Rank 0 during precomputation/staging.\n", rank);
        MPI_Abort(comm, 1);
    }

    // --- Scatter local sizes ---
    MPI_CALL(MPI_Scatter(local_n_all, 1, MPI_INT, local_n_out, 1, MPI_INT, 0, comm));
    MPI_CALL(MPI_Scatter(local_nnz_all, 1, MPI_INT, local_nnz_out, 1, MPI_INT, 0, comm));

    // --- Allocate local buffers ---
    int local_n = *local_n_out;
    int local_nnz = *local_nnz_out;
    int local_rowptr_size = local_n + 1;
    *h_rowptr_out = (int *)malloc(local_rowptr_size * sizeof(int));
    *h_colidx_out = (int *)malloc(local_nnz * sizeof(int));
    *h_vals_out = (double *)malloc(local_nnz * sizeof(double));  // CHANGED: double
    *h_b_out = (double *)malloc(local_n * sizeof(double));       // CHANGED: double
    *h_x_out = (double *)malloc(local_n * sizeof(double));       // CHANGED: double
    int *h_I_segment = (int *)malloc(local_rowptr_size * sizeof(int));
    if (!*h_rowptr_out || !*h_colidx_out || !*h_vals_out || !*h_b_out || !*h_x_out || !h_I_segment) {
        fprintf(stderr, "Rank %d: Malloc error for local buffers (n=%d, nnz=%d)\n", rank, local_n, local_nnz);
        MPI_Abort(comm, 1);
    }

    // --- Scatter Actual Data ---
    if (rank == 0) fprintf(stderr, "Rank 0: Starting MPI_Scatterv calls (using host buffers for J/val)...\n");

    // Scatter J (column indices) - *** Use HOST buffer ***
    MPI_CALL(
        MPI_Scatterv(J_global_host, local_nnz_all, displs_nnz, MPI_INT, *h_colidx_out, local_nnz, MPI_INT, 0, comm));

    // Scatter val (values) - *** Use HOST buffer ***
    MPI_CALL(MPI_Scatterv(val_global_host, local_nnz_all, displs_nnz, MPI_DOUBLE,  // CHANGED: MPI_DOUBLE
                           *h_vals_out, local_nnz, MPI_DOUBLE,                      // CHANGED: MPI_DOUBLE
                           0, comm));

    // Scatter I segments (row pointer offsets) - *** UM pointer is OK here ***
    int *counts_rowptr = NULL;
    if (rank == 0) {
        counts_rowptr = (int *)malloc(size * sizeof(int));
        if (!counts_rowptr) {
            MPI_Abort(comm, 1);
        }
        for (int r = 0; r < size; ++r) counts_rowptr[r] = local_n_all[r] + 1;
    }
    MPI_CALL(MPI_Scatterv(I_global_um, counts_rowptr, displs_rowptr, MPI_INT, h_I_segment, local_rowptr_size, MPI_INT,
                           0, comm));

    // Scatter b vector - Use HOST buffer
    MPI_CALL(MPI_Scatterv(b_global_host, local_n_all, displs_n, MPI_DOUBLE,  // CHANGED: MPI_DOUBLE
                           *h_b_out, local_n, MPI_DOUBLE,                     // CHANGED: MPI_DOUBLE
                           0, comm));

    // Scatter x vector - Use HOST buffer
    MPI_CALL(MPI_Scatterv(x_global_host, local_n_all, displs_n, MPI_DOUBLE,  // CHANGED: MPI_DOUBLE
                           *h_x_out, local_n, MPI_DOUBLE,                     // CHANGED: MPI_DOUBLE
                           0, comm));

    if (rank == 0) fprintf(stderr, "Rank 0: Finished MPI_Scatterv calls.\n");

    // --- Postprocessing: Calculate local row pointers ---
    int nnz_offset = h_I_segment[0];
    for (int i = 0; i < local_rowptr_size; ++i) {
        (*h_rowptr_out)[i] = h_I_segment[i] - nnz_offset;
    }

    // --- Cleanup ---
    free(h_I_segment);  // Free temporary buffer
    if (rank == 0) {
        free(local_n_all);
        free(local_nnz_all);
        free(displs_n);
        free(displs_nnz);
        free(displs_rowptr);
        free(b_global_host);
        free(x_global_host);
        free(counts_rowptr);
        // *** Free Host J and val ***
        free(J_global_host);
        free(val_global_host);
    }

    return 0;  // Success
}


// // #if defined UNC_CG_RUNTIME_PREFIX
// // #define UNC_GPU_RT_CG(symb) UNC_ADD_PREFIX(UNC_CG_RUNTIME_PREFIX, symb)

// // // Functions
// // #define UncGpublasDdot UNC_GPU_RT_CG(blasDdot)
// // #define UncGpublasDcopy UNC_GPU_RT_CG(blasDcopy)
// // // blasDaxpy
// // // blasDscal
// // // blasSetPointerMode
// // // blasHandle_t
// // // blasCreate
// // // blasDestroy
// // // blasStatus_t
// // // sparseStatus_t
// // // HIPBLAS_STATUS_SUCCESS
// // // SPARSE_STATUS_SUCCESS
// // // sparseHandle_t
// // // sparseSpMatDescr_t
// // // sparseDnVecDescr_t
// // // sparseSpMVAlg_t
// // // sparseDestroySpMat
// // // sparseDestroyDnVec
// // // sparseCreateCsr
// // // sparseCreateDnVec
// // // HIPSPARSE_INDEX_32I
// // // HIPSPARSE_INDEX_BASE_ZERO
// // // Types
// // #define UncGpuMemcpyKind UNC_GPU_RT(MemcpyKind)

// // // Enum values
// // #define UncGpuErrorNotReady UNC_GPU_RT(ErrorNotReady)

// // // Special APIs
// // #if defined UNC_HAS_ROCM
// // #define UncGpuFreeHost hipHostFree

// // #elif defined UNC_HAS_CUDA
// // #define UncGpuFreeHost cudaFreeHost

// // #endif
// // #endif  // defined UNC_GPU_RUNTIME_PREFIX

// typedef double real;

// constexpr real tol = 1e-5f;

/* avoid Windows warnings (for example: strcpy, fscanf, etc.) */
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

/* various __inline__ __device__  function to initialize a T_ELEM */
template <typename T_ELEM>
__inline__ T_ELEM convertGet(int);
template <>
__inline__ float convertGet<float>(int x) {
    return float(x);
}

template <>
__inline__ double convertGet<double>(int x) {
    return double(x);
}

template <typename T_ELEM>
__inline__ T_ELEM convertGet(float);
template <>
__inline__ float convertGet<float>(float x) {
    return float(x);
}

template <>
__inline__ double convertGet<double>(float x) {
    return double(x);
}

template <typename T_ELEM>
__inline__ T_ELEM convertGet(float, float);
template <>
__inline__ float convertGet<float>(float x, float y) {
    return float(x);
}

template <>
__inline__ double convertGet<double>(float x, float y) {
    return double(x);
}

template <typename T_ELEM>
__inline__ T_ELEM convertGet(double);
template <>
__inline__ float convertGet<float>(double x) {
    return float(x);
}

template <>
__inline__ double convertGet<double>(double x) {
    return double(x);
}

template <typename T_ELEM>
__inline__ T_ELEM convertGet(double, double);
template <>
__inline__ float convertGet<float>(double x, double y) {
    return float(x);
}

template <>
__inline__ double convertGet<double>(double x, double y) {
    return double(x);
}

static void compress_index(const int *Ind, int nnz, int m, int *Ptr, int base) {
    int i;

    /* initialize everything to zero */
    for (i = 0; i < m + 1; i++) {
        Ptr[i] = 0;
    }
    /* count elements in every row */
    Ptr[0] = base;
    for (i = 0; i < nnz; i++) {
        Ptr[Ind[i] + (1 - base)]++;
    }
    /* add all the values */
    for (i = 0; i < m; i++) {
        Ptr[i + 1] += Ptr[i];
    }
}

struct cooFormat {
    int i;
    int j;
    int p;  // permutation
};

int cmp_cooFormat_csr(struct cooFormat *s, struct cooFormat *t) {
    if (s->i < t->i) {
        return -1;
    } else if (s->i > t->i) {
        return 1;
    } else {
        return s->j - t->j;
    }
}

int cmp_cooFormat_csc(struct cooFormat *s, struct cooFormat *t) {
    if (s->j < t->j) {
        return -1;
    } else if (s->j > t->j) {
        return 1;
    } else {
        return s->i - t->i;
    }
}

typedef int (*FUNPTR)(const void *, const void *);
typedef int (*FUNPTR2)(struct cooFormat *s, struct cooFormat *t);

static FUNPTR2 fptr_array[2] = {
    cmp_cooFormat_csr,
    cmp_cooFormat_csc,
};

static int verify_pattern(int m, int nnz, int *csrRowPtr, int *csrColInd) {
    int i, col, start, end, base_index;
    int error_found = 0;

    if (nnz != (csrRowPtr[m] - csrRowPtr[0])) {
        fprintf(stderr, "Error (nnz check failed): (csrRowPtr[%d]=%d - csrRowPtr[%d]=%d) != (nnz=%d)\n", 0,
                csrRowPtr[0], m, csrRowPtr[m], nnz);
        error_found = 1;
    }

    base_index = csrRowPtr[0];
    if ((0 != base_index) && (1 != base_index)) {
        fprintf(stderr, "Error (base index check failed): base index = %d\n", base_index);
        error_found = 1;
    }

    for (i = 0; (!error_found) && (i < m); i++) {
        start = csrRowPtr[i] - base_index;
        end = csrRowPtr[i + 1] - base_index;
        if (start > end) {
            fprintf(stderr, "Error (corrupted row): csrRowPtr[%d] (=%d) > csrRowPtr[%d] (=%d)\n", i, start + base_index,
                    i + 1, end + base_index);
            error_found = 1;
        }
        for (col = start; col < end; col++) {
            if (csrColInd[col] < base_index) {
                fprintf(stderr, "Error (column vs. base index check failed): csrColInd[%d] < %d\n", col, base_index);
                error_found = 1;
            }
            if ((col < (end - 1)) && (csrColInd[col] >= csrColInd[col + 1])) {
                fprintf(stderr,
                        "Error (sorting of the column indecis check failed): (csrColInd[%d]=%d) >= "
                        "(csrColInd[%d]=%d)\n",
                        col, csrColInd[col], col + 1, csrColInd[col + 1]);
                error_found = 1;
            }
        }
    }
    return error_found;
}

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m, int *n, int *nnz, T_ELEM **aVal,
                       int **aRowInd, int **aColInd, int extendSymMatrix) {
    MM_typecode matcode;
    double *tempVal;
    int *tempRowInd, *tempColInd;
    double *tval;
    int *trow, *tcol;
    int *csrRowPtr, *cscColPtr;
    int i, j, error, base, count;
    struct cooFormat *work;

    /* read the matrix */
    error = mm_read_mtx_crd(filename, m, n, nnz, &trow, &tcol, &tval, &matcode);
    if (error) {
        fprintf(stderr, "!!!! can not open file: '%s'\n", filename);
        return 1;
    }

    /* start error checking */
    if (mm_is_complex(matcode) && ((elem_type != 'z') && (elem_type != 'c'))) {
        fprintf(stderr, "!!!! complex matrix requires type 'z' or 'c'\n");
        return 1;
    }

    if (mm_is_dense(matcode) || mm_is_array(matcode) || mm_is_pattern(matcode) /*|| mm_is_integer(matcode)*/) {
        fprintf(stderr, "!!!! dense, array, pattern and integer matrices are not supported\n");
        return 1;
    }

    /* if necessary symmetrize the pattern (transform from triangular to full) */
    if ((extendSymMatrix) && (mm_is_symmetric(matcode) || mm_is_hermitian(matcode) || mm_is_skew(matcode))) {
        // count number of non-diagonal elements
        count = 0;
        for (i = 0; i < (*nnz); i++) {
            if (trow[i] != tcol[i]) {
                count++;
            }
        }
        // allocate space for the symmetrized matrix
        tempRowInd = (int *)malloc((*nnz + count) * sizeof(int));
        tempColInd = (int *)malloc((*nnz + count) * sizeof(int));
        if (mm_is_real(matcode) || mm_is_integer(matcode)) {
            tempVal = (double *)malloc((*nnz + count) * sizeof(double));
        } else {
            tempVal = (double *)malloc(2 * (*nnz + count) * sizeof(double));
        }
        // copy the elements regular and transposed locations
        for (j = 0, i = 0; i < (*nnz); i++) {
            tempRowInd[j] = trow[i];
            tempColInd[j] = tcol[i];
            if (mm_is_real(matcode) || mm_is_integer(matcode)) {
                tempVal[j] = tval[i];
            } else {
                tempVal[2 * j] = tval[2 * i];
                tempVal[2 * j + 1] = tval[2 * i + 1];
            }
            j++;
            if (trow[i] != tcol[i]) {
                tempRowInd[j] = tcol[i];
                tempColInd[j] = trow[i];
                if (mm_is_real(matcode) || mm_is_integer(matcode)) {
                    if (mm_is_skew(matcode)) {
                        tempVal[j] = -tval[i];
                    } else {
                        tempVal[j] = tval[i];
                    }
                } else {
                    if (mm_is_hermitian(matcode)) {
                        tempVal[2 * j] = tval[2 * i];
                        tempVal[2 * j + 1] = -tval[2 * i + 1];
                    } else {
                        tempVal[2 * j] = tval[2 * i];
                        tempVal[2 * j + 1] = tval[2 * i + 1];
                    }
                }
                j++;
            }
        }
        (*nnz) += count;
        // free temporary storage
        free(trow);
        free(tcol);
        free(tval);
    } else {
        tempRowInd = trow;
        tempColInd = tcol;
        tempVal = tval;
    }
    // life time of (trow, tcol, tval) is over.
    // please use COO format (tempRowInd, tempColInd, tempVal)

    // use qsort to sort COO format
    work = (struct cooFormat *)malloc(sizeof(struct cooFormat) * (*nnz));
    if (NULL == work) {
        fprintf(stderr, "!!!! allocation error, malloc failed\n");
        return 1;
    }
    for (i = 0; i < (*nnz); i++) {
        work[i].i = tempRowInd[i];
        work[i].j = tempColInd[i];
        work[i].p = i;  // permutation is identity
    }

    if (csrFormat) {
        /* create row-major ordering of indices (sorted by row and within each row by column) */
        qsort(work, *nnz, sizeof(struct cooFormat), (FUNPTR)fptr_array[0]);
    } else {
        /* create column-major ordering of indices (sorted by column and within each column by row)
         */
        qsort(work, *nnz, sizeof(struct cooFormat), (FUNPTR)fptr_array[1]);
    }

    // (tempRowInd, tempColInd) is sorted either by row-major or by col-major
    for (i = 0; i < (*nnz); i++) {
        tempRowInd[i] = work[i].i;
        tempColInd[i] = work[i].j;
    }

    // setup base
    // check if there is any row/col 0, if so base-0
    // check if there is any row/col equal to matrix dimension m/n, if so base-1
    int base0 = 0;
    int base1 = 0;
    for (i = 0; i < (*nnz); i++) {
        const int row = tempRowInd[i];
        const int col = tempColInd[i];
        if ((0 == row) || (0 == col)) {
            base0 = 1;
        }
        if ((*m == row) || (*n == col)) {
            base1 = 1;
        }
    }
    if (base0 && base1) {
        printf("Error: input matrix is base-0 and base-1 \n");
        return 1;
    }

    base = 0;
    if (base1) {
        base = 1;
    }

    /* compress the appropriate indices */
    if (csrFormat) {
        /* CSR format (assuming row-major format) */
        csrRowPtr = (int *)malloc(((*m) + 1) * sizeof(csrRowPtr[0]));
        if (!csrRowPtr) return 1;
        compress_index(tempRowInd, *nnz, *m, csrRowPtr, base);

        *aRowInd = csrRowPtr;
        *aColInd = (int *)malloc((*nnz) * sizeof(int));
    } else {
        /* CSC format (assuming column-major format) */
        cscColPtr = (int *)malloc(((*n) + 1) * sizeof(cscColPtr[0]));
        if (!cscColPtr) return 1;
        compress_index(tempColInd, *nnz, *n, cscColPtr, base);

        *aColInd = cscColPtr;
        *aRowInd = (int *)malloc((*nnz) * sizeof(int));
    }

    /* transfrom the matrix values of type double into one of the hipsparse library types */
    *aVal = (T_ELEM *)malloc((*nnz) * sizeof(T_ELEM));

    for (i = 0; i < (*nnz); i++) {
        if (csrFormat) {
            (*aColInd)[i] = tempColInd[i];
        } else {
            (*aRowInd)[i] = tempRowInd[i];
        }
        if (mm_is_real(matcode) || mm_is_integer(matcode)) {
            (*aVal)[i] = convertGet<T_ELEM>(tempVal[work[i].p]);
        } else {
            (*aVal)[i] = convertGet<T_ELEM>(tempVal[2 * work[i].p], tempVal[2 * work[i].p + 1]);
        }
    }

    /* check for corruption */
    int error_found;
    if (csrFormat) {
        error_found = verify_pattern(*m, *nnz, *aRowInd, *aColInd);
    } else {
        error_found = verify_pattern(*n, *nnz, *aColInd, *aRowInd);
    }
    if (error_found) {
        fprintf(stderr, "!!!! verify_pattern failed\n");
        return 1;
    }

    /* cleanup and exit */
    free(work);
    free(tempVal);
    free(tempColInd);
    free(tempRowInd);

    return 0;
}

// /* genTridiag: generate a random tridiagonal symmetric matrix */
// void genTridiag(int *I, int *J, real *val, int N, int nnz) {
//     I[0] = 0, J[0] = 0, J[1] = 1;
//     val[0] = (real)rand() / RAND_MAX + 10.0f;
//     val[1] = (real)rand() / RAND_MAX;
//     int start;

//     for (int i = 1; i < N; i++) {
//         if (i > 1) {
//             I[i] = I[i - 1] + 3;
//         } else {
//             I[1] = 2;
//         }

//         start = (i - 1) * 3 + 2;
//         J[start] = i - 1;
//         J[start + 1] = i;

//         if (i < N - 1) {
//             J[start + 2] = i + 1;
//         }

//         val[start] = val[start - 1];
//         val[start + 1] = (real)rand() / RAND_MAX + 10.0f;

//         if (i < N - 1) {
//             val[start + 2] = (real)rand() / RAND_MAX;
//         }
//     }

//     I[N] = nnz;
// }

// void report_errors(real *x_ref_single_gpu, real *x_ref_cpu, real *x, int row_start_idx, int row_end_idx,
//                    const bool compare_to_single_gpu, const bool compare_to_cpu, bool &result_correct_single_gpu,
//                    bool &result_correct_cpu) {
//     result_correct_single_gpu = true;
//     result_correct_cpu = true;

//     int i = 0;

//     if (compare_to_single_gpu) {
//         for (i = row_start_idx; result_correct_single_gpu && (i < row_end_idx); i++) {
//             if (std::fabs(x_ref_single_gpu[i] - x[i]) > tol || std::isnan(x[i]) || std::isnan(x_ref_single_gpu[i])) {
//                 fprintf(stderr,
//                         "ERROR: x[%d] = %.8f does not match %.8f "
//                         "(Single GPU reference)\n",
//                         i, x[i], x_ref_single_gpu[i]);

//                 result_correct_single_gpu = false;
//             }
//         }
//     }

//     if (compare_to_cpu) {
//         for (i = row_start_idx; result_correct_cpu && (i < row_end_idx); i++) {
//             if (std::fabs(x_ref_cpu[i] - x[i]) > tol || std::isnan(x[i]) || std::isnan(x_ref_cpu[i])) {
//                 fprintf(stderr,
//                         "ERROR: x[%d] = %.8f does not match %.8f "
//                         "(CPU reference)\n",
//                         i, x[i], x_ref_cpu[i]);

//                 result_correct_cpu = false;
//             }
//         }
//     }
// }

// void report_runtime(const int num_devices, const double single_gpu_runtime, const double start, const double stop,
//                     const bool result_correct_single_gpu, const bool result_correct_cpu,
//                     const bool compare_to_single_gpu) {
//     if (result_correct_single_gpu && result_correct_cpu) {
//         printf("Execution time: %8.4f ms\n", stop - start);

//         if (result_correct_single_gpu && result_correct_cpu && compare_to_single_gpu) {
//             printf(
//                 "Non-persistent kernel - 1 GPU: %8.4f ms, %d GPUs: %8.4f ms, speedup: %8.2f, "
//                 "efficiency: %8.2f \n",
//                 single_gpu_runtime, num_devices, stop - start, single_gpu_runtime / stop - start,
//                 single_gpu_runtime / (num_devices * stop - start) * 100);
//         }
//     }
// }

// // Single GPU kernels
// namespace CGComputeKernels {

// __device__ double alpha[1];
// __device__ double beta[1];
// __device__ double tmp_dot_delta0_p_c;

// __global__ void initVectors(real *r, real *x, int num_rows) {
//     cooperative_groups::grid_group grid = cooperative_groups::this_grid();

//     // Use Cooperative Groups to distribute the workload more evenly among threads
//     for (int idx = grid.thread_rank(); idx < num_rows; idx += grid.size()) {
//         r[idx] = 1.0;
//         x[idx] = 0.0;
//     }
// }

// __global__ void initVector(real *vec, int num_rows, double value) {
//     cooperative_groups::grid_group grid = cooperative_groups::this_grid();

//     for (int idx = grid.thread_rank(); idx < num_rows; idx += grid.size()) {
//         vec[idx] = value;
//     }
// }

// __global__ void gpuSpMV2(int *rowInd, int *colInd, double *val, double alpha, double *inputVecX, double *local_res,
//                          int chunk_size, int elm_displs) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int x, y;
//     double a;
//     if (idx < chunk_size) {
//         a = 0.0;
//         x = rowInd[idx];
//         y = rowInd[idx + 1];
//         for (int j = x; j < y; j++) {
//             a += alpha * val[j - elm_displs] * inputVecX[colInd[j - elm_displs]];
//         }
//         local_res[idx] = a;
//     }
// }

// __global__ void gpuSpMV(int *rowInd, int *colInd, double *val, double alpha, double *inputVecX, double *local_res,
//                         int chunk_size, int elm_displs) {
//     cooperative_groups::grid_group grid = cooperative_groups::this_grid();

//     for (int idx = grid.thread_rank(); idx < chunk_size; idx += grid.size()) {
//         double a = 0.0;
//         int x = rowInd[idx];
//         int y = rowInd[idx + 1];
//         for (int j = x; j < y; j++) {
//             a += alpha * val[j - elm_displs] * inputVecX[colInd[j - elm_displs]];
//         }
//         local_res[idx] = a;
//     }
// }

// __global__ void gpuSaxpy(real *x, real *y, int n) {
//     cooperative_groups::grid_group grid = cooperative_groups::this_grid();

//     for (int i = grid.thread_rank(); i < n; i += grid.size()) {
//         y[i] = alpha[0] * x[i] + y[i];
//     }
// }

// __global__ void gpuSaxpyNegative(real *x, real *y, int n) {
//     cooperative_groups::grid_group grid = cooperative_groups::this_grid();

//     for (int i = grid.thread_rank(); i < n; i += grid.size()) {
//         y[i] = -alpha[0] * x[i] + y[i];
//     }
// }

// __global__ void gpuCopyVector2(real *srcA, real *destB, int chunk_size) {
//     int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

//     if (grid_rank < chunk_size) {
//         destB[grid_rank] = srcA[grid_rank];
//     }
// }

// __global__ void gpuCopyVector(real *srcA, real *destB, int chunk_size) {
//     cooperative_groups::grid_group grid = cooperative_groups::this_grid();

//     for (int i = grid.thread_rank(); i < chunk_size; i += grid.size()) {
//         destB[i] = srcA[i];
//     }
// }

// __global__ void gpuScaleVectorAndSaxpy(real *x, real *y, real scale, int chunk_size) {
//     int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

//     if (grid_rank < chunk_size) {
//         y[grid_rank] = beta[0] * x[grid_rank] + scale * y[grid_rank];
//     }
// }

// __global__ void resetLocalDotProduct(double *dot_result) {
//     int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

//     if (grid_rank == 0) {
//         *dot_result = 0.0;
//     }
// }

// template <unsigned int blockSize>
// __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
//     if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//     if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//     if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//     if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//     if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//     if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
// }

// __global__ void gpuDotProduct(real *vecA, real *vecB, double *local_dot_result, int chunk_size) {
//     // extern __shared__ double sdata[];
//     // unsigned int tid = threadIdx.x;
//     // unsigned int i = blockIdx.x*(blockSize*2) + tid;
//     // unsigned int gridSize = blockSize*2*gridDim.x;
//     // sdata[tid] = 0.0;
//     // while (i < chunk_size) { sdata[tid] += vecA[i] * vecB[i]+ vecA[i+blockSize] * vecB[i+blockSize]; i += gridSize; }
//     // __syncthreads();
//     // if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//     // if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//     // if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//     // if (tid < 32) warpReduce<blockSize>(sdata, tid);
//     // if (tid == 0) local_dot_result[blockIdx.x] = sdata[0];
// }

// __global__ void computeAlpha(double *tmp, double *dot) { alpha[0] = *tmp / *dot; }

// __global__ void computeBeta(double *tmp, double *dot) {
//     beta[0] = *tmp / *dot;
//     *tmp = *dot;
// }

// __global__ void resetLocalDotProducts(double *dot_result_delta, double *dot_result_gamma) {
//     int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

//     if (grid_rank == 0) {
//         *dot_result_delta = 0.0;
//         *dot_result_gamma = 0.0;
//     }
// }

// __global__ void gpuDotProductsMerged(real *vecA_delta, real *vecB_delta, real *vecA_gamma, real *vecB_gamma,
//                                      double *local_dot_result_delta, double *local_dot_result_gamma, int chunk_size,
//                                      const int sMemSize) {
//     // cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

//     // int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

//     // extern __shared__ double tmp[];

//     // double *tmp_delta = (double *)tmp;
//     // double *tmp_gamma = (double *)&tmp_delta[sMemSize / (2 * sizeof(double))];

//     // double temp_sum_delta = 0.0;
//     // double temp_sum_gamma = 0.0;

//     // if (grid_rank < chunk_size) {
//     //     temp_sum_delta += (double)(vecA_delta[grid_rank] * vecB_delta[grid_rank]);
//     //     temp_sum_gamma += (double)(vecA_gamma[grid_rank] * vecB_gamma[grid_rank]);
//     // }

//     // cooperative_groups::thread_block_tile<32> tile32 = cooperative_groups::tiled_partition<32>(cta);

//     // temp_sum_delta = cooperative_groups::reduce(tile32, temp_sum_delta, cooperative_groups::plus<double>());
//     // temp_sum_gamma = cooperative_groups::reduce(tile32, temp_sum_gamma, cooperative_groups::plus<double>());

//     // if (tile32.thread_rank() == 0) {
//     //     tmp_delta[tile32.meta_group_rank()] = temp_sum_delta;
//     //     tmp_gamma[tile32.meta_group_rank()] = temp_sum_gamma;
//     // }

//     // cooperative_groups::sync(cta);

//     // if (tile32.meta_group_rank() == 0) {
//     //     temp_sum_delta = tile32.thread_rank() < tile32.meta_group_size() ? tmp_delta[tile32.thread_rank()] : 0.0;
//     //     temp_sum_delta = cooperative_groups::reduce(tile32, temp_sum_delta, cooperative_groups::plus<double>());

//     //     temp_sum_gamma = tile32.thread_rank() < tile32.meta_group_size() ? tmp_gamma[tile32.thread_rank()] : 0.0;
//     //     temp_sum_gamma = cooperative_groups::reduce(tile32, temp_sum_gamma, cooperative_groups::plus<double>());

//     //     if (tile32.thread_rank() == 0) {
//     //         atomicAdd(local_dot_result_delta, temp_sum_delta);
//     //         atomicAdd(local_dot_result_gamma, temp_sum_gamma);
//     //     }
//     // }
// }

// __global__ void computeAlpha_Beta(double *device_merged_dots, int k) {
//     double real_tmp_dot_delta1 = (double)device_merged_dots[0];
//     double real_tmp_dot_gamma1 = (double)device_merged_dots[1];

//     if (k > 1) {
//         beta[0] = real_tmp_dot_delta1 / tmp_dot_delta0_p_c;
//         alpha[0] = real_tmp_dot_delta1 / (real_tmp_dot_gamma1 - (beta[0] / alpha[0]) * real_tmp_dot_delta1);
//     } else {
//         beta[0] = 0.0;
//         alpha[0] = real_tmp_dot_delta1 / real_tmp_dot_gamma1;
//     }

//     tmp_dot_delta0_p_c = real_tmp_dot_delta1;
// }
// }  // namespace CGComputeKernels

#endif  // __UNICONN_EXAMPLES_CG_INCLUDE_COMMON_HPP_