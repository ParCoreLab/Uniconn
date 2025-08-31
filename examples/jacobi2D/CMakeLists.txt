set(MPI_SOURCES
  src/single_stream_mpi.cu
)

set(CCL_SOURCES
  src/single_stream_gpuccl.cu
)
set(SHMEM_SOURCES
  src/single_stream_nvshmem_host.cu
  src/single_stream_nvshmem_device.cu
)
set(UNC_SOURCES
  src/single_stream_uniconn.cu
)


if(UNC_HAS_CUDA)
set_source_files_properties(${MPI_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(${CCL_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(${SHMEM_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(${UNC_SOURCES} PROPERTIES LANGUAGE CUDA)
elseif(UNC_HAS_ROCM)
  set_source_files_properties(${MPI_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${CCL_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${SHMEM_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${UNC_SOURCES} PROPERTIES LANGUAGE HIP)
endif(UNC_HAS_CUDA)


macro(configure_target target_name)
  # set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  if(UNC_HAS_CUDA)
    target_compile_features(${target_name} PUBLIC cuda_std_17)
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
        
        --maxrregcount=32
        >)
  elseif(UNC_HAS_ROCM)
    set_source_files_properties(${target_name}
        PROPERTIES LANGUAGE HIP)
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
      --gpu-max-threads-per-block=1024
    >)
    set_target_properties(${target_name}
        PROPERTIES
        CXX_single_stream 17
        CXX_single_stream_REQUIRED TRUE)
  endif(UNC_HAS_CUDA)

  target_link_libraries(${target_name} PRIVATE Uniconn)
endmacro(configure_target)

string(REPLACE ".cu" "" _tmp_exe_name "src/single_stream_mpi.cu")
get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
add_executable(${_exe_name} src/single_stream_mpi.cu)
configure_target(${_exe_name})

if(UNC_HAS_GPUCCL)
  string(REPLACE ".cu" "" _tmp_exe_name "src/single_stream_gpuccl.cu")
  get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
  add_executable(${_exe_name} src/single_stream_gpuccl.cu)
  configure_target(${_exe_name})

endif(UNC_HAS_GPUCCL)

if(UNC_HAS_GPUSHMEM)
  string(REPLACE ".cu" "" _tmp_exe_name "src/single_stream_nvshmem_host.cu")
  get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
  add_executable(${_exe_name} src/single_stream_nvshmem_host.cu)
  configure_target(${_exe_name})

  string(REPLACE ".cu" "" _tmp_exe_name "src/single_stream_nvshmem_device.cu")
  get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
  add_executable(${_exe_name} src/single_stream_nvshmem_device.cu)
  configure_target(${_exe_name})
endif(UNC_HAS_GPUSHMEM)


string(REPLACE ".cu" "" _tmp_exe_name "src/single_stream_uniconn.cu")
get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
string(CONCAT _exe_name_mpi "${_exe_name}" "_mpi")
add_executable(${_exe_name_mpi} "src/single_stream_uniconn.cu")
target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_MPI=1)
target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_GPUCCL=0)
target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_GPUSHMEM=0)
target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_HOST_ONLY=1)
target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_LIMITED_DEVICE=0)
target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_FULL_DEVICE=0)
configure_target(${_exe_name_mpi})

if(UNC_HAS_GPUCCL)
  string(CONCAT _exe_name_gpuccl "${_exe_name}" "_gpuccl")
  add_executable(${_exe_name_gpuccl} "src/single_stream_uniconn.cu")
  target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_MPI=0)
  target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_GPUCCL=1)
  target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_GPUSHMEM=0)
  target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_HOST_ONLY=1)
  target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_LIMITED_DEVICE=0)
  target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_FULL_DEVICE=0)
  configure_target(${_exe_name_gpuccl})
endif(UNC_HAS_GPUCCL)

if(UNC_HAS_GPUSHMEM)
  string(CONCAT _exe_name_nvshmem "${_exe_name}" "_nvshmem_host")
  add_executable(${_exe_name_nvshmem} "src/single_stream_uniconn.cu")
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_MPI=0)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_GPUCCL=0)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_GPUSHMEM=1)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_HOST_ONLY=1)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_LIMITED_DEVICE=0)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_FULL_DEVICE=0)
  configure_target(${_exe_name_nvshmem})
  string(CONCAT _exe_name_nvshmem "${_exe_name}" "_nvshmem_device")
  add_executable(${_exe_name_nvshmem} "src/single_stream_uniconn.cu")
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_MPI=0)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_GPUCCL=0)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_GPUSHMEM=1)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_HOST_ONLY=0)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_LIMITED_DEVICE=0)
  target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_FULL_DEVICE=1)
  configure_target(${_exe_name_nvshmem})
endif(UNC_HAS_GPUSHMEM)

  